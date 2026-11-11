#include "src/models/qwen36_35b_a3b/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/component_cache_resources.hpp"
#include "gufo/ext/gguf_identity.hpp"
#include "gufo/ext/host_budget.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/image_cache.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen36_35b_a3b/continuation_adapter.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"
#include "src/models/qwen36_35b_a3b/probe.hpp"

namespace gufo::models::qwen36_35b_a3b {
namespace {

using namespace gufo::ext;

constexpr std::string_view kName = "Qwen3.6-35B-A3B";
constexpr std::string_view kShortName = "Qwen3.6";
constexpr std::string_view kStateAbi = "qwen36-35b-a3b-rocm-session-v1";
constexpr auto kTemplateFamily = tokenization::QwenTemplateFamily::kQwen36;

[[noreturn]] void Fail(std::string_view what) {
  throw std::runtime_error(std::string(kName) + " " + std::string(what));
}

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// Qwen3.6-35B-A3B carries its MTP block in the main artifact, so the
// artifact identity covers the draft weights too. Keep this text stable:
// it names the persistent disk cache entries.
std::vector<std::uint8_t> Qwen36CompatibilityIdentity(
    std::string_view artifact_fingerprint, bool has_mtp,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.6-35B-A3B disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen36-35b-a3b\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen36-compiled-v1\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::Qwen36OfficialTemplateSha256()
           << '\n'
           << "state_abi=" << kStateAbi << '\n'
           << "payload_layout=q36-rocm-session-snapshot-v"
           << Session::kSnapshotPayloadVersion << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=q36-native-mtp-v1\n"
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

// Without a disk cache, checkpoints live only as long as this loaded model.
std::vector<std::uint8_t> Qwen36ProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::uint32_t decode_concurrency) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_kind=qwen36-35b-a3b\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "mtp=" << use_mtp << '\n'
           << "draft_max_tokens=" << max_draft_tokens << '\n'
           << "draft_cost_concurrency=" << decode_concurrency << '\n';
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

[[nodiscard]] hip::ComponentCacheHooks HipComponentCacheHooks() {
  return {[] { return ext::HostSnapshotCeilingBytes(); },
          [](std::string_view) {}, [](std::string_view) {}};
}

std::vector<std::int32_t> EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(std::string(kName) +
                                  " token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

std::span<const tokenization::ChatTool> RequestTools(
    const ChatRequest& request) {
  return request.tool_choice == ChatRequest::ToolChoice::kNone
             ? std::span<const tokenization::ChatTool>{}
             : std::span<const tokenization::ChatTool>{request.tools};
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(
      request.reasoning, request.add_vision_id, kTemplateFamily);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

// The text before tokenization. An image renders as its one placeholder, not
// the visual tokens Prepare expands it into. Template errors are request
// errors, as in Prepare.
std::optional<std::string> RenderQwenPrompt(const ChatRequest& request,
                                            std::uint32_t max_context) {
  std::string error;
  auto rendered = tokenization::QwenChatTemplate::Render(
      request.messages, RequestTools(request),
      QwenChatOptions(request, max_context), &error);
  if (!rendered)
    throw std::invalid_argument(error);
  return rendered;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context);
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, RequestTools(request), options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, RequestTools(request), options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages, RequestTools(request),
          QwenChatOptions(request, max_context),
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

// Each cache slot owns one session's recurrent/KV state; target projections
// share batches, and sampling and rollback stay request-local. The adapter
// guards checkpoint rows before every write.
class Qwen36TextRunnerState final : public TextRunnerState {
public:
  Qwen36TextRunnerState(ContinuationAdapter& adapter, cache::Slot& slot)
      : adapter_(&adapter), slot_(&slot) {
    session_ = &adapter.GetSession(slot);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  /// Read-only access, settling the slot's transfers.
  [[nodiscard]] Session& session() const {
    return adapter_->GetSession(*slot_);
  }
  /// Access for an operation that writes state from the current frontier.
  [[nodiscard]] Session& Mutate(bool restart = false) const {
    adapter_->BeforeMutation(*slot_, restart);
    return *session_;
  }
  void ConfigureVision(
      std::shared_ptr<const models::qwen::vision::Prompt> prompt) const {
    adapter_->ConfigureVision(*slot_, std::move(prompt));
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  ContinuationAdapter* adapter_;
  cache::Slot* slot_;
  Session* session_;
  std::size_t position_{0};
};

Qwen36TextRunnerState& RequireState(TextRunnerState& state) {
  auto* q36 = dynamic_cast<Qwen36TextRunnerState*>(&state);
  if (q36 == nullptr)
    throw std::logic_error("text runner state is not " + std::string(kName));
  return *q36;
}

const Qwen36TextRunnerState& RequireState(const TextRunnerState& state) {
  const auto* q36 = dynamic_cast<const Qwen36TextRunnerState*>(&state);
  if (q36 == nullptr)
    throw std::logic_error("text runner state is not " + std::string(kName));
  return *q36;
}

// Arms the model's layer hook from the host's yield for one prefill step.
class PrefillYieldScope {
public:
  PrefillYieldScope(const Model& model, const TextRunnerState& state)
      : model_(model) {
    if (state.prefill_yield_groups() > 1)
      model_.SetPrefillYield(state.prefill_yield_groups(),
                             [&state] { state.YieldPrefill(); });
  }
  ~PrefillYieldScope() { model_.SetPrefillYield(1, {}); }
  PrefillYieldScope(const PrefillYieldScope&) = delete;
  PrefillYieldScope& operator=(const PrefillYieldScope&) = delete;

private:
  const Model& model_;
};

// Beside decoders, prompts keep the 2,048-row pass (about 0.85 s) and split it
// into layer groups; a decode step takes about 16 ms.
constexpr std::size_t kMaxPrefillLayerGroups = 8;
constexpr std::size_t kMaxBatchWidth = 8;

class Qwen36TextRunner final : public HipTextModelRunner {
public:
  Qwen36TextRunner(std::shared_ptr<Model> model, std::uint32_t max_context,
                   bool use_mtp, std::uint32_t max_draft_tokens,
                   std::string artifact_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    // Durable caches need the full artifact fingerprint; an in-process
    // cache is bound to this loaded model.
    if (!artifact_fingerprint.empty()) {
      component_identity_ = Qwen36CompatibilityIdentity(
          artifact_fingerprint, use_mtp_, max_context_, max_draft_tokens_,
          model_->DecodeConcurrency());
    } else {
      component_identity_ =
          Qwen36ProcessIdentity(model_.get(), max_context_, use_mtp_,
                                max_draft_tokens_, model_->DecodeConcurrency());
    }
  }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    return hip::CreateHipComponentCacheResources(
        std::make_unique<ContinuationAdapter>(
            model_,
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_, component_identity_),
        ledger, HipComponentCacheHooks());
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<Qwen36TextRunnerState>(
        dynamic_cast<ContinuationAdapter&>(adapter), slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireState(state).set_position(position);
  }

  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .supports_vision = model_->VisionEncoder() != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
            },
        .sampling_defaults = Qwen36SamplingDefaults(),
        .scheduling =
            TextSchedulingProfile{
                .pass_tokens = model_->PrefillCapacity(),
                .max_layer_groups = kMaxPrefillLayerGroups,
                .max_yield_decode_width = kMaxBatchWidth,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Snapshots live in host memory, not in the device state pool.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages, RequestTools(request),
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderQwenPrompt(request, max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireState(state).ConfigureVision(QwenPrompt(context));
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& q36 = RequireState(state);
    if (q36.position() != prefix.size()) {
      throw std::logic_error(std::string(kName) +
                             " reused prefix does not match checkpoint");
    }
    q36.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& q36 = RequireState(state);
    if (offset != q36.position()) {
      throw std::logic_error(std::string(kName) +
                             " prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(std::string(kName) +
                             " prefill has no remaining input");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
    const auto held = q36.session().Tokens();
    const bool restart =
        !q36.session().IsValid() || held.size() > prefix.size() ||
        !std::ranges::equal(held, std::span(prefix).first(held.size()));
    std::string error;
    auto& session = q36.Mutate(restart);
    const PrefillYieldScope yield(*model_, state);
    if (!session.Sync(prefix, &error)) {
      q36.set_position(0);
      Fail("prefill failed: " + error);
    }
    q36.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& q36 = RequireState(state);
    if (q36.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = q36.session().Logits();
    if (logits.empty())
      Fail("token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    const bool stop = q36.stop_at_eos() && model_->IsStopToken(token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : model_->TokenText(token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(std::string(kName) +
                                  " token ID exceeds engine range");
    }
    auto& q36 = RequireState(state);
    std::string error;
    if (!q36.Mutate().Evaluate(static_cast<std::int32_t>(token), &error))
      Fail("decode failed: " + error);
    q36.set_position(q36.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          std::string(kName) + " MTP decode budget must be at least one token");
    }
    auto& q36 = RequireState(state);
    if (q36.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = q36.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    Session::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    if (!q36.Mutate().DecodeStep(budget, working_sampler, &decoded, &error,
                                  q36.stop_at_eos())) {
      Fail("MTP decode failed: " + error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    q36.set_position(q36.session().Position());
    const auto stats_after = q36.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<Session::BatchOutcome> outcomes(advances.size());
    std::vector<Session::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(kShortName) +
                                    " token exceeds engine range");
      }
      requests.push_back({&RequireState(advance.state.get()).Mutate(),
                          static_cast<std::int32_t>(advance.token),
                          &outcomes[requests.size()]});
    }
    std::string error;
    (void)Session::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(
            std::runtime_error(std::string(kShortName) +
                               " advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& q36 = RequireState(advance.state.get());
      q36.set_position(q36.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<Session::DecodeResult> results(count);
    std::vector<Session::BatchOutcome> outcomes(count);
    std::vector<Session::SpeculativeStats> before;
    std::vector<Session::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& q36 = RequireState(decodes[i].state.get());
      auto& session = q36.Mutate();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], q36.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)Session::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error(std::string(kShortName) +
                               " MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& q36 = RequireState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      q36.set_position(q36.session().Position());
      auto& step = steps[i];
      step.stop = results[i].stop;
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      for (const auto token : results[i].tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      const auto stats = q36.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireState(state).position();
  }

private:
  std::shared_ptr<Model> model_;
  mutable models::qwen::vision::ImageCache image_cache_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  cache::Identity component_identity_;
};

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config) {
  return std::make_shared<Qwen36TextRunner>(
      std::move(model), max_context,
      speculative_config.backend == TextSpeculativeBackend::kMtp,
      speculative_config.max_draft_tokens);
}

std::shared_ptr<ext::TextModelRunner> CreateQwen36TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string open_error;
    auto owner = core::GgufReader::OpenFile(options.model_path, &open_error);
    if (!owner) {
      SetError(error, "Failed to open GGUF: " + open_error);
      return nullptr;
    }
    reader = std::shared_ptr<const core::GgufReader>(std::move(owner));
  }
  if (options.enable_yarn) {
    SetError(error, "--enable-yarn applies only to Qwen3.8-Flash-Next");
    return nullptr;
  }
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    const std::uint64_t native =
        reader->GetMetadataUint64("qwen35moe.context_length").value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error,
               "GGUF has no valid native context length; specify --context");
      return nullptr;
    }
    max_context = static_cast<std::uint32_t>(native);
  }
  const auto& speculative = options.speculative;
  const bool use_mtp = speculative.backend == TextSpeculativeBackend::kMtp;
  if (speculative.backend != TextSpeculativeBackend::kDisabled && !use_mtp) {
    SetError(error,
             "Qwen3.6-35B-A3B HTTP models support only MTP speculative "
             "decoding (--speculative mtp)");
    return nullptr;
  }
  if (use_mtp && !speculative.draft_model_path.empty()) {
    SetError(error,
             "Qwen3.6-35B-A3B MTP is native to the model artifact; omit "
             "--mtp-model");
    return nullptr;
  }
  if (use_mtp &&
      (speculative.max_draft_tokens == 0 || speculative.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.6-35B-A3B MTP drafts a confidence-gated chain; custom "
             "draft floors are unsupported");
    return nullptr;
  }
  if (options.session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return nullptr;
  }
  std::string load_error;
  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            &load_error)) {
    SetError(error, "Unsupported Qwen3.6-35B-A3B chat template: " + load_error);
    return nullptr;
  }
  auto model = Model::Load(
      options.model_path,
      ModelOptions{
          .max_context = max_context,
          .mtp = use_mtp,
          .max_draft_tokens = speculative.max_draft_tokens,
          .vision_model_path = options.vision_model_path,
          .decode_concurrency = static_cast<std::uint32_t>(
              std::clamp<std::size_t>(options.session_count, 1, 8)),
      },
      &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create Qwen3.6-35B-A3B model: " + load_error);
    return nullptr;
  }
  if (max_context > model->MaxContext()) {
    SetError(error,
             "HTTP context exceeds the loaded Qwen3.6-35B-A3B model context");
    return nullptr;
  }
  if (use_mtp && !model->HasMtp()) {
    SetError(error,
             "Qwen3.6-35B-A3B MTP requires a model loaded with its MTP block");
    return nullptr;
  }
  try {
    qwen::vision::PrepareForServing(model->VisionEncoder());
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return nullptr;
  }
  // Only the disk tier needs the full artifact fingerprint, which reads every
  // byte.
  std::string fingerprint;
  if (!options.disk_cache.directory.empty()) {
    fingerprint = options.disk_cache.model_artifact_fingerprint;
    try {
      if (fingerprint.empty())
        fingerprint = core::GgufIdentityHex(*reader);
      (void)Qwen36CompatibilityIdentity(fingerprint, use_mtp, max_context,
                                        speculative.max_draft_tokens,
                                        model->DecodeConcurrency());
    } catch (const std::exception& e) {
      SetError(error,
               std::string("Qwen3.6-35B-A3B disk cache fingerprinting failed: ") +
                   e.what());
      return nullptr;
    }
  }
  return std::make_shared<Qwen36TextRunner>(std::move(model), max_context,
                                            use_mtp,
                                            speculative.max_draft_tokens,
                                            std::move(fingerprint));
}

}  // namespace gufo::models::qwen36_35b_a3b

#endif  // ENGINE_ENABLE_HIP
