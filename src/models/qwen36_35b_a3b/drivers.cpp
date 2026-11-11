#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "gufo/ext/text.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen36_35b_a3b/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace gufo::models::qwen36_35b_a3b {
namespace {

constexpr std::size_t kDefaultContext = 4096;
constexpr std::string_view kName = "Qwen3.6-35B-A3B";
constexpr auto kTemplateFamily = tokenization::QwenTemplateFamily::kQwen36;

struct BenchStats {
  double mean = 0.0;
  double stddev = 0.0;
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty())
    return {};
  if (values.size() == 1)
    return {.mean = values.front(), .stddev = 0.0};
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  double sq_sum = 0.0;
  for (const double val : values) {
    sq_sum += (val - mean) * (val - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  if (start == std::chrono::steady_clock::time_point{})
    return;
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  std::cerr << "Model loaded in " << ms << " ms ("
            << (success ? "success" : "failed") << ").\n";
}

void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

std::optional<gufo::ReasoningEffort> ParseReasoningEffort(
    std::string_view value) {
  if (value == "minimal")
    return gufo::ReasoningEffort::kMinimal;
  if (value == "low")
    return gufo::ReasoningEffort::kLow;
  if (value == "medium")
    return gufo::ReasoningEffort::kMedium;
  if (value == "high")
    return gufo::ReasoningEffort::kHigh;
  if (value == "xhigh")
    return gufo::ReasoningEffort::kXHigh;
  if (value == "max")
    return gufo::ReasoningEffort::kMax;
  return std::nullopt;
}

gufo::ReasoningOptions PromptReasoningOptions(
    const ext::TextPromptOptions& options) {
  gufo::ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

tokenization::ChatTemplateOptions PromptChatOptions(
    const ext::TextPromptOptions& opt) {
  return tokenization::ResolveQwenChatOptions(PromptReasoningOptions(opt),
                                              opt.add_vision_id,
                                              kTemplateFamily);
}

std::shared_ptr<Model> LoadPromptModel(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  if (opt.force_cpu ||
      (!opt.speculative_backend.empty() && opt.speculative_backend != "mtp") ||
      (opt.speculative_backend == "mtp" && opt.min_draft_tokens != 1)) {
    std::cerr << "Qwen3.6 requires ROCm and supports native MTP with "
                 "--min-draft-tokens 1\n";
    return nullptr;
  }
  if (!opt.mtp_model_path.empty() || !opt.vision_model_path.empty() ||
      !opt.image_paths.empty()) {
    std::cerr << "Qwen3.6 prompt is text-only and uses its native MTP block; "
                 "remove --mtp-model, --mmproj and --image\n";
    return nullptr;
  }
  if (opt.use_chat_template &&
      !tokenization::QwenChatTemplate::ValidateGgufTemplate(reader, &error)) {
    std::cerr << "Unsupported Qwen3.6 chat template: " << error << '\n';
    return nullptr;
  }
  auto model = Model::Load(opt.model_path,
                           {.max_context = kDefaultContext,
                            .mtp = opt.speculative_backend == "mtp",
                            .max_draft_tokens = opt.draft_tokens,
                            .vision_model_path = {},
                            .decode_concurrency = 1},
                           &error);
  PrintModelLoadTime(load_start, model != nullptr);
  if (!model)
    std::cerr << "Qwen3.6 load failed: " << error << '\n';
  return model;
}

std::unique_ptr<Session> CreatePromptSession(const ext::TextPromptOptions& opt,
                                             Model& model) {
  std::string error;
  auto session = model.CreateSession(
      opt.speculative_backend.empty() ? core::SessionMode::kAutoregressive
                                      : core::SessionMode::kSpeculative,
      kDefaultContext, &error);
  if (!session)
    std::cerr << "Qwen3.6 session failed: " << error << '\n';
  return session;
}

int GenerateResponse(const ext::TextPromptOptions& opt, const Model& model,
                     Session& session,
                     std::span<const tokenization::TokenId> prompt,
                     std::string* reply = nullptr) {
  if (prompt.empty() || prompt.size() >= session.ContextSize() ||
      opt.max_tokens > session.ContextSize() - prompt.size()) {
    std::cerr << "Qwen3.6 prompt and output exceed the 4096-token CLI context\n";
    return 1;
  }
  const std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  std::string error;
  if (!session.Sync(input, &error)) {
    std::cerr << "Qwen3.6 prefill failed: " << error << '\n';
    return 1;
  }
  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<tokenization::TokenId> generated;
  const auto stats_before = session.Statistics();
  const auto start = std::chrono::steady_clock::now();
  while (generated.size() < opt.max_tokens) {
    Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated.size(), sampler,
                            &decoded, &error)) {
      std::cerr << "Qwen3.6 decode failed: " << error << '\n';
      return 1;
    }
    for (const auto token : decoded.tokens) {
      const auto piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply)
        reply->append(piece);
      generated.push_back(static_cast<tokenization::TokenId>(token));
    }
    if (decoded.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cerr << "Generated " << generated.size() << " tokens ("
              << generated.size() / seconds << " tok/s)\n";
    const auto& stats = session.Statistics();
    if (stats.cycles != stats_before.cycles) {
      std::cerr << "[Speculative]: cycles="
                << stats.cycles - stats_before.cycles
                << " drafted=" << stats.drafted - stats_before.drafted
                << " accepted=" << stats.accepted - stats_before.accepted
                << '\n';
    }
    PrintTokenTrace(generated);
  }
  return 0;
}

std::size_t BenchContext(const ext::TextBenchOptions& options) {
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  return std::max(
      {std::size_t{4096}, max_depth + max_or_zero(options.n_prompts) + 1,
       std::max<std::size_t>(max_depth, 16) + max_or_zero(options.n_gens) + 1,
       options.validate_prefill_tokens + 1});
}

int RunBenchmarkImpl(const ext::TextBenchOptions& options,
                     const std::shared_ptr<const core::GgufReader>& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for " << kName << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  if (std::max({max_or_zero(options.n_depths), max_or_zero(options.n_prompts),
                max_or_zero(options.n_gens), options.validate_prefill_tokens}) >
      std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }
  const std::size_t required_context = BenchContext(options);
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: " << kName
              << " supports only --speculative mtp or off\n";
    return 1;
  }
  if (!options.mtp_model_path.empty()) {
    std::cerr << "Error: MTP is native to the artifact; remove --mtp-model\n";
    return 1;
  }
  if (options.enable_yarn) {
    std::cerr << "Error: --enable-yarn applies only to Qwen3.8-Flash-Next\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: " << kName
              << " bench supports C1; use the serving benchmark for "
                 "concurrent requests\n";
    return 1;
  }
  if (mtp && options.min_draft_tokens != 1) {
    std::cerr << "Error: " << kName << " MTP requires --min-draft-tokens 1\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: " << kName << " context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = Model::Load(
      options.model_path,
      {.max_context = static_cast<std::uint32_t>(required_context),
       .mtp = mtp,
       .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
       .vision_model_path = {},
       .decode_concurrency = 1},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating " << kName << " model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);
  const auto session_mode = options.speculative_backend.empty()
                                ? core::SessionMode::kAutoregressive
                                : core::SessionMode::kSpeculative;

  // Repeat a fixed token pattern for reproducible timing.
  std::vector<std::int32_t> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = pattern[i % pattern.size()];
    }
  }

  if (options.validate_prefill_tokens != 0) {
    // Batched prefill against one-token-at-a-time evaluation of the same
    // prefix; both run on the GPU, so this checks the batched kernels.
    auto sequential = model->CreateSession(
        session_mode, static_cast<std::uint32_t>(required_context), &error);
    auto batched = model->CreateSession(
        session_mode, static_cast<std::uint32_t>(required_context), &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << kName << " prefill validation failed: " << error << '\n';
      return 1;
    }
    for (const std::int32_t token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << kName << " sequential reference failed: " << error
                  << '\n';
        return 1;
      }
    }
    const auto reference = sequential->Logits();
    const auto candidate = batched->Logits();
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      // A full aligned prefix already warms the same chunk shapes and GEMM
      // plans. Other shapes still need an untimed pass before measurement.
      const auto chunk = model->PrefillCapacity();
      const bool warm_prefix = chunk > 0 && depth >= chunk &&
                               depth % chunk == 0 && prompt_length > 0 &&
                               prompt_length % chunk == 0;
      for (std::size_t repetition = warm_prefix ? 1 : 0;
           repetition <= options.repetitions; ++repetition) {
        auto session = model->CreateSession(
            session_mode, static_cast<std::uint32_t>(required_context),
            &error);
        if (!session ||
            (depth > 0 &&
             !session->Sync(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            session_mode, static_cast<std::uint32_t>(required_context),
            &error);
        if (!session ||
            !session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        const std::vector<sampling::TokenId> history(
            tokens.begin(), tokens.begin() + prefix_length);
        sampling::SamplerState sampler(options.sampling, history);
        const auto start = std::chrono::steady_clock::now();
        while (generated.size() < generation_length) {
          Session::DecodeResult decoded;
          if (!session->DecodeStep(generation_length - generated.size(),
                                   sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running " << kName << " decode: " << error
                      << '\n';
            return 1;
          }
          generated.insert(generated.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          const auto stats = session->Statistics();
          std::cerr << kName << " tg depth=" << depth
                    << " cycles=" << stats.cycles
                    << " drafted=" << stats.drafted
                    << " accepted=" << stats.accepted << " output_sha256="
                    << crypto::Sha256Hex(
                           std::span(reinterpret_cast<const std::uint8_t*>(
                                         generated.data()),
                                     generated.size() * sizeof(std::int32_t)))
                    << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

int RunPromptImpl(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start) {
  std::string rendered_prompt = opt.prompt_text;
  std::vector<tokenization::ContentSpan> content_spans;
  if (opt.use_chat_template) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto rendered = tokenization::QwenChatTemplate::Render(
        messages, {}, PromptChatOptions(opt), nullptr, nullptr, nullptr,
        &content_spans);
    if (!rendered.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
    rendered_prompt = *rendered;
  }

  auto model = LoadPromptModel(opt, reader, model_load_start);
  if (!model)
    return 1;
  auto session = CreatePromptSession(opt, *model);
  if (!session)
    return 1;
  try {
    const auto ids = tokenization::QwenChatTemplate::EncodeRendered(
        model->tokenizer(), rendered_prompt, 0, rendered_prompt.size(),
        content_spans, {});
    const std::vector<tokenization::TokenId> prompt(ids.begin(), ids.end());
    return GenerateResponse(opt, *model, *session, prompt);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

int RunChatImpl(const ext::TextPromptOptions& opt,
                const core::GgufReader& reader,
                std::chrono::steady_clock::time_point model_load_start) {
  auto model = LoadPromptModel(opt, reader, model_load_start);
  if (!model)
    return 1;
  auto session = CreatePromptSession(opt, *model);
  if (!session)
    return 1;
  const auto& tokenizer = model->tokenizer();

  std::cout << "=== Gufo Interactive Chat (qwen35moe) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  const auto chat_options = PromptChatOptions(opt);
  std::string user_input;
  while (true) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    std::vector<tokenization::ContentSpan> content_spans;
    const auto rendered_prompt = tokenization::QwenChatTemplate::Render(
        history, {}, chat_options, nullptr, nullptr, nullptr, &content_spans);
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
    const auto prompt_tokens = tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, *rendered_prompt, 0, rendered_prompt->size(), content_spans,
        {});

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;
    try {
      if (GenerateResponse(opt, *model, *session, prompt_tokens,
                           &assistant_reply) != 0)
        return 1;
    } catch (const std::exception& exception) {
      std::cerr << "Generation failed: " << exception.what() << '\n';
      return 1;
    }

    std::cout << '\n';
    tokenization::ChatMessage reply{tokenization::ChatRole::kAssistant, ""};
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = assistant_reply.find(end);
      reply.thought = assistant_reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        reply.content = assistant_reply.substr(boundary + end.size());
      }
    } else {
      reply.content = std::move(assistant_reply);
    }
    history.push_back(std::move(reply));
  }
  return 0;
}

}  // namespace

int RunQwen36Benchmark(const ext::TextBenchOptions& opt,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start) {
  return RunBenchmarkImpl(opt, reader, model_load_start);
}

int RunQwen36Prompt(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start) {
  return RunPromptImpl(opt, reader, model_load_start);
}

int RunQwen36Chat(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start) {
  return RunChatImpl(opt, reader, model_load_start);
}

}  // namespace gufo::models::qwen36_35b_a3b
