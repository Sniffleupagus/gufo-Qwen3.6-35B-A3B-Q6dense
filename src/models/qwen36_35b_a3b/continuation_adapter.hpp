#ifndef GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_QWEN36_35B_A3B_CONTINUATION_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/session_mode.hpp>
#include <gufo/ext/transfer_pool.hpp>
#include <memory>
#include <span>
#include <vector>

#include "src/models/qwen36_35b_a3b/engine.hpp"

namespace gufo::models::qwen36_35b_a3b {

// State inventory of one session. IDs refer to this representation, never to
// an allocation or serving slot.
enum class ContinuationPart {
  kTokens,
  kLogits,
  kMetadata,
  kConvolution,
  kRecurrent,
  kKey,
  kValue,
  kKeyQ8,
  kValueQ8,
  kKeyScale,
  kValueScale,
  kDraftKey,
  kDraftValue,
  kDraftKeyQ8,
  kDraftValueQ8,
  kDraftKeyScale,
  kDraftValueScale,
  kDraftResidual,
  kKeptHidden,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};

inline constexpr std::size_t kContinuationMetadataBytes = 192;

// Component-cache adapter over the Qwen3.6-35B-A3B session. Append rows are
// the target and draft KV caches (F16 plus the 8-bit decode copies and their
// scales) and the token history; recurrent state, the last logits, the draft
// residual, the kept trunk rows and host policy are private state at the
// target frontier. Slots own public model sessions, so the runner keeps using
// Sync/Evaluate/DecodeStep.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, core::SessionMode,
                      std::uint32_t context, cache::Identity);
  ~ContinuationAdapter() override;

  // Settles this slot's transfers and checks it may execute.
  [[nodiscard]] Session& GetSession(cache::Slot&);
  // Call on the host before any operation that may write session state. A
  // write starts at the current frontiers unless `restart` (Sync after a
  // divergence or a failed operation), which rewrites every row.
  void BeforeMutation(cache::Slot&, bool restart);
  // Session::ConfigureVision, guarding the reset it makes when the attached
  // images differ from the computed prefix.
  void ConfigureVision(cache::Slot&,
                       std::shared_ptr<const qwen::vision::Prompt>);

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const override;
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host;
  };
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] std::vector<cache::ComponentPosition> PositionsAt(
      std::uint32_t target, std::uint32_t draft) const;
  [[nodiscard]] void* RowData(State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Rows ValidRows(const State&,
                                      const ContinuationComponent&) const;
  [[nodiscard]] std::vector<Piece> PrivatePieces(
      State&, const ContinuationComponent&, bool loading) const;
  void GuardRows(State&, std::uint32_t target_first, std::uint32_t draft_first,
                 bool release) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);

  std::shared_ptr<Model> model_;
  core::SessionMode mode_;
  std::uint32_t context_;
  std::uint32_t kept_rows_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};

}  // namespace gufo::models::qwen36_35b_a3b
#endif
