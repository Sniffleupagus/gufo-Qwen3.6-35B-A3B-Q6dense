#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::qwen36_35b_a3b {

[[nodiscard]] bool ProbeQwen36(std::string_view model_root);
[[nodiscard]] bool ProbeQwen36Gguf(const core::GgufReader& reader);
/// The Qwen3.6 model card's sampling; the runner descriptor and the provider
/// probe both use these defaults.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
Qwen36SamplingDefaults();
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
Qwen36SamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::qwen36_35b_a3b
