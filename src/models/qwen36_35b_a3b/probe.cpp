#include "src/models/qwen36_35b_a3b/probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::qwen36_35b_a3b {

bool ProbeQwen36Gguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
         "qwen35moe";
}

// Qwen3.6-35B-A3B model card: general thinking and instruct modes both use
// presence penalty 1.5. An unspecified thinking mode behaves as enabled.
std::optional<sampling::TextSamplingDefaults> Qwen36SamplingDefaults() {
  sampling::TextSamplingValues on;
  on.temperature = 1.0F;
  on.top_p = 0.95F;
  on.top_k = 20;
  on.presence_penalty = 1.5F;
  sampling::TextSamplingValues off = on;
  off.temperature = 0.7F;
  off.top_p = 0.8F;
  return sampling::TextSamplingDefaults{on, on, off};
}

std::optional<sampling::TextSamplingDefaults> Qwen36SamplingDefaults(
    const core::GgufReader&) {
  return Qwen36SamplingDefaults();
}

bool ProbeQwen36(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::is_regular_file(path))
      return false;
    std::string error;
    auto reader = core::GgufReader::OpenFile(path.string(), &error);
    return reader && ProbeQwen36Gguf(*reader);
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen36_35b_a3b
