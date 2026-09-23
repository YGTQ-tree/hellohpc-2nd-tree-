#include "maimoe/kernel_api.hpp"

#include "MaiMoeEngine.hpp"

#include <cstdint>

namespace maimoe::kernel {

void process_chart(std::uint64_t chart_id, std::string_view chart_text,
                   std::span<std::uint8_t, kEncodedChartBytes> output) {
    static_assert(engine::kChartEncodedBytes == kEncodedChartBytes);
    engine::EncodedChartLayout layout;
    engine::process_chart_into(
        chart_id, chart_text,
        std::span<std::uint8_t, engine::kChartEncodedBytes>(output.data(), output.size()),
        layout);
}

}
