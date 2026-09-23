#pragma once

#include <cstdint>

// Host <-> Kernel参数（Tiling）。
//
// Host 侧已经拿到了本次调用的 N、D、S、epsilon 和完整的 offsets 表，所以它只需
// 要告诉 Kernel“要处理哪些段”。下面每个字段都是普通整数，整个结构保持 POD，
// 可以直接按值作为 Kernel 参数传递。
struct RsmSolutionTiling {
    uint32_t n;        // 总行数
    uint32_t d;        // 列数
    uint32_t s;        // 分段数
    float epsilon;     // rstd 的方差下限
    uint32_t first;    // 本核负责段号的起点（block 0 从 first 开始）
    uint32_t stride;   // 段号按 first + k * stride 递增
};
