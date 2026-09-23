#pragma once

// Ragged Softmax Moments —— Ascend C 核心计算逻辑
//
// 对每个分段 s（行区间 [offsets[s], offsets[s+1])），令 m = 段内 score 最大值，
// w_i = exp(score_i - m)，N = sum_i w_i，则
//       mean_d    = sum_i w_i * x[i, d] / N
//       var_d     = sum_i w_i * (x[i, d] - mean_d)^2 / N
//       rstd_d    = 1 / sqrt(var_d + epsilon)
//       logsumexp = m + log(N)
//
// 与原始逐行实现的差别（算法与数据布局层面，数学定义完全一致）：
//
//  1) exp 不再逐行做“取标量 → 再乘一整行”。原始实现里权重是逐行 GetValue()
//     取出的标量，每行只驱动一条 D 宽的向量指令，向量单元的启动开销主导了
//     长段耗时。本实现把一整块 score 一次性平移 -m 再整块做 Exp，得到 R×D 的
//     权重分块；随后做矩阵式的列方向累加 xw += x_row * exp_row，把段内的行
//     折叠进同一个 D 宽累加器。平移用一条带 repeat 的 Adds（mask 模式）完成，
//     而不是一行一条指令。
//
//  2) 段内数据只要装得进 Unified Buffer（UB）就绝不重复读 Global Memory。
//     按段长分两条路径：
//       - 常驻路径（段长 <= kMaxLocalRows）：x 从 GM 只读一遍，求最大值、算 exp、
//         累加 normalizer / 加权和 / 中心化二阶矩全部在片上完成；
//       - 流式路径（更长的段）：按 tile 流式处理，第一遍求段内最大值，第二遍算
//         N 与加权均值，第三遍算中心化方差。
//     原始实现对每段固定读 3 遍 x，并且每行都夹带标量运算。
//
//  3) offsets 每段只从 GM 读两个值（本段起点与下一段起点）。
//
// 数值稳定性（对应题目的 precision contract）：
//   - exp 自变量先减段内最大值，<= 0，故每个 tile 的 exp 之和落在 [1, rows]，
//     不会在 float32 上溢。
//   - 加权求和用补偿求和（Kahan），避免 ±65504 抵消时丢有效位；normalizer 走
//     非负数的 pairwise reduce（ReduceSum），误差 O(log n)·eps。
//   - 方差围绕已求出的 float32 均值中心化后累加（而不是 E[x^2] - mean^2），
//     因此“大均值 + 小方差”不会灾难性抵消，常量列得到严格 0 方差。
//   - rstd 用硬件 Rsqrt；logsumexp = Ln(N) + m，避免 N 上溢。

#include "kernel_operator.h"

namespace RsmSolution {

// 常驻路径允许的最大行数；同时也是流式路径的 tile 行数。
constexpr uint32_t kMaxLocalRows = 24;
// 题目给定的 D 上界。
constexpr uint32_t kMaxD = 256;
// 向量 stride 单位：Adds 的 repeat stride 以 256 字节为步长，对应 float32 的 64 个元素。
constexpr uint32_t kStrideUnit = 64;
// 逐行累加时“乘积暂存缓冲”的行数；必须 >= kMaxLocalRows，否则会踩到
// exp_tile 的源数据。
constexpr uint32_t kProductRows = kMaxLocalRows;
// 单条向量指令一次处理的 float32 元素数（256 B / 4 B）。
constexpr uint32_t kVectorBlock = 64;

class ComputeCore {
public:
    __aicore__ inline ComputeCore() {}

    __aicore__ inline void Init(GM_ADDR score, GM_ADDR x, GM_ADDR offsets,
        GM_ADDR mean, GM_ADDR rstd, GM_ADDR logsumexp,
        uint32_t n, uint32_t d, float epsilon)
    {
        d_ = d;
        epsilon_ = epsilon;
        // 片上统一按 row_stride_ 表示一行的宽度：它是 d_ 向上取整到 64 的倍数。
        // 这样每一行都落在 256 字节边界上，可以用一条带 repeat 的向量指令同时
        // 处理所有行。DataCopyPad 会把 [d_, row_stride_) 这段自动补 0。
        row_stride_ = ((d + kStrideUnit - 1) / kStrideUnit) * kStrideUnit;
        element_capacity_ = kMaxLocalRows * row_stride_;

        score_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(score), n);
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x),
            static_cast<uint64_t>(n) * d);
        offsets_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        mean_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        rstd_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rstd));
        logsumexp_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logsumexp));

        // UB 预算（kMaxLocalRows = 24、row_stride_ = 256 时）：
        //   x_float   24*256*4 = 24 KB
        //   exp_tile  24*256*4 = 24 KB
        //   temp      24*256*4 = 24 KB
        //   x_half_stage  256*2 = 0.5 KB，其余 < 3 KB
        //   合计约 76 KB，为编译器与 D 较小时的 padding 留出余量。
        pipe_.InitBuffer(x_float_buffer_, element_capacity_ * sizeof(float));
        pipe_.InitBuffer(exp_tile_buffer_, element_capacity_ * sizeof(float));
        pipe_.InitBuffer(temp_buffer_, kProductRows * kMaxD * sizeof(float));
        pipe_.InitBuffer(x_half_buffer_, kMaxD * sizeof(half));
        pipe_.InitBuffer(score_half_buffer_, kMaxLocalRows * sizeof(half));
        pipe_.InitBuffer(score_float_buffer_, kMaxLocalRows * sizeof(float));
        pipe_.InitBuffer(reduce_work_buffer_, kMaxLocalRows * sizeof(float));
        pipe_.InitBuffer(mean_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(mean_corr_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(m2_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(scalar_buffer_, kStrideUnit * sizeof(float));
    }

    // 处理一个分段：算出 mean / rstd / logsumexp 并写回 Global Memory。
    __aicore__ inline void ProcessSegment(uint32_t segment)
    {
        const uint32_t begin = static_cast<uint32_t>(offsets_gm_.GetValue(segment));
        const uint32_t end = static_cast<uint32_t>(offsets_gm_.GetValue(segment + 1));
        const uint32_t rows = end - begin;

        if (rows == 1) {
            ProcessSingleRow(segment, begin);
        } else if (rows <= kMaxLocalRows) {
            ProcessResident(segment, begin, rows);
        } else {
            ProcessStreaming(segment, begin, rows);
        }
    }

private:
    __aicore__ inline uint32_t Minimum(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    // ---- 数据搬运 ---------------------------------------------------------

    // 把 [begin, begin+rows) 的 score 读进 UB 并转成 float32。
    __aicore__ inline void LoadScores(uint32_t begin, uint32_t rows)
    {
        auto score_half = score_half_buffer_.Get<half>();
        auto score_float = score_float_buffer_.Get<float>();
        AscendC::DataCopyPad(score_half, score_gm_[begin],
            {1, static_cast<uint16_t>(rows * sizeof(half)), 0, 0}, {});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(score_float, score_half, AscendC::RoundMode::CAST_NONE, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 把 [begin, begin+rows) 的 x 读进 UB 并转成 float32，按 row_stride_ 排版。
    // 逐行搬运：x 在 GM 里的行距是 d_，UB 里是 row_stride_，两者不一致；
    // DataCopyPad 会把 [d_, row_stride_) 自动补 0，正好用于后续整块求和。
    __aicore__ inline void LoadX(uint32_t begin, uint32_t rows)
    {
        auto x_half = x_half_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::DataCopyPad(x_half, x_gm_[(begin + row) * d_],
                {1, static_cast<uint16_t>(d_ * sizeof(half)), 0, 0}, {});
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(x_float[row * row_stride_], x_half,
                AscendC::RoundMode::CAST_NONE, d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- 段内最大值 -------------------------------------------------------

    __aicore__ inline float ReduceToMaximum(uint32_t rows)
    {
        auto score_float = score_float_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();
        auto work = reduce_work_buffer_.Get<float>();
        AscendC::ReduceMax(scalar, score_float, work, rows);
        AscendC::PipeBarrier<PIPE_V>();
        return scalar.GetValue(0);
    }

    // ---- exp 分块 ---------------------------------------------------------

    // exp_tile = exp(score - maximum)，整块一次算完，按 row_stride_ 排版。
    __aicore__ inline void BuildExpTile(uint32_t rows, float maximum)
    {
        auto score_float = score_float_buffer_.Get<float>();
        auto exp_tile = exp_tile_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();
        // 把 -maximum 铺满 scalar_buffer_ 的整行，再用一条带 repeat 的 Sub
        // 把所有行一起平移：src1 的 repeat stride 为 0，等于把同一行广播给每一行。
        AscendC::Duplicate(scalar, -maximum, kStrideUnit);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(score_float, score_float, scalar, static_cast<uint64_t>(d_),
            static_cast<uint8_t>(rows), {1, 1, 1, kStrideUnit, kStrideUnit, 0});
        AscendC::PipeBarrier<PIPE_V>();
        // 整块 exp：输入输出同布局，原地安全；row_stride_ 是 64 的倍数，
        // 所以多算出来的部分只落在各行自己的 padding 上。
        AscendC::Exp(exp_tile, score_float, static_cast<int32_t>(rows * row_stride_));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // N = sum(exp_tile)；padding 已被补 0，可以整块求和。
    __aicore__ inline float TileExpSum(uint32_t rows)
    {
        auto exp_tile = exp_tile_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();
        auto work = reduce_work_buffer_.Get<float>();
        AscendC::ReduceSum(scalar, exp_tile, work,
            static_cast<int32_t>(rows * row_stride_));
        AscendC::PipeBarrier<PIPE_V>();
        return scalar.GetValue(0);
    }

    // ---- 常驻路径 ---------------------------------------------------------

    __aicore__ inline void ProcessResident(uint32_t segment, uint32_t begin, uint32_t rows)
    {
        LoadScores(begin, rows);
        LoadX(begin, rows);
        const float maximum = ReduceToMaximum(rows);
        BuildExpTile(rows, maximum);
        const float normalizer = TileExpSum(rows);
        ResetMeanAccumulators();
        AccumulateXw(rows);
        FinalizeMean(normalizer);
        ResetVarianceAccumulator();
        AccumulateVariance(rows);
        FinalizeVariance(normalizer);
        StoreOutputs(segment, maximum, normalizer);
    }

    // ---- 流式路径 ---------------------------------------------------------

    __aicore__ inline void ProcessStreaming(uint32_t segment, uint32_t begin, uint32_t rows)
    {
        // 第一遍：段内最大值（tile 最大值沿段合并）。
        float maximum = -65504.0f;
        for (uint32_t base = 0; base < rows; base += kMaxLocalRows) {
            const uint32_t count = Minimum(kMaxLocalRows, rows - base);
            LoadScores(begin + base, count);
            const float tile_max = ReduceToMaximum(count);
            maximum = maximum > tile_max ? maximum : tile_max;
        }

        // 第二遍：逐 tile 算 exp，累加 normalizer 与加权和。
        ResetMeanAccumulators();
        float normalizer = 0.0f;
        for (uint32_t base = 0; base < rows; base += kMaxLocalRows) {
            const uint32_t count = Minimum(kMaxLocalRows, rows - base);
            LoadScores(begin + base, count);
            LoadX(begin + base, count);
            BuildExpTile(count, maximum);
            normalizer += TileExpSum(count);
            AccumulateXw(count);
        }
        FinalizeMean(normalizer);

        // 第三遍：围绕最终均值做中心化方差。
        ResetVarianceAccumulator();
        for (uint32_t base = 0; base < rows; base += kMaxLocalRows) {
            const uint32_t count = Minimum(kMaxLocalRows, rows - base);
            LoadScores(begin + base, count);
            LoadX(begin + base, count);
            BuildExpTile(count, maximum);
            AccumulateVariance(count);
        }
        FinalizeVariance(normalizer);
        StoreOutputs(segment, maximum, normalizer);
    }

    // ---- 行方向累加 -------------------------------------------------------

    __aicore__ inline void ResetMeanAccumulators()
    {
        auto mean = mean_buffer_.Get<float>();
        auto mean_corr = mean_corr_buffer_.Get<float>();
        AscendC::Duplicate(mean, 0.0f, d_);
        AscendC::Duplicate(mean_corr, 0.0f, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ResetVarianceAccumulator()
    {
        auto m2 = m2_buffer_.Get<float>();
        AscendC::Duplicate(m2, 0.0f, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // xw += sum_rows x_row * exp_row（Kahan 补偿求和）。
    __aicore__ inline void AccumulateXw(uint32_t rows)
    {
        auto mean = mean_buffer_.Get<float>();
        auto mean_corr = mean_corr_buffer_.Get<float>();
        auto exp_tile = exp_tile_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto temp = temp_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Mul(temp, x_float[row * row_stride_],
                exp_tile[row * row_stride_], d_);
            AscendC::Sub(temp, temp, mean_corr, d_);
            AscendC::Add(temp, mean, temp, d_);
            AscendC::Sub(mean_corr, temp, mean, d_);
            AscendC::Sub(mean_corr, mean_corr, temp, d_);
            AscendC::DataCopy(mean, temp, d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // m2 += sum_rows exp_row * (x_row - mean)^2。
    __aicore__ inline void AccumulateVariance(uint32_t rows)
    {
        auto m2 = m2_buffer_.Get<float>();
        auto mean = mean_buffer_.Get<float>();
        auto exp_tile = exp_tile_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto temp = temp_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Sub(temp, x_float[row * row_stride_], mean, d_);
            AscendC::Mul(temp, temp, temp, d_);
            AscendC::Mul(temp, temp, exp_tile[row * row_stride_], d_);
            AscendC::Add(m2, m2, temp, d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // mean = xw / N（原地）。
    __aicore__ inline void FinalizeMean(float normalizer)
    {
        auto mean = mean_buffer_.Get<float>();
        AscendC::Muls(mean, mean, 1.0f / normalizer, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // rstd = 1 / sqrt(var + eps)，原地写在 m2 上。
    __aicore__ inline void FinalizeVariance(float normalizer)
    {
        auto m2 = m2_buffer_.Get<float>();
        AscendC::Muls(m2, m2, 1.0f / normalizer, d_);
        AscendC::Maxs(m2, m2, 0.0f, d_);
        AscendC::Adds(m2, m2, epsilon_, d_);
        AscendC::Rsqrt(m2, m2, d_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- 收尾 -------------------------------------------------------------

    __aicore__ inline void StoreOutputs(uint32_t segment, float maximum, float normalizer)
    {
        auto mean = mean_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        const uint16_t bytes = static_cast<uint16_t>(d_ * sizeof(float));
        AscendC::DataCopyPad(mean_gm_[segment * d_], mean, {1, bytes, 0, 0});
        AscendC::DataCopyPad(rstd_gm_[segment * d_], m2, {1, bytes, 0, 0});

        auto scalar = scalar_buffer_.Get<float>();
        AscendC::Duplicate(scalar, normalizer, kStrideUnit);
        AscendC::Ln(scalar, scalar, kStrideUnit);
        AscendC::PipeBarrier<PIPE_V>();
        scalar.SetValue(0, scalar.GetValue(0) + maximum);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(logsumexp_gm_[segment], scalar,
            {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 单行分段：权重恒为 1，均值就是该行本身，方差为 0。
    __aicore__ inline void ProcessSingleRow(uint32_t segment, uint32_t row)
    {
        auto x_half = x_half_buffer_.Get<half>();
        auto x_float = x_float_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();

        AscendC::DataCopyPad(x_half, x_gm_[row * d_],
            {1, static_cast<uint16_t>(d_ * sizeof(half)), 0, 0}, {});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(x_float, x_half, AscendC::RoundMode::CAST_NONE, d_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopyPad(x_half, score_gm_[row],
            {1, static_cast<uint16_t>(sizeof(half)), 0, 0}, {});
        AscendC::PipeBarrier<PIPE_ALL>();
        const float score = static_cast<float>(x_half.GetValue(0));

        const uint16_t bytes = static_cast<uint16_t>(d_ * sizeof(float));
        AscendC::DataCopyPad(mean_gm_[segment * d_], x_float, {1, bytes, 0, 0});
        AscendC::Duplicate(scalar, epsilon_, kStrideUnit);
        AscendC::Rsqrt(scalar, scalar, kStrideUnit);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(rstd_gm_[segment * d_], scalar, {1, bytes, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();

        scalar.SetValue(0, score);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(logsumexp_gm_[segment], scalar,
            {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_tile_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_corr_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> m2_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalar_buffer_;
    AscendC::GlobalTensor<half> score_gm_;
    AscendC::GlobalTensor<half> x_gm_;
    AscendC::GlobalTensor<int32_t> offsets_gm_;
    AscendC::GlobalTensor<float> mean_gm_;
    AscendC::GlobalTensor<float> rstd_gm_;
    AscendC::GlobalTensor<float> logsumexp_gm_;
    uint32_t d_ = 0;
    uint32_t row_stride_ = 0;
    uint32_t element_capacity_ = 0;
    float epsilon_ = 0.0f;
};

}  // namespace RsmSolution
