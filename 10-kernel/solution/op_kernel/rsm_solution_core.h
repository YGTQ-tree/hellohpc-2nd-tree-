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
// 实现思路（相对原始逐行写法的改动，数学定义完全一致）
// ---------------------------------------------------------------
// 1) 权重不做“逐行标量”。原始实现每行都要 GetValue() 取出标量权重，再驱动一次
//    长度为 D 的向量乘加；单条指令的有效计算量很小，向量单元的启动开销成了
//    长段的主要瓶颈。这里一次处理 R 行（一个 chunk）：
//        exp_buffer = Exp(score_chunk - m)       // R x D 一次算完
//        mean      += sum_rows x_row * exp_row   // 列方向累加
//    仍然是“每行一条 D 宽向量指令”，但没有标量往返，exp 也整块向量化了。
//
// 2) 段内数据按 chunk 流式处理，Global Memory 的读取遍数固定，不随段长增长：
//      - Pass 1：求段内最大值（只读 score，score 比 x 小 D 倍）；
//      - Pass 2：累加 N 与加权和（读 x）；
//      - Pass 3：围绕最终均值累加中心化二阶矩（读 x）。
//
// 3) offsets 每段只从 GM 读两个值。
//
// 数值稳定性（对应题目的 precision contract）
// ---------------------------------------------------------------
//   - exp 自变量先减段内最大值，<= 0，任何 chunk 的 exp 之和都落在 [1, rows]，
//     不会在 float32 上溢；极小权重自然下溢为 0，与 FP64 参考一致。
//   - 加权和用补偿求和（Kahan）：把 -correction 也减掉，吸收“大数加小数”丢掉的
//     低位。±65504 互相抵消的场景（signed_cancellation）依赖这一点。
//   - 方差围绕已求出的 float32 均值中心化后再累加，而不是 E[x^2] - mean^2，
//     所以“大均值 + 小方差”不会灾难性抵消；常量列得到严格 0 方差。
//   - rstd 用硬件 Rsqrt；logsumexp = Ln(N) + m，避免 N 上溢。
//
// 缓冲区容量的两条铁律（踩过坑，写在这里防止回归）
// ---------------------------------------------------------------
//   * 以“行号 × 行宽”索引的缓冲，容量必须 >= kChunkRows * row_stride_，
//     而 row_stride_ 是运行时的（D 向上取整到 kRowAlign 的倍数）。
//   * 一次会写入 d_ 个元素的缓冲（Duplicate / Adds / Reduce* 的输出），
//     容量必须 >= kMaxD，不能只给一个向量宽度。
//   早期版本把两者都按 64 分配，D=80 时越界，读回脏数据，整段输出变成 NaN。

#include "kernel_operator.h"

namespace RsmSolution {

// 每个 chunk 处理的行数。
constexpr uint32_t kChunkRows = 24;
// 题目给定的 D 上界。
constexpr uint32_t kMaxD = 256;
// 行宽对齐粒度：16 个 float = 64 字节，保证按行切分的 Cast/DataCopy 都落在
// 32 字节块边界上。
constexpr uint32_t kRowAlign = 16;

class ComputeCore {
public:
    __aicore__ inline ComputeCore() {}

    __aicore__ inline void Init(GM_ADDR score, GM_ADDR x, GM_ADDR offsets,
        GM_ADDR mean, GM_ADDR rstd, GM_ADDR logsumexp,
        uint32_t n, uint32_t d, float epsilon)
    {
        d_ = d;
        epsilon_ = epsilon;
        row_stride_ = ((d + kRowAlign - 1) / kRowAlign) * kRowAlign;
        tile_capacity_ = kChunkRows * row_stride_;

        score_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(score), n);
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x),
            static_cast<uint64_t>(n) * d);
        offsets_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(offsets));
        mean_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mean));
        rstd_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rstd));
        logsumexp_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logsumexp));

        // UB 预算（kChunkRows = 24、row_stride_ = 256 时）：
        //   x_float / exp_buffer / product 各 24 KB
        //   score_float 24 KB + score_half 12 KB + reduce_work 24 KB
        //   小缓冲按 kMaxD 分配，合计 < 3 KB
        pipe_.InitBuffer(score_float_buffer_, kChunkRows * kMaxD * sizeof(float));
        pipe_.InitBuffer(score_half_buffer_, kChunkRows * kMaxD * sizeof(half));
        pipe_.InitBuffer(reduce_work_buffer_, kChunkRows * kMaxD * sizeof(float));
        pipe_.InitBuffer(x_float_buffer_, tile_capacity_ * sizeof(float));
        pipe_.InitBuffer(exp_buffer_, tile_capacity_ * sizeof(float));
        pipe_.InitBuffer(product_buffer_, tile_capacity_ * sizeof(float));
        pipe_.InitBuffer(x_half_buffer_, kMaxD * sizeof(half));
        pipe_.InitBuffer(mean_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(corr_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(m2_buffer_, kMaxD * sizeof(float));
        pipe_.InitBuffer(scalar_buffer_, kMaxD * sizeof(float));
    }

    __aicore__ inline void ProcessSegment(uint32_t segment)
    {
        const uint32_t begin = static_cast<uint32_t>(offsets_gm_.GetValue(segment));
        const uint32_t end = static_cast<uint32_t>(offsets_gm_.GetValue(segment + 1));
        const uint32_t rows = end - begin;

        if (rows == 1) {
            ProcessSingleRow(segment, begin);
            return;
        }

        // Pass 1：段内最大值。
        float maximum = -65504.0f;
        for (uint32_t base = 0; base < rows; base += kChunkRows) {
            const uint32_t count = Minimum(kChunkRows, rows - base);
            LoadScores(begin + base, count);
            maximum = MaximumOf(maximum, MaximumOfLoaded(count));
        }

        // Pass 2：N 与加权和。
        ResetMeanAccumulators();
        float normalizer = 0.0f;
        for (uint32_t base = 0; base < rows; base += kChunkRows) {
            const uint32_t count = Minimum(kChunkRows, rows - base);
            LoadChunk(begin + base, count);
            BuildExp(count, maximum);
            normalizer += SumOf(count);
            AccumulateWeightedX(count);
        }
        ScaleBy(mean_buffer_, 1.0f / normalizer);

        // Pass 3：中心化二阶矩。
        ResetVarianceAccumulator();
        for (uint32_t base = 0; base < rows; base += kChunkRows) {
            const uint32_t count = Minimum(kChunkRows, rows - base);
            LoadChunk(begin + base, count);
            BuildExp(count, maximum);
            AccumulateCentredVariance(count);
        }
        ScaleBy(m2_buffer_, 1.0f / normalizer);

        StoreOutputs(segment, maximum, normalizer);
    }

private:
    __aicore__ inline uint32_t Minimum(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline float MaximumOf(float lhs, float rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    // ---- 载入 -------------------------------------------------------------

    // score 按 chunk 载入 score_float_（行宽 row_stride_）。
    __aicore__ inline void LoadScores(uint32_t begin, uint32_t rows)
    {
        auto score_half = score_half_buffer_.Get<half>();
        auto score_float = score_float_buffer_.Get<float>();
        for (uint32_t base = 0; base < rows; base += kChunkRows) {
            const uint32_t count = Minimum(kChunkRows, rows - base);
            AscendC::DataCopyPad(score_half, score_gm_[begin + base],
                {1, static_cast<uint16_t>(count * sizeof(half)), 0, 0}, {});
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(score_float[base * row_stride_], score_half,
                AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // x 与 score 一起载入一个 chunk。
    __aicore__ inline void LoadChunk(uint32_t begin, uint32_t rows)
    {
        LoadScores(begin, rows);
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

    // 已装载的 rows 行的 score 最大值（逐行归约）。
    __aicore__ inline float MaximumOfLoaded(uint32_t rows)
    {
        auto score_float = score_float_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();
        auto work = reduce_work_buffer_.Get<float>();
        float maximum = -65504.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::ReduceMax(scalar, score_float[row * row_stride_], work,
                static_cast<int32_t>(d_));
            AscendC::PipeBarrier<PIPE_V>();
            maximum = MaximumOf(maximum, scalar.GetValue(0));
        }
        return maximum;
    }

    // ---- exp --------------------------------------------------------------

    __aicore__ inline void BuildExp(uint32_t rows, float maximum)
    {
        auto score_float = score_float_buffer_.Get<float>();
        auto exp_buffer = exp_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Adds(score_float[row * row_stride_],
                score_float[row * row_stride_], -maximum, static_cast<int32_t>(d_));
            AscendC::Exp(exp_buffer[row * row_stride_],
                score_float[row * row_stride_], static_cast<int32_t>(d_));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float SumOf(uint32_t rows)
    {
        auto exp_buffer = exp_buffer_.Get<float>();
        auto scalar = scalar_buffer_.Get<float>();
        auto work = reduce_work_buffer_.Get<float>();
        float total = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::ReduceSum(scalar, exp_buffer[row * row_stride_], work,
                static_cast<int32_t>(d_));
            AscendC::PipeBarrier<PIPE_V>();
            total += scalar.GetValue(0);
        }
        return total;
    }

    // ---- 累加 -------------------------------------------------------------

    __aicore__ inline void ResetMeanAccumulators()
    {
        auto mean = mean_buffer_.Get<float>();
        auto corr = corr_buffer_.Get<float>();
        AscendC::Duplicate(mean, 0.0f, kMaxD);
        AscendC::Duplicate(corr, 0.0f, kMaxD);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ResetVarianceAccumulator()
    {
        auto m2 = m2_buffer_.Get<float>();
        AscendC::Duplicate(m2, 0.0f, kMaxD);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScaleBy(const AscendC::LocalTensor<float> &value, float scale)
    {
        AscendC::Muls(value, value, scale, kMaxD);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // xw += sum_rows x_row * exp_row，带 Kahan 补偿：
    //   y    = x*w - c
    //   t    = sum + y
    //   c    = (t - sum) - y        ← 本行丢掉的低位
    //   sum  = t - c                ← 把低位补回去
    __aicore__ inline void AccumulateWeightedX(uint32_t rows)
    {
        auto mean = mean_buffer_.Get<float>();
        auto corr = corr_buffer_.Get<float>();
        auto exp_buffer = exp_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto product = product_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Mul(product, x_float[row * row_stride_],
                exp_buffer[row * row_stride_], d_);
            AscendC::Sub(product, product, corr, d_);
            AscendC::Add(product, mean, product, d_);
            AscendC::Sub(corr, product, mean, d_);
            AscendC::Sub(corr, corr, product, d_);
            AscendC::Mul(corr, corr, -1.0f, d_);
            AscendC::Add(mean, product, corr, d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // m2 += sum_rows exp_row * (x_row - mean)^2。
    __aicore__ inline void AccumulateCentredVariance(uint32_t rows)
    {
        auto m2 = m2_buffer_.Get<float>();
        auto mean = mean_buffer_.Get<float>();
        auto exp_buffer = exp_buffer_.Get<float>();
        auto x_float = x_float_buffer_.Get<float>();
        auto product = product_buffer_.Get<float>();
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Sub(product, x_float[row * row_stride_], mean, d_);
            AscendC::Mul(product, product, product, d_);
            AscendC::Mul(product, product, exp_buffer[row * row_stride_], d_);
            AscendC::Add(m2, m2, product, d_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- 写回 -------------------------------------------------------------

    __aicore__ inline void StoreOutputs(uint32_t segment, float maximum, float normalizer)
    {
        auto mean = mean_buffer_.Get<float>();
        auto m2 = m2_buffer_.Get<float>();
        const uint16_t bytes = static_cast<uint16_t>(d_ * sizeof(float));
        AscendC::DataCopyPad(mean_gm_[segment * d_], mean, {1, bytes, 0, 0});
        AscendC::DataCopyPad(rstd_gm_[segment * d_], m2, {1, bytes, 0, 0});

        // logsumexp = maximum + Ln(normalizer)
        auto scalar = scalar_buffer_.Get<float>();
        AscendC::Duplicate(scalar, normalizer, kMaxD);
        AscendC::Ln(scalar, scalar, kMaxD);
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
        AscendC::Duplicate(scalar, epsilon_, kMaxD);
        AscendC::Rsqrt(scalar, scalar, kMaxD);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_work_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_float_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_half_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mean_buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> corr_buffer_;
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
    uint32_t tile_capacity_ = 0;
    float epsilon_ = 0.0f;
};

}  // namespace RsmSolution
