#include "rsm_solution_host.h"

#include <algorithm>

namespace {
// Ascend 910B3 的向量核数量。段数多于核数时按 stride 轮转分配，
// 每个核拿到约 ceil(S / kVectorCores) 个段。
constexpr uint32_t kVectorCores = 40;
}

// Host 只负责任务划分，不做任何数学计算：
//   - 记录本次调用的形状与 epsilon；
//   - 决定启动多少个核：段数少于核数时只启动 S 个核，避免空转；
//   - 告诉每个核它要处理的段号集合（block + k * blockDim）。
// offsets 的具体数值留给 Kernel 在设备侧读取（README 允许 Host 读取 offsets，
// 但把边界判断放在设备侧可以让同一形状下的不同分段方式共用同一份 Host 规划）。
void ConfigureSolutionLaunch(uint32_t n, uint32_t d, uint32_t s, float epsilon,
    const int32_t *offsets, RsmSolutionTiling *tiling,
    uint32_t *block_dim, uint32_t *tiling_key)
{
    (void)offsets;
    const uint32_t blocks = std::min(s, kVectorCores);
    *tiling = {n, d, s, epsilon, /*first=*/0, /*stride=*/blocks};
    *block_dim = blocks;
    *tiling_key = 1;
}
