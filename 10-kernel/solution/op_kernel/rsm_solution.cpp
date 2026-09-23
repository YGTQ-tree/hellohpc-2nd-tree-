#include "rsm_solution_tiling.h"
#include "rsm_solution_core.h"

extern "C" __global__ __aicore__ void rsm_solution(
    GM_ADDR score, GM_ADDR x, GM_ADDR offsets, GM_ADDR mean, GM_ADDR rstd,
    GM_ADDR logsumexp, GM_ADDR workspace, RsmSolutionTiling tiling)
{
    (void)workspace;
    RsmSolution::ComputeCore core;
    core.Init(score, x, offsets, mean, rstd, logsumexp,
        tiling.n, tiling.d, tiling.epsilon);

    // 段号按 stride 轮转，保证相邻段落在不同核上，长段之间天然负载均衡。
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    for (uint32_t segment = tiling.first + block; segment < tiling.s;
         segment += blocks) {
        core.ProcessSegment(segment);
    }
}
