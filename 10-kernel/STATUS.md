# 当前进展与交接说明（Ragged Softmax Moments / 10-kernel）

> 本文档记录截至 2026-09-23 晚上的真实状态，供继续推进时快速接手。
> **结论先行：还没有做出能在评测机上通过正确性的版本，性能也还没有超过基线。**

## 1. 环境事实（已实测复核）

| 项 | 事实 |
|---|---|
| CPU 集群 | `ssh arm` → `armlogin.hpc.sjtu.edu.cn`，登录节点 `kp001.pi.sjtu.edu.cn`，aarch64 |
| CPU 队列 | `kp_run`（≤128 核，2h，每用户 1 个作业）、`kp_interact`（≤8 核，8h） |
| NPU 集群 | `ssh ascend` → `ascend0.xflops.org`，账号 `stu1520`，家目录 `/nfs/home/acct-stu/stu1520`（7 TB，用 16%） |
| NPU 队列 | `contest-slice`（12 核 + 1 卡，2h）、`contest-full`（192 核 + 8 卡，1h，需 LLM Serving Stage1 满分）；另有 `judge-slice` / `judge-full`（评测专用，勿占） |
| NPU 节点 | `ascend6` / `ascend7`，各 192 核、8 张 `Ascend-910B3`；内存按 4 GB/核 分配 |
| 提交作业 | `#SBATCH -p contest-slice -q contest_slice -N 1 -c 12 --gres=npu:1`，**必须带 `--gres=npu:1`**，否则报 `QOSMinGRES` |
| 开发环境 | `source /nfs/scripts/kernel-dev.sh` → mamba 环境 `ascend`，CANN 8.5.0 位于 `/nfs/miniforge3/envs/ascend/Ascend/cann-8.5.0` |
| 编译工具 | `bisheng` / `ccec` / `ld.lld`（CANN 自带）、`g++ 12.4.0`、`cmake 3.31.8`、`python3 3.11.16`（conda 环境内） |
| 设备端编译器 | `/nfs/bin/hellohpc`（CLI）、`/nfs/scripts/kernel-dev.sh`（环境）；评测容器镜像 `/nfs/bin/ubuntu-26.04-arm64-builder.sif` |
| 远端工作副本 | `~/rsm`（本仓库 `10-kernel/` 的 rsync 副本），作业脚本 `job_build.slurm` / `job_dsum.slurm` / `job_diag.slurm`，日志 `~/rsm/logs/` |

**登录节点上编译会失败**：`ld.lld: unknown file type`（PATH 里混入 `/usr/bin/llvm-objdump` 时）。
所有编译/运行都必须放进 `sbatch`。

## 2. 本地已建立的分析工具

| 文件 | 作用 |
|---|---|
| `tools/dev/kernel_precision_model.py` | 用 NumPy 在 CPU 上**逐操作复刻**设备端 float32 运算顺序（分块、exp、Kahan/双 float 累加、中心化方差），可在不占用 NPU 的情况下验证数值方案是否满足 precision contract |

这个模型的价值：它已经独立复现出「参考实现用 FP64 算完只舍入一次」这一关键约束，
并把不满足约束的算法方案（直接累加、单纯 Kahan、按 tile 参考行、按段参考行）逐一排除。

## 3. 已经确认的设备端错误（都已修，但问题未完全解决）

1. **向量宽度 padding 未初始化**：`D` 不一定是 64 的倍数（如 `D=80`），
   向量指令按 64 个 float 一个 repeat 工作，越界到 padding。已改为按 `d_` 计数，
   并把 padding 显式清零。
2. **`score_half_buffer_` 容量按 tile 行数固定**，但流式路径会一次搬 `rows` 行 →
   缓冲越界，读到脏数据（表现为整段 NaN/inf）。已改为分块搬运。
3. **score 在 UB 中的排版与读取不一致**：写入用连续偏移、读取用 `row * row_stride_`。
   已统一为 `row_stride_` 排版。
4. **`scalar_buffer_` 只按一个向量宽度（64）分配**，但多处会一次写入 `d_` 个元素
   （`D=80` 时越界）。已按 `kMaxD` 分配。
5. **流式路径第一遍先分块装载再逐块归约**，会读到尚未写入的行。已改为整块装载后再归约。

## 4. 当前状态

| 项 | 状态 |
|---|---|
| 合规检查 | ✅ 通过（`tools/compliance/run_checks.py`） |
| 编译 | ✅ 通过（`bash solution/build.sh`，bisheng 编译 AIC/AIV） |
| 正确性 | ❌ **未通过**：单行分段（`public_singletons`、`public_single_row`）正确；其余分段输出 NaN/inf |
| 性能 | ❌ 比基线慢：`public_perf_small_s` 基线 1.126 ms，当前解 1.86–2.92 ms |
| 精度模型（CPU） | 57/58 公开用例满足容差；唯一超标的是 `public_adversarial_low_mass_tail`（1.06 倍容许误差），根因是设备 Exp 与 FP64 exp 的 1e-7 级差异被「极大的权重动态范围」放大 |

**遗留的 NaN 问题定位进度**：已确认不是「所有段都错」——单行段正确、多行段全错；
`logsumexp` 输出为 `inf`，说明 `exp(score - maximum)` 溢出，即 **`maximum` 读成了极负值**。
`public_alignment`（`D=80`，offsets `[0,49,100,200,224,250,348]`）是复现用例，
`public_adversarial_segment_limit`（S=8192、每段 1 行）能通过。

**下一步最快的定位手段**：用 `~/kprobe/` 里的独立探针工程（`probe.cpp` + `main.cpp` + `CMakeLists.txt`），
它只调用 `DataCopyPad` / `Cast` / `ReduceMax` / `Adds` / `Exp` 五个原语并回传结果，
一次 `sbatch` 即可判定是哪个原语在 `D=80`、跨行布局下的行为与预期不符。
探针环境要用 `~/kprobe/env.sh`（`CPLUS_INCLUDE_PATH` + `CC/CXX` 指向 conda 工具链），
**不要在 PATH 里加 CANN 的 bin**（会引入不兼容的 `llvm-objdump`）。

## 5. 已确立的性能目标

`public_perf_small_s`（S=4, N=8192, D=192）基线 1.13 ms，S=1/4 时只有 1~4 个核在干活。
满分线是 8 倍综合加速比，因此**必须解决「段数少于核数时并行度不足」**：
需要 Host 侧按行区间把长段拆给多个核（`SyncAll` 两阶段归约，或两个 Kernel），
否则长段类用例的天花板很低。

## 6. 红线自查（每次提交前过一遍）

- 只改 `problem.yaml` 里 `workspace.editable` 列出的 7 个文件；
- 不使用 ACLNN/ATB/框架算子，不引入高阶 Softmax/Norm/Moments/LogSumExp 算子；
- workspace ≤ 4 MiB；不依赖输出/workspace 初值，不做跨调用缓存；
- 不按用例 ID / 形状组合做特判（compliance 检查会做常量折叠匹配）；
- `runner/main.cpp` 只包含一次 `runner_impl.h`，不改计时流程；
- 所有计算都在计时段内完成（不把工作挪到 Host 或计时区间之外）。
