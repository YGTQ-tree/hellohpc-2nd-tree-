# 08-blackhole：球面波形积分优化记录

## 1. 题目在计算什么

程序从三维网格上的两个场分量（复数波形的实部和虚部）取值，在多个球面探测器上做高阶插值，再计算球谐模态积分。MPI 把采样点分给多个进程，最后把各进程的积分相加。输入场每个正式 iteration 都会更新，因此每一轮必须使用新的场值重新插值和积分。

评测检查 `benchmark-result.txt` 中全部数值，容差是绝对误差 `1e-8`、相对误差 `1e-14`；性能记录必须恰好输出一行 `BLACKHOLE_TIME ...`。四种输入的规模和分数不同，`small` 有 300 轮，`medium` 有 22500 轮，`large` 有 4000 轮，`huge` 有 13500 轮。配置目录由评测提供，不能修改。

## 2. 先找出重复做的工作

旧流程每轮、每个探测器都要在球面采样点上寻找所属网格 block，再计算插值窗口（stencil）的起点和 Lagrange 权重。球面位置、网格划分和插值阶数在 benchmark 中不变，这些几何信息重复计算没有必要。与此同时，实部和虚部在同一批点上使用同一个 stencil，分别调用插值会重复遍历相同的网格邻点。

球谐积分也反复计算同样的 Wigner 函数、角度三角函数和模式系数。它们只依赖球面离散、对称方式和模态参数，不依赖每轮变化的场数据，所以可以在参数确定后计算一次并复用。

## 3. 做了哪些优化

### 3.1 缓存固定的角度系数

在 `src/surface_integral.C` 中，为当前的 `spinw`、`maxl` 和模式数生成 theta 方向 Wigner 系数，以及 phi 方向的正弦、余弦系数。后续调用直接读缓存，避免对每个 iteration、采样点和模式重新调用昂贵的三角函数及 Wigner 函数。

缓存中没有场值、插值结果或积分结果。每个正式 iteration 的 `surf_Wave` 仍会读最新场、重新完成插值、重新遍历所有采样点和模态，并进行原有 MPI 求和。缓存参数变化时会重建系数。

### 3.2 缓存网格归属和插值几何

在 `src/MPatch.C` 中，采样点第一次出现时仍按原有 block 边界规则查找所属 block，并保存该 block、每个方向的 stencil 起点和 Lagrange 权重。几何相同的下一次调用复用这些信息。缓存键先用采样坐标首、中、末位置的位模式形成短指纹以定位候选；之后仍逐个比较全部坐标，只有完整相同才会命中，因此指纹碰撞不会把不同球面当成同一几何。

每一轮读取 block 中当轮最新的 `fgfs` 场数组，并使用缓存 stencil 重新计算插值值。缓存的是坐标、block 归属和插值权重，不缓存任何场数据。

### 3.3 合并两个场分量的插值

当待插值变量恰好是实部和虚部时，代码在同一 stencil 循环内一起累加两个输出。这样共享网格索引、权重乘积和内层循环控制，少走一遍相同的网格邻点。对称边界仍按每个场变量自己的 `SoA` 奇偶性处理；未满足双变量条件时仍使用原来的通用插值函数。

以上修改仅作用于 `src/` 中获准修改的实现文件。没有更改 `configs/*.par`，也没有删掉探测器、采样点、模态、iteration 或 MPI 归约。

## 4. 如何在集群上编译和测试

题目规定编译命令为：

```bash
source ./env.sh
make -C src -j blackhole
```

`env.sh` 选择 OpenMPI 4.0.3 和 MPI C/C++/Fortran 编译器。共享集群上批作业启动时先加载模块初始化脚本、设置软件模块路径，再 `source ./env.sh`。登录节点只用于编辑、传文件和提交 Slurm 作业；编译和 MPI 运行应放在获准的计算队列中。下面是一个 small 批作业主体示例，资源申请选项需按集群公告填写：

```bash
source /etc/profile.d/modules.sh
export MODULEPATH=/vault/software/modulefiles:$MODULEPATH
cd ~/hellohpc-08-blackhole-codex
source ./env.sh
make -C src -j blackhole
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll=^hcoll
mpirun -np 32 --bind-to core --map-by core src/blackhole configs/small.par
```

MPI 环境变量是为当前集群的 UCX/HCOLL 启动问题设置的运行参数；如提交环境没有该故障，可按官方运行说明启动。正式评测命令由 OJ 执行，不需要把临时 Slurm 脚本加入提交包。

手工核对结果时，将生成的 `benchmark-result.txt` 与同题的官方参考文件逐个数值比较，并使用 `|actual-expected| <= 1e-8 + 1e-14*|expected|`。不能只看程序是否打印“success”。

## 5. 当前验证结果

最终的短指纹查找版本在远端 `kp_run` 队列用 32 ranks 编译并运行了 `small`：

| 样例 | ranks | 结果 | `BLACKHOLE_TIME max` |
|---|---:|---|---:|
| small | 32 | 5/5 个数值均满足官方容差 | 15.116021694 s |
| medium | 128 | 未完成验证；旧版本运行超过 14 分钟仍未结束，已停止 | — |
| large | 32 | 尚未运行 | — |
| huge | 128 | 尚未运行 | — |

此前的 short fingerprint 变更前版本 small 也通过了数值比较，耗时约 15.1161 s。不同小测试的测时会有波动。small 满分时间为 0.15 s，所以当前实现离满分目标仍很远；medium/large/huge 没有正确性或性能结论。medium 的未完成运行说明不能把当前方案当成整题完成或可得分的保证。

## 6. 提交

提交包只放评测允许改动的 `src/` 和 `env.sh`，文件为 `submission-packages/08-blackhole.zip`。解压后应直接看到 `src/` 和 `env.sh` 两项。不要包含编译生成的 `.o` 文件或 `src/blackhole` 可执行文件；OJ 会在自己的环境中按规定重新编译。题目说明和本 writeup 用于阅读，不是程序提交所需文件。

若需手动重建压缩包，在仓库根目录运行：

```bash
cd 08-blackhole && zip -qr ../submission-packages/08-blackhole.zip src env.sh
```

当前给出的提交包应在解压后把顶层目录名 `08-blackhole/` 中的 `src/` 和 `env.sh` 放到题目工作目录根部；如果 OJ 要求自行上传散文件而非 zip，应直接提交其中的 `src/**` 和 `env.sh`。提交前用 `unzip -l` 确认压缩包中不含配置修改、参考答案、测试输出和编译产物。
