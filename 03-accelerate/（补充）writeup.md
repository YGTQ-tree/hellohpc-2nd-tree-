# 03 核场响应加速：从公式到可验证的 CPU 优化

这份补充说明面向刚接触 HPC 的同学。假设读者会使用 Linux 命令行，知道 Python 和 NumPy 数组是什么，但还没有学习过 SIMD、OpenMP 或 CPU 缓存。

## 1. 题目到底要算什么

输入有七个数组：

- `points`：查询点，形状是 `(Q, D)`；
- `centers`：每个核的中心，形状是 `(C, D)`；
- `weights`、`scales`、`bias`、`trig_scale`：每个核一个参数，形状都是 `(C,)`；
- `trig_vec`：每个核的方向向量，形状是 `(C, D)`。

对查询点 `x = points[q]` 和第 `c` 个核，要先算两个长度为 `D` 的量：

```text
r = sum((x[d] - centers[c,d]) ** 2)
p = sum(x[d] * trig_vec[c,d])
```

这个核的贡献是：

```text
weights[c] * exp(-scales[c] * r)
    + bias[c] * sin(trig_scale[c] * p)
```

最后把所有 `C` 个核的贡献相加，得到 `output[q]`。因此总工作量大约是 `Q * C * D` 次维度计算。公开 `large-b` 的规模是 `Q=98,304`、`C=10,240`、`D=32`，也就是约十亿个查询点和核的组合，每个组合还要扫 32 个维度。

题目要求返回连续的 `numpy.float32` 数组，并满足：

```text
abs(answer - reference) <= 1e-5 + 1e-4 * abs(reference)
```

所以优化不能只看速度。只要误差超过这个界限，性能分也会变成 0。

## 2. 最初的实现为什么慢

最初的 `solver.py` 把循环放到 C，但每个 `(q, c)` 仍然按普通标量方式执行：

1. 用标量 `float` 累加平方距离；
2. 用标量 `float` 累加点积；
3. 调用一次 `expf`；
4. 调用一次 `sinf`；
5. 将结果加入查询点的总和。

这比 Python 三层循环快很多，因为 Python 不再参与十亿次迭代，但仍有两个主要问题：

- 一个查询点要重复读取所有核参数，计算量很大；
- `expf` 和 `sinf` 是通用数学库函数，单次调用成本高，而且标量调用不能充分使用 CPU 的向量寄存器。

原始版本在比赛 ARM 节点的公开 `large-b` 上约为 `1.85 s`。这只是公开数据的参考，不是 OJ 私有用例的成绩。

## 3. 第一步：把工作交给 C 和 OpenMP

Python 函数负责检查形状、把输入转换成 C-order 的 `float32` 数组、分配输出，并通过 `ctypes` 调用 C 函数。内嵌 C 源码在模块导入阶段由 GCC 编译成临时共享库：

```text
gcc -Ofast -fopenmp -fPIC -shared ... -lm
```

编译发生在导入阶段，题目的计时只包住正式的 `compute_field` 调用，因此不会把这次准备工作算进性能时间。代码没有读取参考答案，也没有缓存某次正式输入的结果。

不同查询点之间互不依赖，所以 C 内核用 OpenMP 并行处理查询块。每个线程只更新自己负责的 `output[q]`，不需要锁，也不需要在线程之间合并一个共享的总和。

## 4. 第二步：ARM NEON 向量化 D=16

比赛 CPU 集群是 ARM64 Kunpeng-920。它有 NEON 向量寄存器，一个 `float32x4_t` 可以同时保存 4 个 `float`。D=16 的正式路径使用四个查询点作为 4 个 lane：

```text
x0 = [point[q+0][d], point[q+1][d], point[q+2][d], point[q+3][d]]
```

对这四个查询点同时完成减法、乘法和累加。这样一次指令可以完成原来四次标量操作。

为了让查询点沿 lane 方向连续，内核先把输入从按查询点排列的布局：

```text
points[q][d]
```

转成按维度排列的临时布局：

```text
transposed[d][q]
```

转置后的每次 `vld1q_f32` 都能连续读取四个查询点。临时矩阵每一行额外留出 16 个 `float` 的 padding。这不是为了存数据，而是为了避免当 `Q` 恰好是某个缓存大小的倍数时，很多行反复映射到同一组 L1 cache set，造成严重的缓存冲突。

指数和正弦也提供了 4-lane 的多项式近似：

- `exp4` 先把 `x` 拆成 `k * ln(2) + r`，对小范围 `r` 用多项式计算 `exp(r)`，再用位操作构造 `2^k`；
- `sin4` 先把相位约减到一个较小区间，再用正弦/余弦多项式和象限选择恢复符号。

这条路径只在 ARM64 且 `D=16` 时使用。公开和正式的 D=16 用例已经通过，`large-a` OJ 成绩为 `30/30`，说明这条快速路径在对应数据范围内满足误差要求。

## 5. 为什么 D=32 没有强行使用同一套近似

这是本题最容易踩坑的地方。

`sin(x)` 不只是一次乘法。相位很大时，程序必须先计算 `x` 除以 `2*pi` 后的余数。若相位、点积或累加过程使用 `float32`，相位的大部分低位会在舍入时丢失。即使最后只需要一个 `float32` 输出，中间的正弦值也可能完全错误。

在 ARM 节点上做定向测试时，D=32 的 NEON `sin4` 在小相位上和系统 `sinf` 很接近；相位变大后误差迅速增加，极端情况下结果甚至不在 `[-1, 1]` 的正常范围内。把少数 lane 改成双精度也不能可靠解决所有隐藏数据，因为点积本身的求和误差和系统数学库的相位归约仍然可能不同。

因此最终源码采用保守的分支：

```text
ARM64 + D=16  -> NEON 向量内核
ARM64 + D=32  -> 双精度通用内核
其他平台      -> 双精度通用内核
```

D=32 通用内核对距离 `r`、点积 `p`、`exp`、`sin` 都使用 `double`，并用双精度累加每个查询点的总和，最后才转换为 `float32`。这与参考实现的数值行为更接近，修复了 large-b 的 `wrong_answer`。

代价是 D=32 速度明显下降：最新一次 OJ 结果中，large-b 正确通过，耗时约 `3298 ms`，得分 `35.01/50`。这是一个明确的取舍：先保证所有正式数据正确，再考虑能否设计经过完整误差验证的双精度向量数学。

## 6. 为什么只改 `env.sh` 不够

`env.sh` 在评测开始时被加载，适合做环境设置，例如：

```bash
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

这些变量可以影响线程数量、线程绑定和 OpenMP 调度，但不能改变下面这些事实：

- 每个查询点和核仍然要计算一次距离和点积；
- 标量 `sin`/`exp` 仍然要被调用；
- `float32` 相位归约造成的低位丢失仍然存在；
- 错误的数值结果不会因为线程绑定而变正确。

因此 `env.sh` 只能帮助一个已经正确的内核更稳定地运行，不能替代算法和数据布局优化。本题的主要加速必须写在 `src/solver.py` 中。

## 7. 远程验证过程

本机只有 8 个逻辑 CPU，不适合估计比赛性能。验证分两层进行。

### 7.1 `legion` 私人主机

`legion` 是 24 核 x86 主机，适合检查：

- Python 接口和输出 dtype；
- 样例答案；
- 公开大样例的误差；
- 不同实现的相对趋势。

它不是 Kunpeng ARM 节点，所以不能直接代表 OJ 的绝对时间，也不能运行 ARM NEON 路径。

### 7.2 比赛 CPU 集群

通过 `ssh arm` 进入登录节点，再向 `kp_run` 提交 Slurm 作业，在计算节点上运行。节点信息为 Kunpeng-920、AArch64、GCC 10.3.1。典型作业命令是：

```bash
sbatch --wait --partition=kp_run --qos=kp_run \
  --cpus-per-task=32 --ntasks=1 --time=00:10:00 \
  --wrap='cd ~/hellohpc-03/03-accelerate; ...'
```

正式评测使用的 `/vault/public/xflops/bin/hellohpc` 是一个可执行 CLI，不是可以直接 `import hellohpc` 的 Python 包。节点上的 NumPy 位于：

```text
/usr/local/lib64/python3.9/site-packages
```

因此独立调试脚本需要设置：

```bash
export PYTHONPATH=/usr/local/lib64/python3.9/site-packages
```

这只是远程调试环境问题，提交物仍然只有题目允许的 `solver.py` 和 `env.sh`。

## 8. 正确性检查方法

每次改动都做四类检查：

1. `python3 -m py_compile src/solver.py`，确认 Python 文件能加载；
2. 运行 `benchmark.py sample`，确认形状、dtype、有限性和样例答案；
3. 加载 `public-large-b.npz`，逐元素比较参考文件，使用题目规定的绝对/相对误差；
4. 在 ARM 计算节点上做完整大样例调用，确认不是只在小输入上正确。

还额外构造了高相位输入，把 `trig_scale` 放大到 `10^5`、`10^7` 甚至更高，用来检查 `sin` 相位归约。这个测试发现了公开数据看不出的错误，也解释了为什么早期全 NEON 版本在 OJ large-b 上返回 `wrong_answer`。

## 9. 最终代码的执行流程

一次 `compute_field` 调用可以按下面的顺序理解：

1. Python 检查所有形状和长度；
2. 将输入转成 C-order `float32` 数组；
3. 分配 `float32` 输出数组；
4. 调用导入阶段已经编译好的 C/OpenMP 内核；
5. ARM64 且 D=16 时，转置查询点并使用 NEON 四路并行；
6. 其他情况使用双精度距离、点积和数学函数；
7. 每个查询点得到一个双精度总和；
8. 将总和转换成 `float32` 并返回。

整个调用同步完成，没有把正式计算放到计时区间外，也没有启动外部计算服务、GPU、预计算答案或读取参考文件。

## 10. 当前结果与后续方向

最新 OJ 结果为：

| 用例 | 结果 | 性能 |
| --- | --- | ---: |
| c-small-d16 | accepted | 5.49 ms |
| c-mid-d16 | accepted | 9.81 ms |
| c-mid-d32 | accepted | 57.05 ms |
| c-large-d16 | accepted | 19.11 ms |
| large-a | accepted | 271.49 ms |
| large-b | success | 3298.33 ms |

当前版本的重点是数值正确性和可读性。若继续冲击 large-b，需要实现经过高相位压力测试的双精度向量正弦/指数函数，或找到能证明输入相位范围的题目约束；在没有这两项证据前，直接把 D=32 改回单精度 NEON 会重新引入 `wrong_answer` 风险。

