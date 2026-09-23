# MahjongX 解题记录

本文记录 09 题当前版本的工作方式、运行流程和优化理由。目标是让只熟悉基本 Linux 命令的读者可以跟着复现。当前优化已经在本地源代码中完成，但完整验证仍在 legion 上进行；legion 是 x86_64 WSL2，不是题目指定的鲲鹏 920，因此它只能用于正确性对拍和相对性能观察，不能代表官方排行榜耗时。

## 1. 题目要做什么

`system` 从标准输入逐行读取一局麻将牌谱。启动参数中的配置文件包含对局规则和策略，掩码 `15` 表示分析四名玩家。摸牌事件会触发对摸牌玩家的分析；弃牌或加杠事件会触发对其他玩家的分析。

每个有至少两个候选动作的局面，程序输出一个 `POS` 候选块，其中包含各动作及分数。实际发生的动作确定后，还要输出 `ACTUAL` 结果块。标准错误则需为每次要求的分析输出 `FULL_ANALYZE_STEP` 进度标记。检查器会比对候选动作、分数、可选指标和实际动作；浮点绝对或相对误差上限是 `1e-4`。

输入是管道，程序不能回退或等待读完整局后再处理。交互器每次发送一个事件，需要分析时会等待进度标记，收到后才继续发送。因此 `stdout` 和 `stderr` 的及时 flush 是协议的一部分。

## 2. 规则与计时边界

遵守比赛的个人赛、禁止攻击、禁止针对固定牌谱或参考答案特判、禁止把计时内计算搬到计时外，以及登录节点不得运行高负载程序等规定。只对通用算法和实现做优化，并保留完整的麻将规则与策略分支。

评测按以下次序工作：先运行 `build.sh`，然后每份牌谱运行 `bash run.sh <setup-file>`。`run.sh` 的标准输入是交互器传入的牌谱，它只设置环境并启动 `src/system`，不能自行读取样例或调用交互器。

计时在读取第一行牌谱前开始。配置与策略初始化在计时开始前；牌谱解析、每次策略分析、输出和进度标记均在计时段内。计分取每份牌谱最后一条有效 marker 的 `ELAPSED_SECONDS`，公开样例 game1 和 game2 的用时相加后按 1000 秒满分控制线、2000 秒零分线计算。本地公开样例通过只说明正确性，不代表最终排行榜分数。

可修改的提交文件只有 `build.sh`、`run.sh` 和 `src/**`。`checker/`、`interactor/` 和 `judger/` 是只读评测组件；提交包应使用 `hellohpc pack --output submission.zip` 生成，包含构建脚本、运行脚本和源码，不应混入样例、二进制或评测器。

## 3. 基线和运行环境

原始 Makefile 已经使用 GCC 的 `-O3`、OpenMP 和 pthread，因此单纯把 Debug 改成 Release 不适用于本题。工作区构建脚本会在编译时按可见 CPU 数量定义 `NPROCS`，它决定内部工作区大小；OpenMP 运行线程数则由 `OMP_NUM_THREADS` 控制。

按用户要求，测试只在 `ssh legion` 的私人 WSL2/Ubuntu 24.04 主机上运行，没有在当前电脑或 SJTU 登录节点执行。legion 有 24 个可见 CPU，x86_64，编译器 GCC 13.3；通过 micromamba 创建了独立环境 `~/conda-envs/mahjx`，安装 Boost 1.84、CMake 和构建工具。它没有昇腾设备，也没有官方 ARM CPU，所以不能替代官方容器验证指令集、编译器差异或鲲鹏性能。

原始源码在 legion 上成功构建。使用 24 个 OpenMP 线程运行 game1，耗时约 456 秒，检查器通过。game2 曾在 8 线程运行，耗时约 495 秒，但 event 207/player 3 的候选分数检查失败；因此这个结果不能用于评估改动。现正以 24 线程重新运行 game2，先确认同一构建和线程配置下是否可重现。不同机器、不同线程数或编译器上的数字不应混为一谈。

在 legion 上重建原始代码的示例命令如下（路径按实际工作副本调整）：

```sh
export CPATH="$HOME/conda-envs/mahjx/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$HOME/conda-envs/mahjx/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$HOME/conda-envs/mahjx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make -C src -j 8
make -C src test
```

交互器运行示例：

```sh
export OMP_NUM_THREADS=24
export OMP_DYNAMIC=FALSE
interactor/build/interactor sample/game_record_game1.txt 15 -- \
  ./src/system full_analyze sample/setup_match.txt 15 \
  > game1.out 2> game1.err
checker/build/checker sample/game_record_game1_full_analyze.txt game1.out
```

这里 `game1.err` 是进度标记和诊断信息；正确性检查器读取 `game1.out`。正式运行由评测器通过 `bash run.sh` 启动。

## 4. 当前优化

### 4.1 让小型计分表充分并行

`cal_round_end_pt_exp` 生成一个以两个玩家编号开头的四维计分表。前两个维度各只有 4 个取值，一共 16 个互相独立的玩家组合。原实现只对第一个维度做 OpenMP 循环，所以最多只有 4 个任务；在 24 核或 128 核机器上，大多数线程没有工作。

现在对前两个循环使用 `collapse(2)`，让 OpenMP 把 16 个玩家组合一起分配给线程。每个组合内部仍按原来的 han、fu 顺序计算，并且每个线程写入不同的表格单元，因此不改变单个数值的浮点运算顺序或数据依赖。它主要减少这段计算的等待时间；总收益取决于它在整道题运行时间中的占比。

### 4.2 避免重复扫描牌谱

`set_selector` 一次只分析一个局面，当前牌谱在该次分析过程中不变。原实现多次调用 `count_tsumo_num_all(game_record)`，这个函数需要扫描已有事件；有些调用还位于候选动作循环中。现在在函数开头扫描一次并保存结果，后面都读取这个局部整数。

它不缓存不同局面或不同运行之间的答案。每次分析都仍根据当前输入重新计算摸牌数；只是把同一次分析中重复进行的相同扫描合并为一次。局面数据、候选动作和分数公式均未改变。

## 5. 验证步骤

每次改动都应在 legion 重新构建，运行 `make -C src test`，再依次用交互器跑 game1 和 game2，并用各自的参考文件运行 checker。确认 `stderr` 最后一条 marker 的累计时间有限且为正数，交互器正常退出，两个 checker 均报告误差统计并以退出码 0 完成。当前记录中的 game1 基线通过；优化后的两份样例对拍及 game2 同配置基线复核尚待完成，不将其写成已验证。

性能比较至少固定编译器、`NPROCS`、`OMP_NUM_THREADS` 和牌谱。样例分析局面成本不同，机器调度也会造成波动；正式优化结论应多次运行取中位数。所有性能数据需明确标注机器和线程配置，并以官方提交评测结果为准。

## 6. 提交前清单

- [ ] game1 和 game2 均通过交互器、checker 和协议检查。
- [ ] 没有硬编码牌谱、事件、随机种子、玩家或参考输出。
- [ ] `build.sh` 能在评测环境编译，`run.sh` 从管道读入并立即转发分析结果。
- [ ] 只打包允许提交的原始源码与脚本，不打包编译产物或评测组件。
- [ ] writeup 记录的是实际运行数据；legion 测试明确标为非鲲鹏参考数据。
