# MaiMoe 题解记录

## 1. 题目在做什么

程序为每张 MaiMoe 谱面读取文本，解析谱面事件，计算状态，渲染开始和结束画面，再生成结果记录。输出不是普通截图：每张谱面要产生两个带头部和 SHA-256 摘要的 RGBA 文件，以及一份记录事件统计、状态摘要和画面摘要的结果文件。评测会逐张检查输出，因此加速不能改变任何输出字节。

题目只允许改 `solution/Kernel.cpp` 和 `solution/KstroParam.toml`。引擎、调度器、checker、benchmark 和数据集是只读的。我只在允许的 Kernel 入口和配置上做优化，保留官方解析、状态计算、渲染、散列和序列化实现。

## 2. 先读 Kernel 的数据路径

`process_chart` 是每张谱面的计算入口。原实现先额外扫描整段输入，统计行数和最长行，再复制一份 `std::string`。随后调用引擎的 `process_chart`，引擎为三个输出文件建立独立对象和字节向量。Kernel 又把这三个向量逐字节拼到一个固定大小缓冲区，再对拼好的缓冲区重新计算折叠校验、重算两张画面的 SHA-256，并重复核对结果记录中的摘要和 ID。

这些多余步骤位于可信引擎之前或之后，并没有增加评测所需的独立功能。引擎解析器本身会验证格式和长度，编码器本身会生成固定布局的输出；最终评测还有独立 checker 验证文件集合和内容。因此 Kernel 改为调用引擎已有的 `process_chart_into`，让它把三个文件直接编码到调用方提供的固定输出缓冲区。

这样仍然执行完整的谱面解析、状态计算、画面渲染、SHA-256 和结果编码，只是避免额外输入复制、输出向量分配、拼接复制和重复的摘要校验。`static_assert` 检查两边约定的输出缓冲区大小一致。没有根据样例内容、谱面 ID 或数据集名称绕过处理。

## 3. 配置流水线和线程数

原配置只有一个计算 worker，关闭输入预取、异步读取和并行写出。题目允许最多 8 核，因此先将计算 worker 调到 8，并增加 pipeline 内存预算、每次领取工作的批大小和最长工作优先调度。

`work_lpt` 是 Longest Processing Time first 的缩写，意思是优先处理 manifest 中估算工作量较大的谱面。这样可避免一个 worker 最后独自承担特别大的剩余任务，缩短多 worker 同时工作时的尾部等待。排序来自公开 manifest 的工作量字段，等价于通用负载均衡，不针对某个样例答案。

第一版还打开了 2 个 I/O worker 和分区输出队列。Sample2 测试中，benchmark 检测到候选主进程退出时仍有子进程，日志指出只读输出队列的计数不变量被触发。这个组合不适合提交，所以关闭并行读取和写出，保留 8 路计算 worker 与安全的直接输出路径。修订配置通过了后续 benchmark 的 checker。

## 4. 测试过程

按照约定，没有在当前电脑运行构建或 benchmark。官方样例数据从 `arm` 登录入口只读传出，随后放入 `legion` 上独立的 `maimoe-20260923` 工作目录；编译、运行和计时均在 `legion` 完成。数据约 142 MB。`legion` 是 x86_64，而题目评分环境是 ARM，因此远端结果主要用于验证正确性、线程配置和优化方向；绝对运行时间不等同于正式 ARM 分数。

远程构建命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target maimoe maimoe_kernel maimoe_check -j8
```

先直接运行 Sample1 并调用 checker，结果是 `1024` 张谱面全部通过，输出 `3072` 个文件。之后使用题目提供的 `run_benchmark.py`，每个样例先 warmup 一次，再正式运行三次并求均值；该脚本每次都会调用 checker。

| 样例 | 三次平均 wall time | checker |
| --- | ---: | :---: |
| Sample1 | 97.696 ms | 通过 |
| Sample2 | 215.764 ms | 通过 |
| Sample3 | 188.156 ms | 通过 |
| Sample4 | 822.102 ms | 通过 |

题面给出的 Sample1 满分时间是 100 ms，当前远程 x86 测量低于该值；其它样例也都低于各自基础时间，但 Sample2、Sample3、Sample4 尚未达到满分时间。因为测试机器与正式 ARM 平台不同，不能据此推断正式得分。正式评测将再次由 checker 检查完整输出。

## 5. 最终核对

最终策略保留全部官方引擎计算，只改 Kernel.cpp 的内存路径以及 TOML 中的合法调度参数。提交时按题面只提交 `solution/Kernel.cpp` 与 `solution/KstroParam.toml`。本次没有改 `src/`、`include/`、`tools/`、评测数据或 checker，也没有在登录节点运行计算任务。
