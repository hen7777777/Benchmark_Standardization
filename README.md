# 统一评测入口（只改 checkpoint）

一个入口跑完当前这几组 benchmark。必填参数是模型 checkpoint 路径和 `--gpu`（物理卡号）：

```bash
./evaluate.sh /path/to/checkpoint --gpu 1
```

`--gpu` **不会自动选择**：先 `nvidia-smi` 看哪张卡空，再显式指定。

checkpoint 默认**只加载一次**：预检全部通过后启动一个共享 vLLM 服务，所有阶段共用，
全部跑完才释放（详见下文「统一部署 vLLM」）。

-   `math`、`code`、`complex_reasoning`、`general_reasoning`、`longbench_v1`、`longbench_v2` 用 **lm-evaluation-harness 自带任务**（不传 `--include_path`，不加自定义 YAML）。
-   `ruler` 用**本项目冻结的本地数据**（`datasets/long_context/ruler`），因为 harness 内置的 `ruler` 组每次加载都现场生成 500 条/任务/长度，还要联网抓 SQuAD/HotpotQA，无法复现固定样本。
-   `longbench_v1_cot`、`longbench_v2_cot` 是 **LongBench 的 CoT 变体**（独立阶段、默认不评，见 `longbench_cot/PROTOCOL.md`），用 `--include_path` 加载自定义任务 YAML，不覆盖上面的直答阶段。

## 目录

```
evaluate.sh            唯一入口：探测 Python 环境后调用 run_builtin_suite.py
run_builtin_suite.py   编排：各阶段预检并行执行 → 启动一次共享 vLLM → 逐阶段 python -m lm_eval run → 汇总
vllm_server.py         共享 vLLM OpenAI 服务的生命周期（启动 / 探活 / 退出时终止整个进程组）
preflight.py           内置任务的预检：解析任务名、查 tokenizer 与数据集缓存
preflight_ruler.py     RULER 预检：用当前 checkpoint tokenizer 逐题重算长度，超窗口直接拒绝
prefetch.py            一次性把内置任务需要的数据/模块下载进 cache/（联网一次）
summarize.py           把原生 results_*.json 汇总成 summary.md/csv/json
requirements.txt       依赖记录：已验证的直接依赖版本与 harness 安装约束
docs/INSTALL.md        环境搭建、editable harness、继承环境冲突说明与自检命令
builtin_tasks.json     阶段、任务名、主指标、默认参数（唯一需要改的配置）
ruler/tasks.json       冻结 RULER 的任务规格（13 任务 × 样本版本 × 长度），运行时据此生成 harness 任务 YAML
ruler/runtime_utils.py RULER 的 doc_to_text / process_results（聊天模板渲染与官方包含式评分）
ruler/generation_scripts/ RULER 数据再生成脚本：generate_ruler.py / generate_worker.py / prepare_sources.py / harness_utils.py
ruler/sources/          冻结的生成输入（tokenizer、RULER generator、essays、nltk 数据、task_configs.yaml），再生成不联网
ruler/metadata/         生成种子、长度统计、抽样索引、可复现性校验、依赖锁 requirements.lock.txt
ruler/EXPERIMENTAL_DETAILS.md RULER 数据与评测协议（可直接对应论文方法段）
ruler/RESULTS.md         Qwen3-0.6B 原生范围测试结果
longbench_cot/          LongBench v1/v2 CoT 变体：build_tasks.py 生成任务 YAML，tasks/runtime_utils.py 渲染与评分，PROTOCOL.md 记录协议与偏差
docs/                   使用说明（BENCHMARK_USAGE.md）、长上下文评测协议（LONG_CONTEXT_EVAL_PROTOCOL.md）、小模型 benchmark 说明（SMALL_MODEL_BENCHMARKS.md）
datasets/              冻结数据集（当前只有 long_context/ruler，约 3.8 GB，纳入版本控制；其中 9 个文件 >90 MB，
                       用 GitHub 远端会被 100 MB 单文件上限拒绝，需要本地远端或 Git LFS）
cache/                 HF datasets/modules/hub 缓存、nltk 数据（不进 Git）
runs/                  每次运行的结果、命令、日志、退出码
```

## 首次使用

**先准备环境。** 唯一要求是有一个能导入 `lm_eval` / `transformers` / `datasets` / `evaluate` 的
Python 环境（已验证 Python 3.13.5 + vLLM 0.17.1 + torch 2.10.0 + transformers 4.57.1），
并按固定 commit editable 安装 harness。版本清单、从零安装步骤和两个坑（不要用
`lm_eval[vllm]`、不要用 `lm_eval[math]`）写在 `requirements.txt` 与 `docs/INSTALL.md`；
已有环境直接在 `suite.local.env` 里指 `LMEVAL_PYTHON` 即可，不必重装。

`math`、`code`、`complex_reasoning`、`general_reasoning` 与 `longbench` 的数据来自 HuggingFace Hub。本机没有直连，需要走代理；`prefetch.py` 默认使用 `127.0.0.1:17891`：

```bash
python3 prefetch.py            # 联网一次，把内置任务的数据与 evaluate 的 code_eval 模块抓进 cache/
```

RULER 不需要这一步：它的数据就在 `datasets/long_context/ruler`。

然后写一个不进 Git 的 `suite.local.env`：

```bash
LMEVAL_PYTHON=/path/to/venv/bin/python
# 可选：RULER_DATA_ROOT=/absolute/path/to/datasets/long_context/ruler
```

再用入口跑：

```bash
./evaluate.sh <checkpoint> --gpu 1 --smoke      # 每任务 1-2 条，先确认链路
./evaluate.sh <checkpoint> --gpu 1              # 默认：四个能力阶段 + longbench_v1 + longbench_v2
./evaluate.sh <checkpoint> --gpu 1 --stages all # 再加上 LongBench CoT 变体 + ruler（ruler 5200 条，很慢）
./evaluate.sh <checkpoint> --gpu 1,3 --tensor-parallel-size 2   # 一个模型切两张卡
./evaluate.sh <checkpoint> --gpu 1 --stages ruler --ruler-samples 10 --ruler-lengths 4096   # 最小 RULER 验收
./evaluate.sh <checkpoint> --gpu 1 --stages longbench_v1_cot,longbench_v2_cot              # 单独跑 LongBench CoT 变体
```

## 统一部署 vLLM（默认：模型只加载一次）

默认后端 `--model-backend server`。预检全部通过后，套件只启动**一次**
`python -m vllm.entrypoints.openai.api_server`，把 checkpoint 装进这一个进程；每个阶段
改用 harness 自带的 `--model local-completions`，通过 HTTP 打同一个端口，最后在 `finally`
里停掉整个进程组。改动前的 `--model vllm` 会把引擎建在**每个** `lm_eval run` 进程里，
所以每个阶段都要重新加载一次权重、重新热身。

```bash
./evaluate.sh <checkpoint> --gpu 1                             # 默认：统一部署，跑完即释放
./evaluate.sh <checkpoint> --gpu 1 --model-backend inprocess   # 旧行为：逐阶段各自建引擎
./evaluate.sh <checkpoint> --gpu 1 --num-concurrent 8          # 每阶段最多 8 个并发 HTTP 请求
./evaluate.sh <checkpoint> --gpu 1 --server-port 39000         # 固定端口（默认自动挑空闲端口）
```

-   **加载次数**：整次运行只加载一次模型。L40 上 Qwen3-0.6B 的 `SERVER READY` 实测约 17 s，
    六个阶段共用这一次加载；模型常驻也让各阶段共享 KV / prefix cache。
-   **参数对应**：服务端参数逐项对应旧的引擎参数（`--dtype bfloat16`、`--max-model-len`、
    `--tensor-parallel-size`、`--gpu-memory-utilization`、`--enforce-eager`、`--max-num-seqs`、
    `--max-num-batched-tokens 2048`、`--generation-config vllm`、`--seed`），并按同一个 `--gpu`
    设置 `CUDA_VISIBLE_DEVICES`。
-   **并发推理**：每阶段的 `num_concurrent` 默认取该阶段的 `max_num_seqs`
    （数学 / 代码 / 复杂推理 / 日常推理为 4，两个 longbench 为 1），即旧引擎能同时调度的批大小；
    `--num-concurrent N` 可整体上调。真正做 continuous batching 的是 vLLM，客户端只负责把请求压进去。
    阶段之间仍然串行，所以显存峰值与单阶段时相同。
-   **释放时机**：`runs/<name>/server.json` 与 `manifest.json` 的 `server` 字段记录 `argv`、端口、
    `pid`、`ready_seconds`，服务日志在 `runs/<name>/server.log`。正常结束、异常退出、Ctrl-C 都会在
    `finally` 里先 `SIGTERM`、必要时再 `SIGKILL` 整个进程组；子进程还加了 `PR_SET_PDEATHSIG`，
    套件自己被 `kill -9` 时内核会连带杀掉服务，避免共享卡上留下常驻模型。
    `--server-keep-alive` 可显式保留服务（仅调试用，会一直占着卡）。
-   **端口与依赖**：默认绑 `127.0.0.1` 上的空闲端口，只有本机可访问。`/v1/completions` 需要
    `aiohttp` / `requests` / `tenacity`（见 `requirements.txt`）；缺失时套件在启动服务之前就报错，
    并提示改用 `--model-backend inprocess`。
-   **预检失败不占卡**：服务在预检全部通过之后才启动，因此任务名、数据缓存或长度有问题时，
    不会先把模型加载进显存。

## GPU 选择与退出看门狗

**不自动选卡。** `--gpu` 必填，值是 `CUDA_VISIBLE_DEVICES` 的物理卡号，个数必须与 `--tensor-parallel-size` 一致。共享机器上“看起来空闲”的卡随时可能被别人占走，把运行悄悄换到另一张卡会让结果难以归因，所以套件不做空闲探测；传 `auto` 会直接报错。要并行多个评测就分别指定卡号；同一张卡跑多个进程也是允许的（各自设相同的 `CUDA_VISIBLE_DEVICES`），套件不持有任何锁。

**结果已落盘但进程不退出时自动收尾。** 原生阶段一旦写出 `results_*.json`（实际路径是 `<stage>/results/<model_name>/results_<时间戳>.json`）就说明评测已经算完，剩下的只是退出；如果之后整个 `results` 目录和日志都静默超过 `--shutdown-grace` 秒（默认 60）仍不退出（vLLM 引擎常见），套件会向进程组发 `SIGTERM`、必要时再 `SIGKILL`。此时结果保留、该阶段按成功计，`runs/<name>/<stage>/exit.json` 的 `watchdog` 字段与 `native.log` 末尾会记录这次强制收尾，`manifest.json` 也记了 `shutdown_grace`。看门狗按“日志与 results 目录里最新文件时间”判断是否静默，因此不会把正在写 `samples_*.jsonl` 的正常收尾误判成卡死；想更保守可以调大，例如 `--shutdown-grace 180`。


## 预检并行执行

每个阶段在真正推理前会先跑一轮预检：解析任务名、核对 tokenizer 与数据集缓存、检查长度与截断。
预检不加载模型、不占用 GPU，各阶段之间互不依赖，所以默认**同时启动**，全部结束后一起汇总：

```
PREFLIGHT: 6 stage(s) starting in parallel (all at once)
PREFLIGHT PASSED: math (10.0s)
PREFLIGHT PASSED: code (10.5s)
...
PREFLIGHT DONE: 6/6 passed in 20.0s wall (74.6s of stage time)
```

失败也不会互相取消：所有阶段都跑完才报错，错误里列出全部失败阶段，日志和 `preflight_exit.json`
按阶段分开存放，因此一次就能看全，不必逐个重跑。`--preflight-jobs N` 可以限制并发数
（默认 `0` = 全部同时启动）；`--preflight-only` 仍然只跑这一轮预检就退出。
`manifest.json` 记录 `preflight_jobs` 与该轮总墙钟时间 `preflight_wall_seconds`。


## 阶段选择（--stages）

类别通过 `--stages` 指定（不是 `--task`），接受 `all` 或逗号分隔的阶段名，例如：

```bash
./evaluate.sh <checkpoint> --stages math                     # 单个类别
./evaluate.sh <checkpoint> --stages math,code                # 多个类别
./evaluate.sh <checkpoint> --stages longbench_v1_cot,longbench_v2_cot   # 只跑 CoT 变体
./evaluate.sh <checkpoint> --stages all                      # 全部阶段
```

可选阶段共 9 个：`math`、`code`、`complex_reasoning`、`general_reasoning`、
`longbench_v1`、`longbench_v2`、`longbench_v1_cot`、`longbench_v2_cot`、`ruler`。
默认值（不传 `--stages`）是前六个直答阶段，不含 CoT 与 `ruler`；`--stages all`
会把这些一起算进去。冒烟验证加 `--smoke --limit 2`。

## 阶段与任务

| 阶段                | 类别     | 来源                           | 任务名                                       | 主指标                                                     |
| ------------------- | -------- | ------------------------------ | -------------------------------------------- | ---------------------------------------------------------- |
| `math`              | 数学     | harness 内置                   | `gsm8k_cot`                                  | `exact_match,strict-match`；`exact_match,flexible-extract` |
|                     |          |                                | `minerva_math500`                            | `exact_match,none`；`math_verify,none`                     |
| `code`              | 代码     | harness 内置                   | `humaneval` / `humaneval_plus`               | `pass@1,create_test`                                       |
|                     |          |                                | `mbpp` / `mbpp_plus`                         | `pass_at_1,none`                                           |
| `complex_reasoning` | 复杂推理 | harness 内置                   | `bbh`                                        | `exact_match,get-answer`                                   |
|                     |          |                                | `drop`                                       | `em,none`；`f1,none`                                       |
|                     |          |                                | `mmlu_pro`                                   | `exact_match,custom-extract`                               |
| `general_reasoning` | 日常推理 | harness 内置                   | `mmlu`                                       | `acc,none`                                                 |
| `longbench_v1`      | 长上下文 | harness 内置                   | `longbench`（21 个数据集）                   | `score,none`                                               |
| `longbench_v2`      | 长上下文 | harness 内置                   | `longbench2`                                 | `acc,none`                                                 |
| `longbench_v1_cot`  | 长上下文 | 自定义 CoT（`--include_path`） | `longbench_v1_cot`（19 个数据集）            | `score,none`                                               |
| `longbench_v2_cot`  | 长上下文 | 自定义 CoT（`--include_path`） | `longbench_v2_cot`（20 个子数据集）          | `acc,none`                                                 |
| `ruler`             | 长上下文 | 本地冻结数据                   | `ruler_local_<样本数>_<长度>`（13 个子任务） | `ruler_score`（0-100）                                     |

## LongBench CoT 阶段说明

`longbench_v1_cot` 与 `longbench_v2_cot` 是把 LongBench 的两条直答链路改成“先推理、后作答”的独立阶段，**默认不参与**，需要显式 `--stages longbench_v1_cot,longbench_v2_cot`（可单独跑其中一个）。它们用 `--include_path longbench_cot/tasks` 加载运行时生成的任务 YAML，评分工具在 `longbench_cot/tasks/runtime_utils.py`。

-   v1 CoT：官方没有 CoT 协议；本变体在官方直答提示词后追加 `Let's think step by step`，取最后一个 `Final answer:` 之后的内容按官方 per-dataset 指标打分，排除 `lcc` / `repobench-p` 两个代码补全任务。
-   v2 CoT：官方是两段式 CoT（`0shot_cot.txt` 生成推理 + `0shot_cot_ans.txt` 生成答案）；lm-eval 单次生成只能合并为一段，提取正则与 `acc` 与官方 `pred.py` 完全一致。

完整协议、官方出处与逐项偏差见 `longbench_cot/PROTOCOL.md`。

## RULER 阶段说明

**数据。** `datasets/long_context/ruler/<10samples|50samples|100samples>/<4k|8k|16k|32k|64k|128k|256k>/<任务>.jsonl`，13 个 RULER v1 官方任务 × 7 个名义长度 × 三个嵌套样本版本。目录名与 token 数的对应：4k=4096、8k=8192、16k=16384、32k=32768、64k=65536、128k=131072、256k=262144。

| 目录         | 每任务每长度 | 每个长度的 13 任务总数 | 七个长度的记录数 |
| ------------ | -----------: | ---------------------: | ---------------: |
| `10samples`  |           10 |                    130 |              910 |
| `50samples`  |           50 |                    650 |            4,550 |
| `100samples` |          100 |                  1,300 |            9,100 |

三个版本共用同一批唯一样本（`10samples` ⊂ `50samples` ⊂ `100samples`），不是三个独立数据集。默认入口评 `100samples` 版的 4k/8k/16k/32k：13 × 4 × 100 = **5,200 条**。

**抽样规则。** 完整协议写在 `datasets/long_context/ruler/SAMPLING_PROTOCOL.md`，可直接对应论文方法段；精确索引见 `sampling_subsets.json`，长度与答案深度统计见 `length_statistics.json`，QA 题目来源见 `qa_sampling.json`，生成种子见 `generation_protocol.json` / `generation_commands.json`。要点：

-   主种子 `S=20261004`。每个任务/长度 100 条母集的生成种子是 `S + 1000*j + t`。
-   子集选择种子 `S + 100000 + t`（与长度无关）：对 `range(100)` 做 `random.Random(seed).shuffle` 无放回排列，10 版取前 10、50 版取前 50、100 版保留原池 0–99 顺序。同一任务的各长度用同一组子集索引。
-   QA 从 SQuAD v2 可回答 dev 问题与 HotpotQA dev distractor 问题里无放回抽 100 条，种子 `S + 200000 + t`，问题身份跨长度固定。
-   没有按模型成绩、答案难度或 needle 位置筛样；没有按深度分层。
-   长度用固定参考 tokenizer `Qwen/Qwen3-0.6B` 计数，包含完整提示、非 thinking 聊天模板、官方 answer prefix 与任务输出预算。名义 4K 不代表每题恰好 4096 tokens。

**评测协议。** 冻结题目不因 checkpoint 变化而重新生成。用**当前 checkpoint** 的聊天模板渲染同一 input（单 user turn、`add_generation_prompt=True`、`enable_thinking=False`），再拼该题官方 answer prefix；不在 harness 外层重复套模板。评分沿用官方大小写不敏感的字符串包含规则（QA 取任一 alias 命中），每任务样本均值，每长度对 13 任务等权平均，不做跨长度或跨 benchmark 合成总分。

**超长处理。** 换 checkpoint 后 tokenizer 不同，实际 token 数会变。`preflight_ruler.py` 用当前 checkpoint tokenizer 逐题重算，超过有效窗口直接报错退出，不截断、不静默丢题。逐题审计写在 `runs/<name>/ruler/input_lengths.jsonl`，含实际 token 数、参考 token 数与差值。

**参数。**

```bash
--ruler-samples {10,50,100}          # 默认 100
--ruler-lengths 4096 8192 ...        # 默认 4096 8192 16384 32768；不得超过 --context-length
--ruler-data-root /path/to/data      # 默认 <suite>/datasets/long_context/ruler，也可用 RULER_DATA_ROOT

任务名由 `ruler/tasks.json` 里的映射决定：`--ruler-samples 100 --ruler-lengths 4096` 生成 `ruler_local_100samples_4k`（13 个子任务 + 1 个聚合组），仅当对应目录存在 13 个 jsonl 时才会运行。
```

64K/128K/256K 数据完整保留，但需要更大的 `--context-length`（例如 `--context-length 262144 --ruler-lengths 262144`），且必须先确认模型确实支持该窗口。

## RULER 数据再生成

冻结数据本身不需要重新生成；只有复现或换参数时才用 `ruler/generation_scripts/` 从 `ruler/sources/` 重新生成。生成完全离线（`HF_HUB_OFFLINE=1`），只依赖已冻结的 `sources/`：

```bash
cd ruler/generation_scripts
python3 generate_ruler.py --run-dir ../../runs/ruler_generation --workers 8
```

-   `generate_ruler.py` 遍历 7 个长度 × 13 个任务，生成每个长度/任务 100 条母集，再按种子切出 10/50/100 三个嵌套子集，写出任务 YAML 与 `metadata/generation_commands.json`、`metadata/subsets.json`、`metadata/length_statistics.json`。
-   `generate_worker.py` 是单个长度/任务的实际生成器（由 `generate_ruler.py` 并发调用），直接运行官方 RULER 任务脚本，不经过会吞子进程错误的 `prepare.py`。
-   `harness_utils.py` 是生成任务 YAML 引用的 `doc_to_text` / `process_results`（旧式独立任务接口；统一入口评测改用 `ruler/runtime_utils.py`）。
-   `prepare_sources.py` 只用于**冷启动重建** `sources/`（联网下载 tokenizer、essays、SQuAD/HotpotQA、nltk 数据），需 `--ruler-repo` 指向 RULER 源码仓库。因 `sources/generator` 已是官方 `scripts/data` 快照，普通再生成不需要它。

生成输出在 `--run-dir` 下，目录形如 `data/{10,50,100}/{4096..262144}/<task>.jsonl`。要替换评测用的冻结数据，把它重命名映射到 `datasets/long_context/ruler/{10samples,50samples,100samples}/{4k..256k}/<task>.jsonl`（`10→10samples`、`4096→4k` 等）。固定种子与抽样索引在 `ruler/metadata/` 与 `datasets/long_context/ruler/sampling_subsets.json`，逐项协议见 `ruler/EXPERIMENTAL_DETAILS.md`。

`ruler/sources/`（约 161 MB）与 `datasets/long_context/ruler`（约 3.8 GB）都纳入版本控制，换机器不联网也能再生成和评测。

## 已知边界

-   `--smoke` 只验收流程：RULER 取每个任务/长度前 1 条，默认共 52 条，不是 10 条随机版本的成绩，也不是任何完整分数。
-   HumanEval+ 用 harness 内置任务 `humaneval_plus`（数据集 `evalplus/humanevalplus`），主指标 `pass@1,create_test`；首次使用需联网 prefetch 一次该数据集。
-   两个套件的分数**不可混用**：`std_*` 套件有自己的 few-shot 与模板，本套件的这四个能力阶段与 longbench 用 harness 原生默认。
-   `requirements.txt` 是已验证版本的记录，不是传递依赖锁，也尚未在全新环境完成从零安装与推理复测；L40 实测的 `numpy==2.4.4` / `setuptools==84.0.0` 与两者的声明上限冲突，属继承环境现象，不要为了消除冲突去升降版本。
-   `cache/` 约 1.2 GB，不进 Git；换机器要重跑一次 `prefetch.py`。`datasets/long_context/ruler` 约 3.8 GB，纳入版本控制，换机器不需要重新生成。目录名用 `100samples/4k` 这种可读写法，协议 JSON 内部仍以 token 数记录长度。
-   RULER 的 `ruler_score` 是包含式评分（0-100），与其它阶段的 accuracy 不是同一量纲，汇总表里不要跨阶段平均。
