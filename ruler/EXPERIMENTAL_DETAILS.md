# RULER 数据与评测协议

本实验使用 RULER v1 的官方 13 个任务配置，自行生成离线数据，用 EleutherAI lm-evaluation-harness 的外部 YAML 任务接口评测。没有使用 RULER v2，没有使用 harness 默认在线生成的 500 条/任务数据。精确代码提交、tokenizer revision、Python 版本见 `metadata/protocol.json`，全部 Python 依赖见 `metadata/requirements.lock.txt`。

## 长度与样本规模

长度为 4,096、8,192、16,384、32,768、65,536、131,072、262,144 tokens，K 定义为 1,024。每个长度覆盖全部 13 个子任务。三档数据分别为每子任务 10、50、100 条，单个长度各为 130、650、1,300 条，七个长度各为 910、4,550、9,100 条。三档合计有 14,560 条文件记录，但只有 9,100 个独立的长度/任务/样本身份，不能把嵌套版本当作独立重复实验。

使用 `Qwen/Qwen3-0.6B` 的固定 tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`。已逐项确认词表与聊天模板与本地 `/data/lpg_models/Qwen/Qwen3-0.6B` 相同。固定快照保存在 `sources/tokenizer/`，生成和评测均不从网络加载 tokenizer。

长度指完整聊天 prompt 的 token 数，加上该任务预留的最大生成 token 数，上界为标称长度；不是恰好等于标称长度的纯正文。聊天模板使用一个 user turn、`add_generation_prompt=True`、`enable_thinking=False`，将官方 answer prefix 接在 assistant turn 起始位置。模板开销预留 12 tokens，并在生成后对实际渲染的 prompt 重新计数，超预算直接报错。任务内示例、任务说明、问题、answer prefix 和聊天模板均计入输入预算。不在 harness 中再次套聊天模板。

官方生成器通过搜索背景单位数量及减少背景长度满足预算；保持此算法，没有截断完整 prompt 来凑长度。因段落/句子/背景单位大小与重试不同，实际长度会略短于标称上限；各版本、各长度、各任务的最小/均值/最大长度及填充率见 `metadata/length_statistics.json`。论文应同时报告标称预算和实际长度分布。

## 13 个任务配置

| 子任务 | 官方配置 | 最大生成 tokens |
|---|---|---:|
| niah_single_1 | 重复噪声；词语 key、7 位数字 value；1 key/1 value/1 query | 128 |
| niah_single_2 | Paul Graham essays；词语 key、7 位数字 value；1/1/1 | 128 |
| niah_single_3 | essays；词语 key、UUID value；1/1/1 | 128 |
| niah_multikey_1 | essays；词语 key、数字 value；4 keys/每 key 1 value/1 query | 128 |
| niah_multikey_2 | 干扰 needle 背景；词语 key、数字 value；1 目标 key | 128 |
| niah_multikey_3 | 干扰 needle 背景；UUID key 与 value；1 目标 key | 128 |
| niah_multivalue | essays；1 key/4 values/1 query | 128 |
| niah_multiquery | essays；4 keys/每 key 1 value/4 queries | 128 |
| vt | 重复噪声；1 条变量链、4 hops；保留官方单个示例 | 30 |
| cwe | 10 个常见词各出现 30 次，其他词各出现 3 次；保留官方单个示例 | 120 |
| fwe | 6 字母编码词；Zipf 参数 2.0；忽略最高频噪声，输出其余 top 3；无额外示例 | 50 |
| qa_1 | SQuAD v2 dev 中可回答问题；加入干扰文档 | 32 |
| qa_2 | HotpotQA dev distractor；加入干扰文档 | 32 |

准确参数来自 `sources/task_configs.yaml` 和 `sources/generator/synthetic/constants.py`；未列出的参数保持该固定官方版本的默认值。CWE 长背景超过 wonderwords 词库时按官方实现使用 english_words 词表。NIAH multiple-query 的 4 个 key 是官方生成器将 key 数提升到 query 数后的结果。

## 随机性与嵌套抽样

主种子固定为 20261004。任务序号按 `sources/task_configs.yaml` 的插入顺序从 0 开始；长度序号按上述升序从 0 开始。每个长度/任务独立启动 Python 进程，以主种子加 1000 倍长度序号再加任务序号为 generation seed。直接运行官方任务脚本，不调用会捕获子进程错误的 `prepare.py`，不进行生成分块。固定 `PYTHONHASHSEED=0`、单线程 BLAS 和 `TOKENIZERS_PARALLELISM=false`；并发 worker 数不改变任务随机流。

保留官方合成采样规则：数字均匀随机、词 key 从固定 wonderwords 词库中随机抽取、UUID 使用 Python `random.getrandbits(128)` 和 UUID v4 格式。essay 背景按官方顺序连接与重复；needle 的插入位置从 40 个离散深度候选中无放回抽取。重复噪声和干扰 needle 背景的插入位置从背景单位位置均匀无放回抽取。多个 needle 的位置不独立；不得将其描述为每条样本都均匀覆盖全部 token 深度。VT 的链顺序和插入、CWE 的词选择与排列、FWE 的词频构造均保持官方算法；FWE 使用确定性 Zipf 频数，而不是把每次词出现都描述为独立随机抽样。

QA 不取验证集开头的 100 条。从 SQuAD 的 5,928 个可回答问题和 HotpotQA 的 7,405 个 dev distractor 问题中，分别均匀无放回抽取 100 条。QA selection seed 为主种子加 200000 再加任务序号，与长度无关，因此各长度使用相同源问题集合与顺序。SQuAD 不可回答问题按官方规则排除，没有按答案长度或模型表现筛选问题。干扰文档与长度搜索继续使用该长度/任务的 generation seed。每行保存 `qa_source_id`、`qa_source_index`、`qa_selection_seed`；不同长度只保证问题身份配对，不声称干扰文档或位置完全配对。

每个长度/任务生成 100 条后，以主种子加 100000 再加任务序号为 subset seed，使用 Python `random.Random(seed).shuffle` 打乱 0–99 的索引。10 条取排列前 10 个，50 条取前 50 个，100 条保留完整母集原顺序。子集 seed 与长度无关，QA 的 10/50 条源问题在不同长度间也配对。完整排列的实际选中索引、sample ID 和 seed 见 `metadata/subsets.json`；不根据模型输出或得分挑样，不做位置分层。小规模 10 条版本定位为流程测试，不能宣称覆盖所有深度；首个参考答案位置的十分位计数保存在长度统计中，它不是所有 needle 位置的统计。

Python 和 NumPy 的随机状态均由官方脚本设置。背景长度搜索和超预算重试也消耗随机数，属于生成过程的一部分。保留该行为；生成脚本、语料、词表、tokenizer、依赖版本与种子共同构成复现条件，单独给出 seed 不足以重建相同数据。

## 固定输入与溯源

QA 原文件、218 个官方 URL 对应的 essay 原始内容、解码后的文本、连接顺序及 english_words 的实际 Git LFS 文件都保存到 `sources/`。URL 与逐页编码见 `sources/corpus_sources.json`。下载时将 Paul Graham 的 HTTP 链接改为 HTTPS，将 GitHub raw 链接改为 raw.githubusercontent.com。文本解码使用 UTF-8，实际旧网页中的 Windows-1252 标点使用 Windows-1252 并逐页记录；不使用旧下载脚本的 `unicode_escape` 解码。HTML 仍按官方方式提取首个 font 元素并用 html2text 转换，repo 文本按文件名排序后在 HTML 文本之前连接。所有 URL 必须下载成功，没有静默跳过缺失 essay。未来完全复现应重用本地快照，而非重新下载可能改变的网页。

`sources/generator/` 是固定官方代码副本，没有改写合成任务算法。QA 的抽样通过 worker 载入官方脚本后、生成前在内存中重排 `QAS` 实现。官方 NIAH 输出的 `index` 实际是答案字符位置，本数据将其保存为 `upstream_index`，另用真正的样本序号作为 `index`；唯一身份是 `sample_id`。

## Harness 评测

外部任务 YAML 在 `tasks/`，每个长度/版本对应一个包含 13 任务的 group。任务读取固定 JSONL，不调用 harness 内置 RULER 的数据生成函数。RULER 得分使用官方不区分大小写的子串命中：非 QA 对所有要求答案的命中比例计分；QA 对任意可接受答案命中计分，再取样本均值。报告 0–100 分，各子任务均值以相同权重汇总，不提前对样本分数取整。特别保留官方 QA 的“任意 alias 命中”，没有采用会把全部 SQuAD aliases 当作必须同时返回的平均命中率。清理预测文本控制字符的方法与官方实现一致。

推理为 vLLM 0.17.1、PyTorch 2.10.0+cu128、bfloat16、temperature 0、do_sample false、seed 20261004。服务使用 `--generation-config vllm`，避免本地模型 generation_config 的额外采样设置改变协议；任务生成上限为表中预算。禁用 thinking，保留官方 answer prefix，harness 外层 `num_fewshot=0`；VT/CWE 内嵌示例仍然存在。服务与客户端均使用固定 tokenizer 快照，客户端发 token IDs 到 `/v1/completions`，不重复渲染模板。

客户端向 harness 传入的 `max_length` 为预算加 1，因为该固定版本 TemplateAPI 在构造时内部减 1；这样有效总长度上限恰好等于数据预算，避免其无条件左截断路径改变合法样本。每行 `prompt_tokens`、`max_gen_toks`、`total_tokens_reserved` 保存实际计数。实际执行命令见 `metadata/eval_commands_native_10.json`，服务日志见 `logs/vllm_native.log`，逐样本输出与 harness 结果保存在 `results/native/`。

使用两个独立 vLLM 副本分卡执行：GPU 1 服务端口 8013，评测 4k/8k/16k；GPU 3 服务端口 8014，评测 32k。两者模型、tokenizer、dtype、generation config、seed 与生成限制一致；不是 tensor parallel 模型切分。GPU 3 日志见 `logs/vllm_native_gpu3.log`。

复现检查使用独立的新 Python 进程重新生成全部 13 个任务的 4k、100 条母集，1,300 条记录逐字节一致；命令与结果见 `metadata/reproducibility_check.json`。该实测检查针对 4k；其他长度沿用相同固定输入与确定性代码，不声称逐个重生成检查过所有长度。全数据覆盖、嵌套一致性、QA 跨长度身份与长度预算检查见 `metadata/dataset_validation.json`。

按用户选择，当前只对原生上下文范围 4k、8k、16k、32k 跑每子任务 10 条的全任务测试。Qwen3-0.6B 模型卡标称 32,768 tokens，配置上限 40,960，`rope_scaling=null`。64k、128k、256k 数据完整保留，但本实验不修改 RoPE、不将这些长度的结果报告为原生上下文性能。10 条版本的测试结果不是 50/100 条版本的结果；尚未运行的组合不可填入结果表。

## 论文英文方法段落

We generated RULER-v1 evaluation data using all 13 official task configurations at total sequence budgets of 4,096, 8,192, 16,384, 32,768, 65,536, 131,072, and 262,144 tokens, measured with a pinned Qwen3-0.6B tokenizer. Each budget includes the task instructions, in-context demonstrations where provided by RULER, the Qwen3 non-thinking chat template, the official answer prefix, and the task-specific generation allowance. For every task and length, we generated a pool of 100 examples using independently seeded official generators. The 10- and 50-example settings are nested, uniformly sampled subsets of the corresponding 100-example pool. SQuAD-v2 answerable development questions and HotpotQA development distractor questions were sampled uniformly without replacement from their full eligible development sets; the selected question identities were held fixed across context lengths. We preserved the official background construction, needle placement, task complexity, and generation allowances. All random seeds, subset indices, source question IDs, generator and evaluation-framework commits, tokenizer revision, dependency versions, source-corpus snapshots, and measured prompt lengths were retained. We evaluated Qwen3-0.6B through vLLM and lm-evaluation-harness with non-thinking prompts and greedy decoding, using the official RULER answer-containment metrics and equally weighted task averages. The initial evaluation uses 10 examples per task at 4k–32k; longer datasets are retained for subsequent context-extension experiments.

## 官方来源

- [RULER 官方仓库](https://github.com/NVIDIA/RULER)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)
