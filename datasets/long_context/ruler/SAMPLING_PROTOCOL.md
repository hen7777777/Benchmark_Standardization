# RULER 冻结数据、抽样与论文报告协议

> 本文件与冻结样本同目录保存（`datasets/long_context/ruler/`），是论文方法描述的权威版本。
> 精确抽样索引见同目录 `sampling_subsets.json`，长度与位置统计见 `length_statistics.json`，
> QA 来源问题见 `qa_sampling.json`，生成种子见 `generation_commands.json` 与 `generation_protocol.json`。
> 生成侧原始记录仍保存在 `ruler/metadata/`，两侧内容由哈希校验保持一致。


本项目采用RULER v1官方13个任务配置，使用程序生成及冻结的背景语料构造样本；Qwen3只提供长度计数的tokenizer，**没有调用Qwen3模型生成数据**。这是本地缩小评测协议，不是官方默认每任务500条的设置，也不是RULER v2。

## 数据规模与路径

数据位于 `datasets/long_context/ruler/<10samples|50samples|100samples>/<4k|8k|16k|32k|64k|128k|256k>/<任务>.jsonl`。目录名与token数的对应：4k=4096、8k=8192、16k=16384、32k=32768、64k=65536、128k=131072、256k=262144。全部文件纳入Git；原始生成源码、语料/QA来源快照及参考tokenizer在 `ruler/sources/`，原始抽样记录在 `ruler/metadata/`。

| 数据版本 | 每任务每长度 | 每个长度的13任务总数 | 七个长度的记录数 |
|---|---:|---:|---:|
| `10samples` | 10 | 130 | 910 |
| `50samples` | 50 | 650 | 4,550 |
| `100samples` | 100 | 1,300 | 9,100 |

七个名义总长度预算为4096、8192、16384、32768、65536、131072、262144 tokens，K=1024。三版本共14,560条文件记录，但只有9,100个唯一“任务/长度/池索引”身份，不能视为三个独立数据集或独立重复实验。默认入口评`100samples`版的4k/8k/16k/32k：13×4×100=5,200条。64K以上数据完整保留，须明确所用模型上下文设置，不能默认为原生窗口成绩。

## 精确抽样规则

任务顺序及0起始序号t来自 `sources/task_configs.yaml`：

| t | 任务 | 主要配置 | 最大输出tokens |
|---:|---|---|---:|
| 0 | niah_single_1 | 重复噪声；词key/7位数字value；1 key、1 value、1 query | 128 |
| 1 | niah_single_2 | essay背景；词key/数字value；1/1/1 | 128 |
| 2 | niah_single_3 | essay背景；词key/UUID value；1/1/1 | 128 |
| 3 | niah_multikey_1 | essay背景；4 keys，每key 1 value，查询1个 | 128 |
| 4 | niah_multikey_2 | 干扰needle背景；词key/数字value；1目标key | 128 |
| 5 | niah_multikey_3 | 干扰needle背景；UUID key/value；1目标key | 128 |
| 6 | niah_multivalue | essay背景；1 key对应4 values | 128 |
| 7 | niah_multiquery | essay背景；4 keys、每key 1 value、4 queries | 128 |
| 8 | vt | 重复噪声；1条变量链、4 hops，保留官方内嵌示例 | 30 |
| 9 | cwe | 10个常见词各30次，其他词各3次，保留官方内嵌示例 | 120 |
| 10 | fwe | 6字母编码词；Zipf参数2.0；输出排除最高频噪声后的top3 | 50 |
| 11 | qa_1 | SQuAD v2可回答dev问题与干扰文档 | 32 |
| 12 | qa_2 | HotpotQA dev distractor问题与干扰文档 | 32 |

长度序号j按上述七档升序从0到6。主种子S=20261004：

- 每个任务/长度100条母集的生成种子：`S + 1000*j + t`。
- 子集选择种子：`S + 100000 + t = 20361004 + t`，与长度无关。
- QA源问题选择种子：`S + 200000 + t = 20461004 + t`，与长度无关。

先生成每个任务/长度的100条母集。用Python `random.Random(subset_seed).shuffle(list(range(100)))` 得到无放回随机排列；`10samples`取前10个索引，`50samples`取前50个，`100samples`保留原池0–99顺序。因此集合满足10⊂50⊂100，但`100samples`的排列不等于`50samples`排列直接扩展。

同一任务的各长度使用相同子集索引。没有按模型成绩、答案难度或needle位置筛样；没有按深度分层。`metadata/subsets.json`保存全部273个“版本×任务×长度”单元的seed、pool_indices和sample_ids。

QA分别从SQuAD的5,928个可回答dev问题和HotpotQA的7,405个dev distractor问题中均匀无放回抽100条，seed分别为20461015与20461016。SQuAD不可回答问题按官方规则排除，不按答案长度筛选。QA问题身份及顺序跨长度保持配对，每行有qa_source_id/index/selection_seed；不同长度不保证干扰文档或位置相同。其他合成任务采用不同的长度生成种子，不能称所有任务跨长度都是同一道题。

## 长度、生成与位置分布

生成时固定参考tokenizer为 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。保存的长度满足：完整Qwen3非thinking聊天prompt的token数＋该任务最大输出预算≤名义长度。任务说明、内嵌示例、问题、聊天模板及官方answer prefix均计入；12-token模板开销预留后，还重新计数实际渲染prompt。

生成器搜索/减少背景单位来满足预算，没有靠截断完整prompt凑长度。名义4K不代表每题恰为4096 tokens。每个单元的min/mean/max、平均填充率和首答案深度十分位计数见 `metadata/length_statistics.json`。

保留固定官方生成算法：essay needle位置从40个离散深度候选无放回抽取；重复噪声/干扰needle背景使用背景单位位置的无放回抽样。多needle位置并非独立；不能声称每题均匀覆盖所有token深度。深度直方图只记录首个参考答案，不是全部needle位置。FWE采用确定性Zipf频数，不是每次词出现独立抽样。

每个任务/长度单独Python进程生成，固定PYTHONHASHSEED=0、单线程BLAS和TOKENIZERS_PARALLELISM=false。官方Python/NumPy随机状态、背景长度搜索和超预算重试均属于复现条件；仅给seed不足以复现，必须保留代码、词表/语料、tokenizer和依赖。`metadata/requirements.lock.txt`记录旧生成环境；新评测环境另见run manifest。

## 统一入口推理和评分

冻结题目不因checkpoint变化而重新生成。新入口用**当前checkpoint**的聊天模板渲染同一input：单user turn，add_generation_prompt=True，enable_thinking=False，在assistant起始位置追加官方answer prefix；不在harness外层重复套模板。对原Qwen3-0.6B，已验证渲染与原冻结模板相同。

名义长度仍使用参考Qwen tokenizer定义；新checkpoint的实际token数、与源计数差值、有效窗口均逐题记录。超有效窗口即拒绝运行，不截断、不静默丢题；不同tokenizer并不自动拥有同样的实际长度。默认总窗口32768，BF16、greedy单候选、任务预算不变，模型与harness seed=20261004。

评分沿用官方大小写不敏感的字符串包含规则，先清理预测的控制字符。QA任意合法alias命中得100，否则0；其他任务按要求参考值的命中比例得0–100分，允许部分得分。每任务取样本均值，每长度对13任务等权macro平均，不跨长度或跨benchmark合成总分。这是containment评分，不是严格答案字符串完全相等。官方输出通常round到两位；本harness保留全精度，论文可显示两位但聚合时不提前取整。

`--smoke`使用所选版本每个任务/长度的前1条（默认100版池index0），默认共52条，仅验收流程。它不是10条随机版本的成绩。实际跑完10版须设置--ruler-samples 10且不加--smoke。

## 已验证范围及论文限制

- 导入核验覆盖全部273单元：14,560记录、嵌套与精确顺序、种子重放、QA源ID/答案/问题、保存长度统计、NIAH答案存在性和冻结文件哈希。
- 旧记录只有4K的13任务×100条母集重新生成后逐字节一致，不声称七档均已重新生成核验。
- 旧推理记录为10版的4K–32K，共520条；另有独立walkthrough4K130条。新入口验收另存，不能混成同一次实验，也不能将其填为100版成绩。
- 当前单一数据生成seed不能提供跨生成seed方差。greedy不保证跨引擎/硬件逐位一致；配对比较应固定样本ID与推理协议。

## 可用于论文的方法段落

我们基于固定版本RULER v1的全部13项官方任务配置，使用程序和冻结背景语料生成每任务、每名义长度100个样本。长度预算为4K、8K、16K、32K、64K、128K和256K，按冻结的Qwen3-0.6B tokenizer计数，包含完整提示、非thinking聊天模板、官方answer prefix及任务输出预算。我们对每个100样本池进行固定种子的无放回随机排列，取前10和前50构成嵌套子集；QA从完整可用dev问题池中无放回抽样，问题身份跨长度固定。正式默认设置使用100版的4K–32K，共5,200个任务/长度样本。评测通过lm-evaluation-harness和vLLM完成，使用当前checkpoint的非thinking聊天模板及greedy解码，重新审计实际token长度并禁止截断。采用官方答案包含评分，并分别报告各长度的13任务等权均值及分项得分。所有生成/选择种子、子集索引、QA来源ID、源码提交、参考tokenizer、依赖、语料快照和文件哈希均保留。该100样本协议是本地缩小设置，实际模型成绩须来自完成相应全部样本的独立运行。

We constructed a reduced RULER-v1 evaluation set using all 13 official task configurations, with 100 examples per task at nominal total sequence budgets of 4K, 8K, 16K, 32K, 64K, 128K, and 256K tokens (K=1,024). Budgets were measured with a pinned Qwen3-0.6B tokenizer and included the full prompt, non-thinking chat template, official answer prefix, and task-specific generation allowance. Nested 10- and 50-example subsets were selected by taking prefixes of a seeded uniform permutation of each 100-example pool. QA source questions were sampled without replacement from the eligible development sets and held fixed across lengths. Our default evaluation configuration uses 100 examples per task at 4K–32K, yielding 5,200 task–length examples. We use lm-evaluation-harness with vLLM, the evaluated checkpoint's non-thinking chat template, and greedy decoding. Actual token lengths are audited for each checkpoint, and overlength inputs are rejected without truncation. We report official answer-containment scores and an equally weighted mean across the 13 tasks separately at each length. Seeds, selected indices, QA source IDs, code revisions, tokenizer revision, dependency versions, corpus snapshots, and file hashes are retained. This is a locally reduced protocol rather than the official default 500-example setting; reported model scores must come from a completed evaluation of the specified samples.

固定RULER源码：`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`。[官方仓库](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)。固定harness：`d6de81643928d653435c431bae19945d41d32520`。
