# RULER 与 LongBench 系列：固定版本评测细节及跨模型协议

本次核对的是本地 harness 提交 `d6de81643928d653435c431bae19945d41d32520`，覆盖 RULER、LongBench v1、LongBench-E、LongBench v2。逐任务配置、输出上限和数据来源已导出到 [源码清单](../audit/long_context/harness_inventory.json) 和 [任务参数表](../audit/long_context/task_limits.csv)。以后使用的是本文确定的非 thinking 协议，参数不根据模型得分调整。

**协议对齐不等于推理输出逐 token 确定。** 本次完成源码核对、外部任务配置与评分函数、离线行为验证；没有运行新的 LongBench 模型评测，也没有下载其完整数据内容。已有 Qwen3 RULER 实测结果保持原样。

## 1. 四套 benchmark 的执行方式

| Benchmark | 当前 harness 的请求类型 | 输出限制的含义 | 数据与评分 |
|---|---|---|---|
| 内置 `ruler` | `generate_until` | 逐任务限制新生成 tokens | 运行时按 tokenizer 造数据，默认 4k、500 条/任务/长度；本项目不使用这条数据链路 |
| 本项目 `ruler_local_*` | `generate_until` | 官方逐任务预算 | 已冻结的七档长度、13 任务、10/50/100 嵌套数据；QA 任意 alias 命中 |
| `longbench` / `longbench_e` | `generate_until` | 每个任务的 `max_gen_toks` | v1 为 21 个任务，E 有 13 个任务配置；QA/F1、ROUGE-L、分类、检索、计数、代码相似度等 |
| `longbench2` | `multiple_choice` → 四次 `loglikelihood` | 不自由生成答案，`max_gen_toks` 不适用 | 比较 A/B/C/D 标签续写的条件 log-likelihood，最高者为预测；报告 `acc` |

LongBench2 的选项正文出现在 prompt 中，续写候选仅为 A/B/C/D。候选字母编码未必永远只有一个 token，必须使用实际编码长度。相同条件概率分数时，harness 的 `numpy.argmax` 选择第一个候选，即候选顺序不能变。

本地 [LongBench2 公共配置](../vendor/lm-evaluation-harness/lm_eval/tasks/longbench2/_longbench_common_yaml) 使用 `recursal/longbench-v2`，split 名叫 `train`，但它是此 benchmark 的评测数据划分，并不意味着训练模型。实际共有 20 个子任务配置，包含 README 没列出的 `legal_multi`。总组按样本数加权；官方数据共 503 题，mirror 的各配置数量还需在实际加载时核对。

官方 LongBench v2 的公开脚本则是**生成答案并解析字母**，直接回答上限 128 tokens；CoT 分两次请求，先生成最多 1024 tokens 推理，再生成最多 128 tokens 答案，脚本温度为 0.1。因此，harness 原生 `longbench2` 的 label likelihood、官方 direct generation、官方 CoT 是三种不同评测，分数不能混为同一设置。[官方 v2 推理脚本](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py)

本次固定采用 **harness LongBench2 label-likelihood、non-thinking**，不用 `--gen_kwargs` 给它设置“128 tokens”，也不把它标为官方生成式 leaderboard 复现。

## 2. 输出 token 上限

`max_gen_toks` 限制新生成 tokens，不包括输入；EOS 可以使输出提前结束。不要给全 benchmark 统一覆盖成 128 或 512，否则会改变任务协议。

### RULER

| 任务 | 最大新生成 tokens |
|---|---:|
| 八个 NIAH 子任务 | 128 |
| VT | 30 |
| CWE | 120 |
| FWE | 50 |
| QA_1 / SQuAD、QA_2 / HotpotQA | 32 |

本项目数据的标称长度包含完整 Qwen3 非 thinking prompt 和上述输出预算。内置 RULER 生成器及 chat template 的长度预留行为不能直接当成这份已验证数据的行为。

### LongBench v1 / LongBench-E

E 的同名任务使用相同输出上限；表中 E 列表示有对应 `_e` 数据配置。

| 任务 | 最大新生成 tokens | E | 指标 | 输入模板策略 |
|---|---:|:---:|---|---|
| narrativeqa | 128 | — | 英文 QA F1 | chat |
| qasper | 128 | ✓ | 英文 QA F1 | chat |
| multifieldqa_en | 64 | ✓ | 英文 QA F1 | chat |
| multifieldqa_zh | 64 | — | 中文 QA F1 | chat |
| hotpotqa | 32 | ✓ | 英文 QA F1 | chat |
| 2wikimqa | 32 | ✓ | 英文 QA F1 | chat |
| musique | 32 | — | 英文 QA F1 | chat |
| dureader | 128 | — | 中文 ROUGE-L F1 | chat |
| gov_report | 512 | ✓ | ROUGE-L F1 | chat |
| qmsum | 512 | — | ROUGE-L F1 | chat |
| multi_news | 512 | ✓ | ROUGE-L F1 | chat |
| vcsum | 512 | — | 中文 ROUGE-L F1 | chat |
| trec | 64 | ✓ | 类别字符串匹配 | raw |
| triviaqa | 32 | ✓ | 英文 QA F1 | raw |
| samsum | 128 | ✓ | ROUGE-L F1 | raw |
| lsht | 64 | — | 类别字符串匹配 | raw |
| passage_count | 32 | ✓ | 提取数字后的正确比例 | chat |
| passage_retrieval_en | 32 | ✓ | 提取段落编号后的正确比例 | chat |
| passage_retrieval_zh | 32 | — | 提取段落编号后的正确比例 | chat |
| lcc | 64 | ✓ | 首个有效代码行的字符串相似度 | raw |
| repobench-p | 64 | ✓ | 首个有效代码行的字符串相似度 | raw |

输出上限与 [官方配置](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json) 一致。`lcc` 是代码补全任务，当前 harness README 中“分类”的文字描述不正确，应以配置和代码指标为准。

## 3. Chat template、BOS 与 thinking 的固定规则

框架允许 `--apply_chat_template`，默认不启用。该参数是整次运行的全局设置，不会自动根据 LongBench 的任务类型切换。任务 YAML 中的 `metadata.chat_policy` 只是记录，也不会自动执行模板开关。

本次确定的规则：

1. **当前 Qwen3 RULER 数据**：任务函数已经渲染了 Qwen3 非 thinking 模板并接上官方 answer prefix；CLI 不再传 `--apply_chat_template`，编码不追加 BOS。继续复用已实测流程。
2. **其他模型的 RULER**：复用冻结 JSONL 的原始 `input`、答案与 sample ID，用目标模型自己的固定模板渲染一次，保留官方 assistant answer prefix。本项目现有 `ruler_local_utils.py` 固定使用 Qwen tokenizer，不能直接把其已渲染 Qwen prompt 当作其他模型的通用模板。跨架构 renderer 尚未实测。
3. **LongBench v1/E**：TREC、TriviaQA、SAMSum、LSHT、LCC、RepoBench-P 使用 raw prompt；其他任务用一个 user turn 和目标模型原生非 thinking 模板。这是沿用官方任务级模板例外；不根据某模型分数临时选择 raw/chat。[官方 v1 推理脚本](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py)
4. **LongBench2 label likelihood**：固定使用原生非 thinking 模板，再评分候选标签续写；不开启 CoT、RAG 或 no-context 变体。
5. 无额外自定义 system instruction、外层 few-shot 为 0、每样本一次响应。VT/CWE 与 LongBench few-shot 任务原本内嵌的示例仍保留；不能因此统称全部任务为 zero-shot。

模板只渲染一次；原生模板包含的默认 system 文本也要保存。chat 渲染后的 token 化不再人为添加 BOS。raw 任务使用目标 tokenizer 固定版本的标准特殊 token 行为，并记录实际编码；Qwen3 此处不添加 BOS。跨 tokenizer 不要求人为插入相同 BOS ID。

对于 Qwen3，原生 `hf` / `vllm` 后端可以使用 `enable_thinking=False`；要显式写出，不能依赖模型默认值。固定版本 API 后端的 `apply_chat_template()` **没有传入这个参数**，因此仅传 CLI 的 `--apply_chat_template` 或在 API model_args 写 `enable_thinking=False` 不足以证明 thinking 被关闭。API 方式应先用目标 tokenizer 显式渲染非 thinking 文本，再使用 `/v1/completions` 发 token IDs；现有 Qwen RULER 已这样做。

没有可关闭 thinking 模式的 reasoning-only 模型不能直接归入此非 thinking 对照组。需作为独立 reasoning 设置报告，并另行定义包含推理的总输出预算，不能只删除 `<think>` 后仍称与短输出 baseline 相同。

## 4. 解码参数：不能只看 `do_sample=False`

固定使用 `temperature=0.0`、`do_sample=false`、`top_p=1.0`、`repetition_penalty=1.0`、seed `20261004`，每样本一次生成。greedy 时 top-k/top-p 采样筛选不参与随机抽样；不沿用各 checkpoint 的采样预设。HF 使用单 beam；vLLM 使用中性默认惩罚及任务参数，服务或原生引擎指定 `generation_config=vllm`。

**实测的 API 差异：** v1/E YAML 是 `temperature=1, do_sample=False`。当前 `hf` 和原生 `vllm` 的 `normalize_gen_kwargs()` 会把温度改成 0；`local-completions` 的 `_create_payload()` 则删除 `do_sample`，仍将 `temperature=1` 发到服务。因此，用内置 LongBench YAML 换成 API 后端，不保证仍为 greedy。

离线调用真实 payload 构造函数复现了上述差异，没有发送 HTTP 请求。证据见 [offline_verification.json](../audit/long_context/offline_verification.json)。导出的 34 个 v1/E 任务 YAML 已显式改为温度 0。CLI 也可用 `--gen_kwargs temperature=0,do_sample=False` 修正原任务温度，但这不会修复 E 任务覆盖或评分差异；不要同时全局覆盖任务 `max_gen_toks`、`until` 等参数。

参数覆盖顺序：任务 YAML 的 generation_kwargs → CLI `--gen_kwargs` 更新已有字段 → 模型后端转换 → 引擎/服务执行。`max_gen_toks` 没有写在任务时才可能落到后端默认值；本协议逐任务写清楚，不依赖默认 256。HF checkpoint 还可能自带 `num_beams`、惩罚等 generation_config，因此跨模型实验需记录并中和这些实际生效的配置；尚未部署新的跨模型执行器，不能声称 YAML 自动中和了所有 HF 模型预设。

## 5. EOS、换行停止与答案后处理

`until: []` 表示没有额外文本停止串，不表示禁止 EOS。后端仍会使用模型 EOS，并受 token 预算限制。不要全局加入 `\n`、`\n\n` 或 `<think>` 停止串。

原内置 LongBench 的 TREC、TriviaQA、SAMSum、LSHT 及其 E 版本使用真实换行字符作为停止串；已经确认其 codepoint 是 10，不是两个字符的反斜杠+n。其他任务为 `until: []`。

官方 v1 评分会对上述四类任务执行 `prediction.lstrip('\n').split('\n')[0]`。本地 harness 当前 `metrics.py` 仅 `strip()`，没有这一步。输出有多行时，两者可能不同，不能引用 `utils.scorer_e()` 就认为它自动参与了 YAML 评分；YAML 实际调用的是 `metrics.get_*_with_score`。

导出的对齐版本统一使用 **EOS/预算停止，`until: []`**，再对这四类任务使用官方首行预处理：避免模型先生成换行就立即被停止成空答案。官方 SAMSum 推理有额外换行 EOS 和最低长度规则，本对齐版本没有照搬其模型相关实现，而是用固定 128-token 上限加首行评分。因此此协议命名为 `long_context_nonthinking_v1`，不能声称与官方所有运行细节完全相同。

QA/F1、ROUGE 等按所有参考答案的最高分计分；不把 aliases 当成必须同时输出的答案。代码任务按固定 harness 实现跳过空白、含反引号/`#`/`//` 的行，取第一个有效行。对齐 helper 明确选择 `difflib.SequenceMatcher`，避免某个模型环境额外安装 python-Levenshtein 后改变 fuzzywuzzy 的底层相似度算法。保留原始输出，只有评分阶段执行规定的后处理；不额外剥离 thinking 内容来补救预算耗尽。

## 6. 长度与截断：先定义实验对象

当前 Qwen3 原生评测总预算为 32768；RULER 则按现有各长度的任务预算运行。模型上下文能力不足的组合不能强行运行或 silently crop。

**主协议使用完整输入，不截断。** LongBench 的输入原本就有长短分布，不把它当成自行生成的七档固定 token 数据。开始跨模型比较前，按完整渲染后的实际 token 数，在全部待比较模型上确定同一组可容纳的 sample IDs；先冻结共同集合，再运行推理。各模型不得独立过滤不同样本，也不能按答案或得分筛选。报告这组样本的每任务数量、原始覆盖率及实际长度，它属于共同可评测子集，不称完整 benchmark。

如果全部样本都能容纳，则运行完整数据。新增上下文更短的模型后，如果原共同集合不再适用，必须建立新实验并对所有对照模型重新运行该集合，不能直接复用不同样本上的旧总分。

RULER 复用同一冻结输入及身份，禁止按模型重抽样或截断掉 needle。当前 4k–256k 标签来自 Qwen3 参考 tokenizer；换 tokenizer 后记录目标模型实际长度，不能同时声称原标签就是每个模型的精确 token 预算。整格不支持时记为未评测，不临时删掉超长样本后计算“同一格”的分数。

若后续要评测**完整 LongBench 数据在固定窗口下的截断性能**，单独建立实验。确定的截断规则是：令可用输入预算等于总预算减任务输出预算；在完整渲染后的 token IDs 中保留前半段与后半段，奇数时后半段多保留一个 token。它与主协议的完整上下文性能分别报告。官方 v1 也使用中间截断，但其脚本先截 raw prompt 再加模型模板；本规则把最终模板及输出预算也计入窗口，不能称其逐字复现。

当前后端实际行为：

| 后端/请求 | 超长处理 | 对齐注意事项 |
|---|---|---|
| API `generate_until`、tokenized_requests=True | 无条件左截断到 `max_length-max_gen_toks`，即使 `truncate=False` | 必须在提交请求前确认完整输入合规；harness 样本日志的原 prompt 不能证明 HTTP 输入未截断 |
| 原生 `hf` 生成 | 对编码后的输入执行左侧长度限制 | 不把 `truncation=False` 当作完整 prompt 的证明 |
| 原生 `vllm` 生成 | `maybe_truncate`，默认左截断；可显式 `truncation_side=middle` | 仅在独立的截断实验使用该设置；完整输入主协议提前计数 |
| 原生 `vllm` loglikelihood | 拼接 context/候选后左截断到 `max_length-1` | `truncation_side=middle` 不控制此路径，LongBench2 不能照搬生成式设置 |

API 构造函数还会将传入的 `max_length` 减 1。本固定版本若有效总预算为 32768，API 要传 32769；原生 HF/vLLM 传 32768。该补偿不是输出多生成一个 token。其他 harness 版本须以其源码行为为准，不盲目沿用。

LongBench2 原生 likelihood 要让完整 prompt + 最长候选续写 + 后端预留 1 token 落在预算内。使用 chat template 时候选前的 delimiter 也与 raw 模式不同；不能在同一结果表内交替使用这两种模式。

## 7. LongBench-E 的覆盖、分桶与总分

当前内置 `longbench_e` group 只展开 **9** 个叶子任务，遗漏 `gov_report_e`、`multi_news_e`、`passage_count_e`、`passage_retrieval_en_e`。这些 YAML 已存在，但没有进入根 group。导出的 `longbench_aligned_e` 包含全部 **13** 个任务。

E 的 `_e` 表示采用 LongBench-E 数据，不是 v2，也不是“题目难度加大”。当前内置 YAML 只报告整体任务均分，没有自动调用 `utils.scorer_e()` 生成三档结果。对齐任务增加每任务 `score_0_4k`、`score_4_8k`、`score_8k_plus`。

分桶严格使用数据里的原始 `length` 字段，边界为 `<4000`、`4000≤length<8000`、`≥8000`；它不是 4096/8192，更不是目标 tokenizer 重新测出的 token 数。截断后仍保留原始桶标签，并另存实际输入长度。[官方 E 评分逻辑](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py)

v1/E 对任务均分等权汇总，不能把样本数不同的任务按行数加权；harness 的 group 实现收集叶子任务，因此不应改成先算类别均分再把六类等权平均。LongBench2 按样本数加权的 `acc` 则要保留。内置 LongBench 的 `score` 是 0–1，展示时乘 100；现有本地 RULER 为 0–100，不能把两个原始数字直接比较。E 空桶记为 NaN/无结果，不填 0，不将其当作缺失模型能力。

## 8. 已导出的可复用配置

任务与协议文件：

- [protocol.json](../audit/long_context/protocol.json)：确定的参数、模板、长度、评分规则及实测状态。
- [longbench_tasks/](../audit/long_context/longbench_tasks/)：34 个 v1/E 任务 YAML、六个 group 和固定评分函数。
- [longbench2_tasks/](../audit/long_context/longbench2_tasks/)：20 个 label-likelihood 任务及样本加权总组。
- [官方源代码快照](../audit/long_context/sources/longbench_official/SOURCE.json)：提交 `2e00731f8d0bff23dc4325161044d0ed8af94c1e` 下的推理、评分、模板和输出上限文件。

v1/E 数据来源固定为 `Xnhyacinth/LongBench` revision `2e9ade51ebf45d98942056c0716234f9d5d257d5`，v2 为 `recursal/longbench-v2` revision `380536fcbe7aac4a1d3e3b6502c2abd612399231`。**固定 revision 已完成，完整数据内容及共同样本集合尚未冻结**；不能把“有 YAML”描述为完成模型评测或完成数据验证。

| group | 任务数 | CLI 模板设置 |
|---|---:|---|
| `longbench_aligned_chat` | 15 | `--apply_chat_template`，显式 non-thinking |
| `longbench_aligned_raw` | 6 | 不传 `--apply_chat_template` |
| `longbench_aligned_e_chat` | 8 | `--apply_chat_template`，显式 non-thinking |
| `longbench_aligned_e_raw` | 5 | 不传 `--apply_chat_template` |
| `longbench2_aligned` | 20 | `--apply_chat_template`，label likelihood、non-thinking |

`longbench_aligned` / `longbench_aligned_e` 是全任务汇总定义。由于 CLI 模板开关是全局的，正式运行应拆 chat/raw 两组，最后按全部 21/13 个叶子任务等权合并；不能把两组均分简单平均。全组只在已逐任务预渲染输入的执行流程中使用。

这些 YAML 的 metadata 不实现自动模板分组或长度拒绝。首次运行前还需加载固定 revision 数据、完成完整 prompt 长度与共同集合冻结，再调用推理后端；不要未经长度处理就把全部长文档交给默认左截断路径。当前没有把这一未实测执行阶段伪装成可直接运行的完整命令。

本次为离线评分安装了 jieba 0.42.1、fuzzywuzzy 0.18.0、rouge 1.0.1；harness、Transformers、datasets 及这些评分依赖的实际版本保存在离线验证记录中。已有 RULER 运行的历史 dependency lock 未改写。

重新导出配置与运行离线验证：

```bash
cd /data/lpg_projects/mj/test
.venv/bin/python scripts/audit_long_context.py
.venv/bin/python scripts/verify_long_context_protocol.py
```

验证检测的是：API 温度与原生处理是否一致、输出预算是否与官方快照一致、21/13 任务是否完整、chat/raw 分组是否正确、函数引用能否加载、首行评分及 E 桶边界是否正确。它不运行 checkpoint，不产生新的模型分数。

## 9. 推理确定性与实验记录

当前在线 vLLM 0.17.1 的 RULER 重跑已观察到相同数据/参数下输出差异。固定 seed、greedy 和非 thinking 可以消除协议选择及采样随机性，但不能保证动态调度和数值计算逐 token 一致。[vLLM 0.17.1 官方复现说明](https://docs.vllm.ai/en/v0.17.1/usage/reproducibility/)

同一模型比较表固定模型后端及精确依赖版本、dtype=bfloat16、无量化、GPU 类型/卡数、TP/DP、batch/concurrency、KV cache dtype 和调度配置。改变后端、量化或推理方式应形成独立实验，不能只改模型路径却沿用混合配置的旧结果。

原生 vLLM extra 在此 harness 提交要求 `vllm>=0.18`，已有 API 服务使用 0.17.1。后续原生评测须准备单独环境，固定并记录实际选定版本；本次没有升级原服务或宣称这两个版本相同。若要求同硬件/版本下逐 token 确定，应按选定 vLLM 版本验证离线确定性调度或 batch invariance，再做受控重复验证。本次尚未启用或实测它们，不能写“已保证完全确定性”。

每次实验保留 checkpoint 标识、数据 revision/样本 ID、协议 ID、逐任务输出预算、模板全文与参数、完整 prompt token 数、实际后端输入长度、原始输出、候选概率（v2）、逐样本分数、实际命令和环境版本。discard 的结果按本项目约定整体移动到对应 run 的 `tmp/`；已有结果不因新协议发布而回填或覆盖。
