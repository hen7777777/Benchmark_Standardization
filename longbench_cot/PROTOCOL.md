# LongBench CoT 协议说明

本目录为 LongBench v1 / v2 的 **chain-of-thought（CoT）评测变体**，作为独立阶段
（`longbench_v1_cot`、`longbench_v2_cot`）加入统一入口，**不覆盖**已有的直答阶段
（`longbench_v1` / `longbench_v2`，走 harness 内置任务）。任务 YAML 与运行时工具
由 `build_tasks.py` 从 harness 的直答任务定义一次性生成，`tasks/runtime_utils.py`
负责提示词渲染与评分。

## 数据与提示词来源

- v1：`Xnhyacinth/LongBench`（与 harness 内置 `longbench` 组同一镜像、同一 `test` 分片）。
  每个数据集的直答提示词原样复制 harness 的 `doc_to_text`，因此与直答阶段逐字一致，
  只是额外追加一句推理指令。该镜像的 `context` / `question` 字段本身已带脚手架，
  harness 直答任务同样如此，CoT 阶段保持一致以便直接对比。
- v2：`recursal/longbench-v2`（与 harness 内置 `longbench2` 组同一镜像、同一 `train` 分片，
  官方 `THUDM/LongBench-v2` 的社区镜像）。提示词按官方 `prompts/0shot_cot.txt` +
  `prompts/0shot_cot_ans.txt` 的内容构造。

## v1 CoT（`longbench_v1_cot`，19 个数据集）

LongBench v1 官方仓库 **没有** CoT 协议（`prompts/0shot_cot*.txt` 是 v2 的多选题模板）。
因此 v1 CoT 采用“先推理、后作答”的标准化变体：

1. 提示词 = 官方直答提示词 + 追加：
   `Let's think step by step, then write your final answer on a new line starting with "Final answer:".`
2. 一次生成（`max_gen_toks=1024`，greedy），模型先输出推理、再输出
   `Final answer: <答案>`。
3. 评分：取生成文本中**最后一个** `Final answer:`（不区分大小写）之后的内容作为最终答案，
   套用官方 per-dataset 指标（与 harness `metrics.py` 同源，见 `runtime_utils.py`）：
   - QA：`qa_f1_score` / `qa_f1_zh_score`
   - 摘要：`rouge_score` / `rouge_zh_score`
   - 分类（trec / lsht）：`classification_score`
   - 合成（passage_count / passage_retrieval_*）：`count_score` / `retrieval_score` / `retrieval_zh_score`
4. 组聚合指标 `score`（各叶子指标统一映射到 `score`，`weight_by_size=False`）。

排除 `lcc`、`repobench-p` 两个代码补全任务（补全不是“推理后作答”，CoT 不适用）。
提取不到 `Final answer:` 时回退为对整段生成打分（宁可在指标上退化，也不抛错），
这一回退规则已写入代码注释。

## v2 CoT（`longbench_v2_cot`，20 个子数据集）

官方 v2 CoT 是**两段式**（`pred.py --cot`）：

1. 第一段用 `0shot_cot.txt` 生成推理（`max_new_tokens=1024`）。
2. 第二段用 `0shot_cot_ans.txt`（文档省略、嵌入第一段推理）生成最终答案
   （`max_new_tokens=128`）。
3. 提取：正则 `The correct answer is \(([A-D])\)`，失败再试
   `The correct answer is ([A-D])`（先去掉 `*`）。
4. 评分：提取的字母 == 答案字母（`acc`）。

lm-evaluation-harness 的 `generate_until` 一次只生成一段，因此本实现把两段合并成
**一次生成**：提示词 = `0shot_cot.txt` 的文档/问题/选项 + `Let's think step by step`，
紧接着 `0shot_cot_ans.txt` 的收尾指令
（`... most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".`），
`max_gen_toks=1024`。提取正则与准确率评分与官方 `pred.py` **完全一致**，唯一差异是
“两段 API 调用”合并为“一次生成”。数据 `answer` 为 0-3 的整数索引，`runtime_utils.v2_process`
按 `{A:0,B:1,C:2,D:3}` 换算后比较。

## 与官方/直答阶段的差异边界

| 项 | 官方 v1 | 官方 v2 CoT | 本套件 v1 CoT | 本套件 v2 CoT |
|---|---|---|---|---|
| 提示词 | 直答 `dataset2prompt.json` | 两段 `0shot_cot*` | 直答 + 推理指令 | 两段合并为一段 |
| 生成 | greedy，各数据集不同 max_gen | 1024 + 128 两段 | 1024 一段 | 1024 一段 |
| 提取 | 整段打分 | 正则取字母 | 取 `Final answer:` 之后 | 正则取字母（同官方）|
| 评分 | per-dataset 指标 | acc | per-dataset 指标（同官方）| acc（同官方）|

温度差异：官方 v2 CoT 用 `temperature=0.1`（API 采样）；本套件与其余阶段一致用 greedy
（`do_sample=False`）以保证可复现。写论文时请注明这一处偏差。
