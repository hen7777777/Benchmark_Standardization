## L40 实操：小模型标准评测流程与 lm-eval-harness 使用记录（2026-10-05）

目标是为 0.5B–10B 的训练 checkpoint 建立固定、可复用的评测流程。本次以 /data/models/Qwen/Qwen3-0.6B 验证文中十项 benchmark 的数据加载、提示构造、推理、答案提取和评分，不建立新的模型榜单。Qwen3-0.6B 模型卡写明 Pretraining & Post-training；它在这里是管线验证模型。正式训练曲线换入自己的完整 checkpoint 即可，继续使用冻结的数据、模板和评分配置。

验证状态：原生 vLLM 后端已跑通全部十项；HumanEval 与 HumanEval+ 共享同批候选，因此十项对应九个运行入口。最终 smoke_final 每个叶子任务取 1 条，共 104 道不同样本；包括 BBH 全部 27 个配置、MMLU 全部 57 个学科、MMLU-Pro 全部 14 个领域。另以 GSM8K 验证了 HF 后端和直接执行 harness CLI。这些是流程验证，不是完整 benchmark 分数；全量 HF 后端和所有可能的小模型架构没有逐项实测。

### 1. 文件位置、版本与回滚

工作目录：/data/projects/mj/test。原有 RULER/LongBench 文件保留；本次数据、任务与结果放在 runs/standard_benchmarks_20261005/，使用独立环境 .venv-standard。harness vendor 源码未修改；std_* 是通过 --include_path 加载的外部任务。

修改前回滚提交：1c6c89e4b843df5cb230fb640accfd5b347e96c5；原文件备份：audit/standard_eval_before_20261005T124531Z/existing_code.tgz。它保存 README、scripts、docs 和原有长上下文审计配置，便于比较；不要用破坏性的 git reset 或删除旧 runs 来“恢复环境”。

固定 harness 提交 d6de81643928d653435c431bae19945d41d32520，lm_eval 0.4.14.dev0；Python 3.13.5，PyTorch 2.10.0，Transformers 4.57.1，vLLM 0.17.1，datasets 5.0.1，EvalPlus 0.3.1，math-verify 0.9.0，antlr4-python3-runtime 4.11.1。完整可见包版本见 environment.lock.txt，Python 路径和环境继承关系见 environment_paths.json。Minerva task 为 3.1，GSM8K CoT 为 3.0；代码 scorer 的冻结版本见 data_manifest.json。

主要脚本：scripts/bootstrap_standard_env.sh 幂等配置并校验新环境；scripts/prepare_standard.py 冻结数据并导出任务；scripts/run_standard.py 运行全部或指定任务，并在请求前检查长度；scripts/standard_code_utils.py 实现官方代码评分适配；scripts/verify_standard.py 验证评分器；scripts/summarize_standard.py 汇总唯一题数、提取失败和输出预算诊断。

### 2. 环境与代理：先采用已有 set_home.sh

```bash
source /data/projects/mj/set_home.sh
cd /data/lpg_projects/mj/test
.venv-standard/bin/python -c 'import lm_eval, torch, transformers, vllm, datasets, evalplus; print("imports OK")'
```

实测直连 huggingface.co 和 hf-mirror.com 都在 12 秒后超时。source /data/lpg_projects/mj/set_home.sh 后，Hugging Face 数据 API 返回 200；该环境代理使用 127.0.0.1:17891。直接 source 会执行脚本自身的联网检查并输出较多 curl 信息，这是脚本行为。已有 17891 端口不能重复绑定；本次曾建立的独立 19873 SSH 代理通道已关闭，后续使用已有脚本。

旧 .venv 是面向 API 客户端的数据/评测环境，缺少 torch、vLLM，也没有 pip 模块。直接使用 .venv/bin/python -m pip 会报 No module named pip。新 .venv-standard 继承 /home/lpg/anaconda3 的 torch/vLLM，并通过 .pth 引用旧 .venv 的 harness、datasets 和评分类库；额外评分依赖只安装到新环境，没有升级旧环境或共享 conda。调用新环境的 Python 最明确，避免 PATH 中的 lm_eval 脚本仍指向旧客户端环境。

第一次无约束安装 harness[hf] 时，uv 尝试下载更新的 torch/CUDA 13 依赖。本次已停止该安装，改为继承现有 GPU 运行库，并固定额外依赖。不要在共享机器上通过一次无约束 pip/uv 安装隐式替换 torch、CUDA、Transformers 和 vLLM。当前环境的 .pth 继承关系也是复现条件；environment.lock.txt 是记录，不应在新机器上不加判断地整体升级所有共享包。

环境重建/校验入口是 bash scripts/bootstrap_standard_env.sh。它只管理 .venv-standard，要求已有 /home/lpg/anaconda3 的 GPU 依赖和旧客户端 .venv，校验核心版本后使用 uv --no-deps 安装十二项固定评分依赖。Minerva utils 显式要求 antlr4-python3-runtime 4.11；直接安装其它默认 extras 可能得到 4.13 并触发导入断言，因此这里固定 4.11.1，已验证数学评分函数可运行。

```bash
source /data/projects/mj/set_home.sh
cd /data/projects/mj/test
bash scripts/bootstrap_standard_env.sh
```

### 3. 统一协议：固定 Base 续写形式

协议命名 small_base_raw_v1。所有任务使用纯文本续写，不传 --apply_chat_template，不自动包装聊天角色，不添加自定义 system instruction。few-shot CoT 属于提示示例，不是额外训练。本次 Qwen tokenizer 显式 add_bos_token=False；换模型时应核对其 tokenizer/BOS 规范并固定记录，不能把 Qwen 的特殊 token 设置机械套给所有架构。

生成任务固定 temperature=0、do_sample=False、单次生成，逐任务固定 max_gen_toks 和停止串。vLLM 使用 generation_config=vllm，中和模型卡自带的随机采样配置。HF 的 greedy 模式可能提示 temperature/top_p/top_k 被忽略，这不等于评测失败；需要根据实际 do_sample 与 beam 设置判断。若换入自带 beam/惩罚设置的 checkpoint，应同时核对 generation_config，避免悄悄改变协议。

本流程没有渲染 Qwen thinking/chat 模板，因此不应称它为 Qwen 官方 non-thinking chat 成绩。enable_thinking=False 的作用依赖 chat template；在 raw Base 续写中写这个参数不能证明模型绝不会输出思考内容。生成中的思考、重复或格式错误保留并按固定评分器处理，不事后删掉后补分。

本次上下文总预算 32768，Qwen 配置的 max_position_embeddings 为 40960、rope_scaling=null，没有扩展上下文。vLLM 只传 max_model_len；该版本禁止同时传 max_length 和 max_model_len，会报 Either max_length or max_model_len may be provided, but not both。运行脚本对生成检查 input_tokens + max_gen_toks，对 likelihood 检查编码后的 context+continuation 并预留 1 token；超限报 INPUT_TOO_LONG，拒绝静默截断。直接裸跑 CLI 不附带此保护，所以正式批量流程优先用 run_standard.py。

### 4. 十项 benchmark 的入口、数据和评分

GSM8K：gsm8k → std_gsm8k_cot。openai/gsm8k main/test，1319 题；固定 8-shot CoT；输出上限 1024；停止 Q:、</s>、<|im_end|>；主指标 exact_match,strict-match，flexible-extract 作为诊断。

MATH500：math500 → std_minerva_math500。HuggingFaceH4/MATH-500 test，固定 500 题；4-shot Minerva CoT；输出上限 2048，停止 Problem:；保留 exact_match 与 math_verify 两条路径，主协议沿用 Minerva exact_match，不根据本次得分选择更高的评分器。它不是完整 MATH。

HumanEval：humaneval → std_humaneval_pair 的 pass_at_1。EvalPlus HumanEval v0.1.10，164 题；0-shot 函数体续写；输出上限 1024；与 HumanEval+ 共用原始候选，基础测试由官方 EvalPlus 的 base 路径执行。结果应标明 HumanEval / EvalPlus-base，不声称完全复刻 OpenAI 原始运行环境。

HumanEval+：同一 std_humaneval_pair 的 pass_at_1_plus。相同 164 题、相同候选；官方 EvalPlus plus_input、特殊比较和时间限制参与评分；通过标准为 base 与 plus 都通过。不是重复推理后取两个不同候选的最好分。

MBPP：mbpp → std_mbpp_original。google-research-datasets/mbpp full/test，原始 500 题；固定 task_id 2/3/4 三个示例，全部三条可见测试；[BEGIN] 后补全、[DONE] 停止；输出上限 1024；冻结的 Hugging Face code_eval 执行原始测试，报告 greedy pass_at_1。

MBPP+：mbppplus → std_mbppplus_pair 的 pass_at_1_plus。官方 EvalPlus MBPP v0.2.0 的 378 题；固定三条不与评测集重叠的原始 train 示例 601/609/613；示例与目标题使用纯文本 docstring 代码续写形式，显示一条接口 assert；输出上限 1024、停止 [DONE]。同批候选另报 378 题集合上的 EvalPlus-base pass_at_1，不能把它改名为原始 MBPP 500 题成绩。

BBH：bbh → std_bbh_cot_fewshot。SaylorTwift/bbh 固定 revision；全部 27 个 harness 配置，共 6511 题；3-shot CoT；输出上限 1024，保留该版本的 </s>、Q、双换行停止串及 get-answer 提取器；按 harness group 的 weight_by_size=True 汇总。27 个配置包含同一推理题族的多个规模版本。

DROP：drop → std_drop。EleutherAI/drop validation，9536 题；固定 first_n 的前三条 train 示例；显式 Passage/Question/Answer 模板；输出上限 128，真实换行停止；官方 numeracy-sensitive EM/F1 的 harness 实现。把默认 YAML 的目标改为预处理后的 answers[0]，避免引用不存在的 answer 字段；点号停止改为换行，避免小数或缩写被截断。这是明确的自定义 3-shot 协议。

MMLU：mmlu → std_mmlu。cais/mmlu，57 学科、test 共 14042 题；各学科 dev 的固定 5-shot；multiple_choice → 四个候选标签的 loglikelihood，按样本数加权 acc。这里没有自由生成输出预算；不能给它加 CoT 生成参数再称为相同指标。

MMLU-Pro：mmlu_pro → std_mmlu_pro。TIGER-Lab/MMLU-Pro 的冻结修正版本，14 领域、test 共 12032 题；领域内 validation 固定 5-shot CoT；输出上限 2048、停止 Question:；custom-extract 提取最终字母，失败记错，不随机猜测、不按正确答案重试；全量 group 按样本数加权。

DROP 的多 span 答案还有一个接口边界：当前继承的 harness process_results 将单次生成字符串作为一个预测 span 与参考 span 列表比较；它保留官方数字敏感匹配，但不自动把任意逗号拆成多个实体。正式研究若改变多 span 输出/拆分规则，需建立新协议，保留原始输出，不能把新规则的分数直接接到本协议曲线上。

### 5. 最容易漏掉的 Plus 问题：测试强度与示例泄漏

固定版本内置 humaneval_plus 继承 humaneval 的普通 code_eval；mbpp_plus 也继承通用代码评分。task 名称含 plus 不能单独证明执行了 EvalPlus 的特殊 oracle、官方数值容差和依输入耗时定义的 timeout。这里没有宣称内置 scorer 在所有题上都错误，而是显式选择官方 EvalPlus scorer，确保实验所称“Plus”与实际测试链一致。

本流程把 EvalPlus 接到 lm-eval 的 process_results：使用官方 get_groundtruth 和 check_correctness，执行 base_input / plus_input、atol、特殊 oracle；min_time_limit=1.0 秒、gt_time_limit_factor=4.0，fast_check=True，Plus 分数要求 base 与 plus 同时 PASS。原始 MBPP 用冻结 code_eval，单候选 timeout=3 秒、num_workers=1。生成和评分都留在同一个 harness 任务执行链中，不把 Plus 降级为另一套提示上的普通单元测试。

实测发现：内置 mbpp_plus 继承的 3-shot 示例为 2/3/4，而当前 Plus 378 题中也有 Mbpp/2、Mbpp/3、Mbpp/4。最初试跑后已修正为固定 train 示例 Mbpp/601、Mbpp/609、Mbpp/613，并验证其与 378 个评测 task_id 的交集为空。题集仍为 378，没有悄悄删除前三题变为 375；旧配置快照保存在 protocol_history/pre_mbppplus_overlap_fix/，旧结果保留，最终应使用 smoke_final。

EvalPlus MBPP 读取器把序列化对象恢复为 tuple、set、complex 等 Python 对象，直接 json.dumps(get_mbpp_plus()) 实测报 Object of type complex is not JSON serializable。正确做法是保存官方原始序列化 JSONL，并用官方 loader 解码；提供给 harness 推理的数据只需要 task_id、prompt、entry_point 等字段，不擅自把特殊测试输入转成字符串。

代码测试需要显式允许执行：流程脚本用 --allow-code-execution，并设置 HF_ALLOW_CODE_EVAL=1；裸 CLI 还需 --confirm_run_unsafe_code。评测器的子进程/超时/可靠性限制不等于操作系统安全隔离，不应在报告中把它写成已部署容器 sandbox。这里的 syntax_valid 只是语法诊断，语法正确的程序仍可能功能错误。

### 6. 可以直接复制的运行命令

全流程 smoke：每个叶子任务取 1 题，所有入口使用同一个加载好的 vLLM 模型。--experiment 必须使用新名称，脚本遇到已有结果目录会报错，避免覆盖或混入旧 checkpoint 的输出。

```bash
source /data/lpg_projects/mj/set_home.sh
cd /data/lpg_projects/mj/test
.venv-standard/bin/python scripts/run_standard.py \
  --checkpoint /data/lpg_models/Qwen/Qwen3-0.6B \
  --backend vllm --gpu 0 --max-length 32768 \
  --gpu-memory-utilization 0.15 \
  --tasks all --limit 1 --bootstrap-iters 0 \
  --allow-code-execution --experiment qwen3_pipeline_smoke_new
```

正式全量：去掉 --limit；其他提示、shots、预算、种子和评分规则保持不变。下例是正式运行入口，本次没有执行这条全量命令。替换 checkpoint 需要完整的 HF 权重、config 和 tokenizer；只含 LoRA adapter 的目录不能直接当作完整 checkpoint。

```bash
cd /data/lpg_projects/mj/test
.venv-standard/bin/python scripts/run_standard.py \
  --checkpoint /path/to/your/full_checkpoint \
  --backend vllm --gpu 0 --max-length 32768 \
  --gpu-memory-utilization 0.15 \
  --tasks all --allow-code-execution \
  --experiment checkpoint_step_100000_full
```

只跑数学或知识任务可用 --tasks gsm8k,math500 或 --tasks mmlu,mmlu_pro。run_standard.py 的 key humaneval 已含 HumanEval 与 HumanEval+；mbppplus 含 Plus 378 题的基础/增强配对分数，mbpp 才是原始 500 题。GPU 号是 CUDA_VISIBLE_DEVICES 的物理卡号；程序内部使用 cuda:0。当前共享 GPU 有其他任务，0.15 只是本次 vLLM 内存占比，不是跨模型通用值；换较大 checkpoint 时须根据模型权重与 KV cache 调整。

HF 后端已用 GSM8K 验证。适合先检查自己的 HF checkpoint 是否能加载；本次没有逐项全量验证该后端。

```bash
cd /data/lpg_projects/mj/test
.venv-standard/bin/python scripts/run_standard.py \
  --checkpoint /data/lpg_models/Qwen/Qwen3-0.6B \
  --backend hf --gpu 0 --tasks gsm8k --limit 1 \
  --bootstrap-iters 0 --experiment hf_load_smoke_new
```

直接调用 harness CLI 的等价 GSM8K smoke 也已实测成功，说明任务能够在 harness 注册与执行，不依赖另写的推理程序。此版本显式使用 run 子命令；不传全局 --num_fewshot，保留各任务自带的 8/4/0/3/5-shot 设置，也不全局覆盖 max_gen_toks。

```bash
cd /data/lpg_projects/mj/test
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv-standard/bin/python -m lm_eval run \
  --model hf \
  --model_args pretrained=/data/lpg_models/Qwen/Qwen3-0.6B,dtype=bfloat16,max_length=32768,add_bos_token=False \
  --include_path runs/standard_benchmarks_20261005/tasks \
  --tasks std_gsm8k_cot --device cuda:0 --batch_size 1 \
  --limit 1 --seed 20261005 --log_samples \
  --output_path runs/standard_benchmarks_20261005/results/cli_smoke_new
```

若把 CLI --tasks 换为 std_bbh_cot_fewshot、std_mmlu、std_mmlu_pro，会展开完整 group；代码项使用 std_humaneval_pair、std_mbpp_original、std_mbppplus_pair，并显式允许代码执行。其它入口为 std_minerva_math500、std_drop。正式运行建议保留 run_standard.py 的长度防护和结果目录防覆盖检查。

### 7. 数据冻结、离线运行与恢复

data/ 保存全部原始 split 的 JSONL；MMLU 从 all 按 subject 拆分，保留原顺序。sources/ 保存官方 EvalPlus 序列化数据与冻结 code_eval；tasks/ 保留外部 YAML、few-shot、函数引用与 group 聚合。data_manifest.json 记录来源和规模，revisions.json 记录 HF SHA，sha256_manifest.json 记录逐文件 SHA256；每次运行在模型请求前核对哈希，改变冻结文件会失败。正式运行在脚本中设置 HF_HUB_OFFLINE=1 和 HF_DATASETS_OFFLINE=1。

openai/gsm8k：740312add88f781978c0658806c59bc2815b9866。

HuggingFaceH4/MATH-500：6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be。

EleutherAI/drop：619386fc7ffe6dace3cc6d3483c69b164ebf6bf3。

cais/mmlu：c30699e8356da336a370243923dbaf21066bb9fe。

TIGER-Lab/MMLU-Pro：b189ec765aa7ed75c8acfea42df31fdae71f97be。

google-research-datasets/mbpp：4bb6404fdc6cacfda99d4ac4205087b89d32030c。

SaylorTwift/bbh：b5306be6f827cfafbb545ff5a51f96916029b0fd。

HumanEvalPlus release v0.1.10 的 SHA256 为 42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f；MbppPlus v0.2.0 为 b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad；code_eval space 提交 4242b7b28325ff9d4d5e60522020d8cebc89d672。重新评测优先复用这些文件；如要更新数据，应建立新的协议目录，而不是在原训练曲线中替换 main。

prepare_standard.py 可以从已固定的 revisions.json 重建缺失的原始数据并重新导出配置，需要网络与 set_home.sh 的代理。它不是每次评测都必须执行的前置步骤；正式 checkpoint 对比直接使用已有冻结目录。更换示例、数据版本、答案提取或停止串后，需要重新建立结果系列，并保留修改前配置。

### 8. 本次实际结果：只用于检查流程

最终结果目录 runs/standard_benchmarks_20261005/results/smoke_final/。共 104 道唯一题、275 个请求：47 个生成请求和 MMLU 的 228 个选项 likelihood 请求。最长输入 2688 tokens，输入加预算最大 4032，小于 32768；0 个超限。GSM8K 的两个 filter 各写一条日志，所以它有 2 条 filter 记录但只有 1 道题；不能把日志行数误当模型独立采样数。

各入口的 smoke 指标（均为 0–1，不代表全量）：GSM8K 1/1，strict 与 flexible 都为 1.0；MATH500 1 题，Minerva exact_match=0.0、math_verify=1.0；HumanEval/Plus 同一题为 0.0/0.0；原始 MBPP 1 题为 1.0；MBPP+ 378 集合中的首题为 base=1.0、plus=1.0；BBH 27 题 group=8/27≈0.2963；DROP 1 题 EM/F1 均为 0.0；MMLU 57 题 acc=31/57≈0.5439；MMLU-Pro 14 题为 0.0。小样本 group 权重与完整集合的权重分布不同，不能把这些数放进论文的全量成绩表。

MATH500 的实际差异来自输出“Final Answer: (3, π/2)”没有严格沿用 Minerva 的“The final answer is ... I hope it is correct.”形式；Minerva 提取路径与 math_verify 的解析路径因此给出不同结果。这是可追踪的解析差异，不是把一次输出解释成两个模型能力结论，也没有为抬高成绩临时改变主评分器。

BBH 有 4/27 条 get-answer 提取失败；MMLU-Pro 有 1/14 条 custom-extract 失败。按 tokenizer 重新编码输出估计，BBH 有 1 条接近 1024 输出预算，Pro 有 2 条接近 2048；原始日志可见重复生成。这项 near_output_budget 是输出重新编码长度的诊断，原生 harness 没有保存 vLLM 的 finish_reason，因此不能将它等同于已严格确认的 length 终止。

HF 脚本和裸 CLI 的 GSM8K 8-shot/单题结果均为 strict=1、flexible=1，extra_checks.json 记录退出码都为 0。语法检查与功能结果必须分开：本次 HumanEval 候选语法有效，但基础/Plus 测试均失败。

### 9. 评分器验证与结果如何复核

verify_standard.py 已通过：HumanEval 与 MBPP 的官方 reference candidate 在 base/plus 都为 1；明确错误程序都为 0；只记忆 HumanEval/0 基础测试输入的审计夹具为 base=1、plus=0，确认增强测试确实被执行；原始 MBPP 的 canonical 程序通过，错误程序失败；DROP 的数字 8 与 9 的 EM/F1 均为 0；全部冻结文件哈希一致。审计夹具仅用于验证评分器，没有提供给模型。

```bash
cd /data/lpg_projects/mj/test
.venv-standard/bin/python scripts/verify_standard.py
.venv-standard/bin/python scripts/summarize_standard.py --experiment smoke_final
```

summary.json 保存每项与分项的指标、唯一题数和失败；*_results.json 保存展开后的 task 配置与聚合；*_samples.jsonl 保存 doc_id、题目、完整实际提示 arguments、原始 resps、filtered_resps、目标和每题分数；request_lengths.jsonl 保存每个请求的 token 长度与 prompt SHA256；code_diagnostics.jsonl 保存基础/Plus 状态和语法诊断；diagnostics.json 汇总空输出、提取失败和接近预算的条数。run_config.json 保存命令、模型路径、权重/配置/tokenizer 哈希、种子与库版本。

每轮 checkpoint 需要新 experiment 名称。score 下降时先看相同 doc_id 的 raw output、提取结果、输出预算和程序失败状态；如果 task/data/scorer 哈希不同，应先判断协议变了。GSM 两个 filter 的日志要按 doc_id 去重统计题数；MMLU 的 resps 是 likelihood 数值结果，不是生成文本，不能和 generate_until 一样直接调用 strip 或统计生成 token 数。

### 10. 后续用于训练曲线的约束

当前 ten-benchmark 主协议的正式入口已经可运行，但本次没有执行全部约数万题，也没有测量各 checkpoint 的稳定方差。训练监控若采用子集，应先冻结 sample IDs 和每项数量，保持所有 checkpoint 相同；这次 --limit 1 是取各叶子任务的首题，仅用于 smoke，不作为建议的长期研究子集。完整 MMLU/MMLU-Pro 与 BBH 使用样本数加权；不能把固定每学科 1 题的结果当成完整集合的加权结果。

greedy pass@1 是每题单候选通过率。若将来增加 sampled pass@k，需要固定每题候选数、temperature/top-p/seed，并用合法的 pass@k 估计；不能把多个候选中挑出的成功代码当作单次 greedy 成绩。对不同模型沿用相同 task/prompt/scorer，但 tokenizer 和上下文能力必须单独记录；超限时明确拒绝或另建预定义截断实验，不让不同模型静默删掉不同题目。

共享 GPU 的使用率与显存当时均已有其他任务，所以日志中的 elapsed_seconds 用于定位运行问题，不是独占硬件延迟 benchmark。本次 vLLM/HF 评测进程完成后退出，未启动常驻推理服务。个别 pandas/numexpr 版本提示、HF greedy 参数提示以及 vLLM 子进程退出时 NCCL 警告没有造成评测失败；判断完成应检查 RUN_COMPLETE、summary.json 的 failures 和 extra_checks.json 的退出码，而不是只看最后一条 warning。
