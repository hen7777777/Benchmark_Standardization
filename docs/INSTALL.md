# 环境与依赖安装

`harness_builtin_suite` 只依赖一个能导入 `lm_eval` / `transformers` / `datasets` / `evaluate` 的
Python 环境，外加一块可用的 NVIDIA GPU。`evaluate.sh` 依次探测
`LMEVAL_PYTHON` → `suite.local.env` → `.venv/bin/python` → `.venv-standard/bin/python` →
`PATH` 上的 `python3` / `python`，取第一个能导入上述四个包的解释器；都不行就以退出码 2
结束并打印候选列表，不会静默换环境。

## 已验证环境

| 项 | 值 |
|---|---|
| Python | 3.13.5 |
| GPU | NVIDIA L40（本机 6 卡，共享） |
| harness | `d6de81643928d653435c431bae19945d41d32520`（editable Git checkout） |
| 关键版本 | vLLM 0.17.1、torch 2.10.0、transformers 4.57.1、datasets 5.0.1 |
| API 客户端 | aiohttp 3.14.3、requests 2.34.2、tenacity 9.1.4（默认 `--model-backend server` 需要） |

完整清单见 [`requirements.txt`](../requirements.txt)。实测的 `numpy==2.4.4`、
`setuptools==84.0.0`、`numba==0.61.2` 与 `numpy`/`setuptools` 的声明上限冲突，属于
L40 继承 `/home/lpg/anaconda3` 环境的结果；`requirements.txt` 里写的是能解析出来的区间，
装完核对版本即可，不要为了“消除冲突”去动这个环境的 numpy / setuptools。

## 从零安装

```bash
cd <suite 根目录>
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# harness 必须来自固定 commit 的 Git checkout：入口会核验 HEAD
mkdir -p vendor
git clone https://github.com/EleutherAI/lm-evaluation-harness.git vendor/lm-evaluation-harness
git -C vendor/lm-evaluation-harness checkout --detach d6de81643928d653435c431bae19945d41d32520
python -m pip install -e ./vendor/lm-evaluation-harness

python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` 里包含 `aiohttp` / `requests` / `tenacity`。默认的
`--model-backend server` 用 harness 的 `local-completions` 客户端通过 HTTP 打共享 vLLM 服务，
需要这三个包；`--model-backend inprocess`（旧的逐阶段 `--model vllm` 引擎）不需要。
缺包时套件会在启动服务之前报错，不会先把模型加载进显存。

两个容易踩的坑：

- **不要**用 `lm_eval[vllm]`。该 extra 要求 `vllm>=0.18`，和已验证的 0.17.1 冲突；
  vLLM 按要求文件里的固定版本单独安装。
- **不要**用 `lm_eval[math]`。它的 ANTLR 版本要求与本项目固定的
  `antlr4-python3-runtime==4.11.1` 不同，装到 4.13 会让 Minerva 评分器导入即断言失败。

## 接现有环境

已有可用环境时不必重装，在 suite 根目录写一个不进 Git 的 `suite.local.env`：

```bash
LMEVAL_PYTHON=/path/to/venv/bin/python
# 可选：RULER_DATA_ROOT=/absolute/path/to/datasets/long_context/ruler
```

`suite.example.env` 是一份可直接复制的最小模板。

## 数据与联网

- `math` / `code` / `complex_reasoning` / `general_reasoning` / `longbench_v1` / `longbench_v2`
  的数据来自 HuggingFace Hub，先联网跑一次 `python prefetch.py` 灌进 `cache/`（不进 Git），
  之后一律离线运行。
- `ruler` 用仓库内的冻结数据 `datasets/long_context/ruler`，不需要联网，也不需要在目标机上重新生成。
- 单独下载 `requirements.txt` 或代码快照不能还原完整离线评测环境：还需要 checkpoint 和 `cache/`。

## 自检

```bash
./evaluate.sh /path/to/checkpoint --gpu 0 --smoke --preflight-only   # 不加载模型，只查任务名/长度/缓存
./evaluate.sh /path/to/checkpoint --gpu 0 --stages math --smoke       # 最小真实推理链路（统一部署，起一次 vLLM 服务）
./evaluate.sh /path/to/checkpoint --gpu 0 --stages math --smoke --model-backend inprocess   # 旧路径对照
```

`--smoke` 跑完会在 `runs/<name>/` 留下 `server.json`、`server.log` 与每个阶段的
`command.json` / `native.log` / `exit.json`；服务在套件退出时就会被终止，
不会留在卡片上。
