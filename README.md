# 检索规划器（Retrieval Planner）

输入一个**研究问题**，模型输出一份**结构化检索规划 JSON**：把复杂问题拆成若干有序、带依赖关系的搜索步骤（每步含具体搜索词、目的、依赖）。

训练流程严格分为两步：

- **步骤 0 · 基线 SFT**：用「问题 → 人工/老师写的最优规划」样本，LoRA 微调小模型，让它看懂问题、输出合法 JSON、学会基础拆任务。
- **步骤 1 · OPD 同策略蒸馏循环**：当前学生模型自己生成规划（会犯错）→ 更强的老师大模型逐段评判并修正 → 用老师的优质规划构造样本 → LoRA 续训；下一轮用更新后的学生继续采样，持续学习自己真实会犯的错误。

- **学生模型**：Qwen2.5-1.5B-Instruct（LoRA，可训练参数 1.18%）
- **老师模型**：DeepSeek（`deepseek-chat`，JSON 模式）
- **实验跟踪**：SwanLab

---

## 目录结构

```
检索/
├── configs/
│   ├── paths.py            # 全局路径 / 常量（模型目录、数据划分、老师配置）
│   └── schema.py           # 系统提示词、prompt 构造、JSON 抽取与 schema 校验
├── utils/
│   ├── deepseek.py         # chat_json：JSON 模式 + 重试
│   ├── teacher.py          # 老师：直接生成规划 / 逐段评判修正 / 修复非法 JSON
│   ├── training.py         # 模型加载、挂 LoRA、样本编码、训练循环
│   └── inference.py        # 学生模型批量生成（左 padding）
├── scripts/
│   ├── 01_download_model.py  # 下载 Qwen2.5-1.5B-Instruct（ModelScope）
│   ├── 02_gen_questions.py   # 合成研究问题（10 领域）
│   ├── 03_gen_sft_data.py    # 老师生成 SFT 规划
│   ├── 04_sft_train.py       # 步骤 0：基线 SFT
│   ├── 05_opd_loop.py        # 步骤 1：OPD 蒸馏循环
│   └── 06_evaluate.py        # 固定评估集对比 base/sft/opd
├── data/
│   ├── questions.jsonl          # 501 条研究问题
│   ├── sft_train.jsonl(190) / sft_val.jsonl(10)
│   ├── eval_questions.jsonl     # 30 条固定评估集（永不训练）
│   └── opd_round_1/2/3.jsonl    # 每轮完整记录（学生规划+错误+点评+老师规划）
├── outputs/
│   ├── sft_baseline/        # 基线 SFT LoRA
│   ├── opd/round_1/2/3/     # 三轮 OPD LoRA（round_3 为最终版）
│   └── eval/                # 评估结果 summary.json + 各模型输出
├── run.py                   # 交互式启动脚本（输入问题→输出规划）
├── .env                     # DEEPSEEK_API_KEY / SWANLAB_API_KEY（不入库）
└── requirements.txt
```

---

## 环境与安装

- Windows + NVIDIA 8GB 显存（RTX 4060 Laptop 验证），Python **3.11**（勿用 3.13/3.14，torch 无稳定适配）。
- 用 [uv](https://docs.astral.sh/uv/) 建虚拟环境：

```powershell
uv venv --python 3.11
# 先装 CUDA 版 torch（默认 PyPI 会装到 CPU 版！）
uv pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

在项目根目录建 `.env`：

```
DEEPSEEK_API_KEY=sk-xxxx
SWANLAB_API_KEY=xxxx
```

> 学生模型默认下载到 `D:\models\Qwen2.5-1.5B-Instruct`，可在 `configs/paths.py` 的 `MODEL_DIR` 修改。

---

## 快速开始（按顺序执行）

```powershell
uv run python scripts/01_download_model.py   # 下载模型
uv run python scripts/02_gen_questions.py    # 合成问题
uv run python scripts/03_gen_sft_data.py     # 生成 SFT 数据
uv run python scripts/04_sft_train.py        # 步骤 0：基线 SFT
uv run python scripts/05_opd_loop.py         # 步骤 1：OPD 循环
uv run python scripts/06_evaluate.py         # 评估对比
```

交互式启动（推荐）：

```powershell
.\.venv\Scripts\python.exe run.py
```

出现 `问题>` 后输入研究问题、回车即可（生成约需 10~30 秒），输入 `q` 退出。

在代码中调用：

```python
from utils.inference import StudentModel
m = StudentModel(adapter_path="outputs/opd/round_3")
plan, raw, errors = m.generate("你的研究问题……")  # 返回 (规划|None, 原文, 错误)
print(plan)
```

---

## 试运行示例

输入：`帮我规划下北京旅游路线`，最终模型（`outputs/opd/round_3`，temperature=0.3）输出：

```json
{
  "goal": "制定一条包含必游景点、特色美食、文化体验和住宿推荐的北京旅游线路，满足不同游客需求，包括亲子、情侣、背包客等。",
  "steps": [
    {
      "id": 1,
      "action": "search",
      "query": "北京必游景点 地铁导航 景点分布",
      "purpose": "获取北京主要景点的位置信息及地铁导航，为后续行程安排提供基础。",
      "depends_on": []
    },
    {
      "id": 2,
      "action": "search",
      "query": "北京特色美食 推荐餐厅 餐厅评价",
      "purpose": "收集北京最具代表性的美食种类及其代表性餐厅，丰富饮食体验。",
      "depends_on": []
    },
    {
      "id": 3,
      "action": "search",
      "query": "北京文化体验 地域文化 文物古迹 美术馆",
      "purpose": "了解北京的文化内涵与历史遗迹，增加深度体验内容。",
      "depends_on": []
    },
    {
      "id": 4,
      "action": "search",
      "query": "北京住宿推荐 评分 大酒店 小众民宿",
      "purpose": "获取北京不同预算和需求下的住宿选择，涵盖酒店、民宿等多种类型。",
      "depends_on": []
    },
    {
      "id": 5,
      "action": "search",
      "query": "北京亲子游 家庭活动 游乐园 学校参观",
      "purpose": "针对家庭游客的需求，搜集亲子友好型景点和活动建议。",
      "depends_on": [1]
    },
    {
      "id": 6,
      "action": "search",
      "query": "北京情侣游 花园 高档酒店 摄影地点",
      "purpose": "针对情侣游客的需求，搜集浪漫氛围、高档酒店和摄影打卡地。",
      "depends_on": [1]
    },
    {
      "id": 7,
      "action": "search",
      "query": "北京背包客游 自然风光 漫步公园 青年旅社",
      "purpose": "针对背包客游客的需求，搜集自然风光、徒步和青年旅社等户外体验。",
      "depends_on": [1]
    },
    {
      "id": 8,
      "action": "search",
      "query": "北京旅游线路设计 北京一日游 北京二日游 北京三日游",
      "purpose": "综合以上要素，设计一条完整的北京旅游线路，考虑时长、性价比和多样性。",
      "depends_on": [1, 2, 3, 4]
    }
  ]
}
```

模型把一个开放问题拆成 **8 个搜索步骤**：先并行获取景点、美食、文化、住宿等基础信息（无依赖），再针对亲子 / 情侣 / 背包客等不同人群细分（依赖步骤 1），最后综合形成线路（依赖前面多个步骤），`depends_on` 关系正确、无环路。

---

## 检索规划 JSON Schema

```json
{
  "goal": "本次检索要达成的总体目标",
  "steps": [
    {
      "id": 1,
      "action": "search",
      "query": "具体的搜索词",
      "purpose": "这一步要获取什么",
      "depends_on": []
    }
  ]
}
```

- `steps` 2~8 步；`id` 从 1 递增；`depends_on` 只能引用更早步骤的 id，无依赖为 `[]`。

---

## 训练结果

### 数据规模
- 研究问题 501 条（10 个领域，均为「需多步检索、非一句话可答」的问题）。
- 前 200 条用于 SFT（190 训练 / 10 验证）；201–230 为固定 30 条评估集；其余 271 条为 OPD 问题池。
- LoRA：r=16、alpha=32、dropout=0.05，目标为全部线性层，可训练参数 **18,464,768（1.18%）**。

### 固定评估集（30 条，老师 1–10 分打分）

| 模型 | JSON 合法率 | 老师平均分 | 平均步数 |
|---|---|---|---|
| base（基座） | 73.33% | 5.32 | 4.3 |
| sft（基线 SFT） | **93.33%** | 6.96 | 5.8 |
| opd_final（3 轮 OPD） | 86.67% | **7.00** | 6.4 |

- **SFT 是主要提升来源**：合法率 +20%、质量分 +1.64，模型学会了输出合法 JSON 并做合理的任务拆解。
- **OPD 在合法样本上质量继续小幅提升**（6.96 → 7.00），规划更细致（平均 6.4 步、依赖更完整）。
- OPD 合法率较 SFT 略有回落（93.3%→86.7%）：每轮仅约 29 条样本、2 个 epoch，小模型在少量数据续训后格式稳定性有轻微波动（采样温度 0.3 也带来随机性）。

### Evaluation 各指标含义

固定评估集（`outputs/eval/summary.json`）与 SwanLab `evaluation` 实验中记录的指标：

| 指标 | 含义 |
|---|---|
| **JSON 合法率（valid_rate）** | 30 条固定问题中，模型输出能通过 `validate_plan` schema 校验的比例；非法输出不参与后续打分 |
| **老师平均分（avg_score）** | 对**合法**规划由老师按 1–10 打分后取平均（打分逻辑见下） |
| **平均步数（avg_steps）** | 合法规划的平均步骤数，反映规划的细致程度（步数多通常拆解更细，但过多也可能冗余） |

SwanLab 上对应记录为 `eval/{model}_valid_rate` 与 `eval/{model}_score`（`{model}` 为 base / sft / opd_final）。

### 质量分如何评定（打分逻辑）

质量分**不是人工标注、也不是规则公式计算，而是由 DeepSeek 老师扮演「严格评审」，对「问题 + 规划」文本整体打分**：

1. 仅对**合法**规划打分（非法 JSON 计入合法率，不打分）；
2. 把「原始问题 + 规划」发给老师，老师按以下 **4 个维度综合判断**，输出 `{"score": 数字, "reason": "理由"}`；
3. 程序将分数截断到 1–10；调用时 **temperature=0**，使同一规划多次评分尽量稳定、可复现。

| 维度 | 看什么 | 典型扣分 |
|---|---|---|
| **完整性** | 是否覆盖回答问题所需的关键子问题 | 漏掉关键方面 |
| **可执行性** | `query` 是否具体、能直接放进搜索引擎 | query 太泛、搜不到有用结果 |
| **合理性** | 步骤顺序、`depends_on` 依赖是否正确 | 依赖更后的步骤、顺序混乱 |
| **精简性** | 是否有冗余 / 重复 / 无用步骤 | 步骤重复、凑步数 |

> 这是**整体综合打分**，不是各维度分别打分再加权。
>
> **局限**：① 裁判与生成数据、OPD 批改是同一个 DeepSeek，存在「同源审美」偏差；② 老师只看文本、看不到真实网页，分高 ≠ 实际检索结果好；③ 衡量的是规划文本质量，不是「按规划执行检索 → 最终答案」的端到端效果。可通过换不同模型交叉打分、人工抽检、执行搜索后用答案质量评分来改进。

### 训练 loss
- SFT：3 epochs，loss 约 0.73 → 0.35。
- OPD 三轮（各 2 epochs）：分别收至 0.68 / 0.75 / 0.51；每轮初始 loss（约 0.97–0.99）明显高于 SFT 结束时，说明老师修正规划确实带来了学生需要学习的新内容，而非简单重复。

### SwanLab
- 项目：https://swanlab.cn/@spiderma/retrieval-planner
- SFT run：https://swanlab.cn/@spiderma/retrieval-planner/runs/c91x7xgt
- OPD run：https://swanlab.cn/@spiderma/retrieval-planner/runs/qqs8mcdx
- 评估 run：https://swanlab.cn/@spiderma/retrieval-planner/runs/crcj70x8

---

## 已知局限与改进方向

1. **OPD 规模偏小**：当前 3 轮、每轮约 29 条。增加轮次、每轮样本量（如 80–120）通常能让质量提升更明显；每轮训练时可**混入部分 SFT 数据做 replay**，缓解格式退化。
2. **老师看不到真实检索结果**：只从文本层面判断规划好坏，无法预判「这个搜索词实际搜出来的网页质量」。可在老师评判时接入真实搜索 API，把 top 结果摘要喂给老师做 grounding。
3. **老师调用成本**：OPD 每轮都要调用老师，问题越多成本越高；可缓存、去重、并对简单问题跳过批改。
4. **评估维度**：当前为老师单一打分，可增加多老师交叉打分、人工评测、以及「按规划真实执行检索后的答案质量」端到端指标。
5. **学生模型较小（1.5B）**：复杂长规划能力有限，可换 3B/7B 学生进一步提升上限。

---

## 安全提示

本项目的 API Key 仅通过 `.env` 读取，代码中不硬编码。若 Key 曾在对话或日志中明文出现，**任务完成后请尽快在对应平台轮换（重置）Key**。
