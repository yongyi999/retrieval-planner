# -*- coding: utf-8 -*-
"""全局路径与常量配置。"""
from pathlib import Path
import os

# 项目根目录（本文件位于 configs/ 下）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "configs"

# 学生模型下载目录（放在 D 盘，节省 C 盘空间）
MODEL_DIR = Path(r"D:\models\Qwen2.5-1.5B-Instruct")

# 数据文件
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"        # 合成的研究问题池
SFT_TRAIN_FILE = DATA_DIR / "sft_train.jsonl"        # 基线 SFT 训练集
SFT_VAL_FILE = DATA_DIR / "sft_val.jsonl"            # 基线 SFT 验证集
EVAL_QUESTIONS_FILE = DATA_DIR / "eval_questions.jsonl"  # 固定评估问题（不参与训练）

# 模型输出
SFT_CKPT = OUTPUT_DIR / "sft_baseline"               # 基线 SFT 的 LoRA 权重
OPD_DIR = OUTPUT_DIR / "opd"                         # OPD 每轮 LoRA 权重
EVAL_DIR = OUTPUT_DIR / "eval"

# DeepSeek API
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_TEACHER = "deepseek-chat"                   # 老师模型

for _d in (DATA_DIR, OUTPUT_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
