# -*- coding: utf-8 -*-
"""步骤 0：基线 SFT（LoRA），用 SwanLab 记录训练曲线。"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import swanlab
from dotenv import load_dotenv

from configs.paths import SFT_TRAIN_FILE, SFT_VAL_FILE, SFT_CKPT
from utils.training import run_lora_train


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(epochs: int = 3, lr: float = 2e-4, grad_accum: int = 8):
    load_dotenv(ROOT / ".env")
    swanlab.login(os.getenv("SWANLAB_API_KEY"))

    train = load_jsonl(SFT_TRAIN_FILE)
    val = load_jsonl(SFT_VAL_FILE)
    print(f"train {len(train)} | val {len(val)}")

    swanlab.init(
        project="retrieval-planner",
        experiment_name="sft_baseline",
        config={"stage": "sft", "train": len(train), "val": len(val),
                "epochs": epochs, "lr": lr, "grad_accum": grad_accum},
    )
    run_lora_train(
        train, str(SFT_CKPT),
        epochs=epochs, lr=lr, grad_accum=grad_accum,
        log_fn=swanlab.log, tag="sft",
    )
    swanlab.finish()
    print("基线 SFT 完成。")


if __name__ == "__main__":
    main()
