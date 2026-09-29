# -*- coding: utf-8 -*-
"""步骤 0 前置：从 ModelScope 下载 Qwen2.5-1.5B-Instruct 到 D 盘。"""
import sys
from pathlib import Path

# 让脚本能 import configs.paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modelscope import snapshot_download
from configs.paths import MODEL_DIR


def main():
    print(f"目标目录: {MODEL_DIR}")
    if (MODEL_DIR / "config.json").exists():
        print(f"模型已存在，跳过下载: {MODEL_DIR}")
        return
    model_dir = snapshot_download(
        "Qwen/Qwen2.5-1.5B-Instruct",
        local_dir=str(MODEL_DIR),
    )
    print(f"下载完成: {model_dir}")


if __name__ == "__main__":
    main()
