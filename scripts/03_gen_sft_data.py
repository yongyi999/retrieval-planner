# -*- coding: utf-8 -*-
"""步骤 0 数据准备：划分问题池，用老师（DeepSeek）合成 SFT 训练/验证集。"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import (
    QUESTIONS_FILE, SFT_TRAIN_FILE, SFT_VAL_FILE, EVAL_QUESTIONS_FILE,
)
from utils.teacher import generate_plan

# 数据规模（可按需调整）
N_SFT = 200          # SFT 问题数
N_EVAL = 30          # 固定评估问题数（不参与训练）
VAL_RATIO = 0.05     # SFT 中验证集占比
MAX_WORKERS = 8      # 并发调用数


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    questions = load_jsonl(QUESTIONS_FILE)
    print(f"问题池共 {len(questions)} 条")

    sft_qs = questions[:N_SFT]
    eval_qs = questions[N_SFT:N_SFT + N_EVAL]
    opd_qs = questions[N_SFT + N_EVAL:]
    print(f"划分：SFT {len(sft_qs)} | Eval {len(eval_qs)} | OPD {len(opd_qs)}")

    # 固定评估集（只存问题，永不训练）
    write_jsonl(EVAL_QUESTIONS_FILE, eval_qs)

    # 并发生成 SFT 规划
    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(generate_plan, q["question"]): q for q in sft_qs}
        for fut in as_completed(futures):
            q = futures[fut]
            plan = fut.result()
            done += 1
            if plan is not None:
                results[q["id"]] = {"question": q["question"], "plan": plan}
            if done % 20 == 0:
                print(f"  生成进度 {done}/{len(sft_qs)}，成功 {len(results)}")

    rows = [results[q["id"]] for q in sft_qs if q["id"] in results]
    n_val = max(1, int(len(rows) * VAL_RATIO))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    write_jsonl(SFT_TRAIN_FILE, train_rows)
    write_jsonl(SFT_VAL_FILE, val_rows)
    print(f"完成：train {len(train_rows)} 条，val {len(val_rows)} 条")
    print(f"  -> {SFT_TRAIN_FILE}\n  -> {SFT_VAL_FILE}\n  -> {EVAL_QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
