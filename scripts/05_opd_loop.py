# -*- coding: utf-8 -*-
"""步骤 1：OPD 同策略蒸馏循环。
每轮：①学生 on-policy 采样 → ②老师逐段评判/修正 → ③构造训练样本 → ④LoRA 续训。"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import swanlab
from dotenv import load_dotenv

from configs.paths import (
    QUESTIONS_FILE, SFT_CKPT, OPD_DIR,
)
from utils.inference import StudentModel
from utils.teacher import refine_plan, refine_raw
from utils.training import run_lora_train

# 规模参数
N_SFT = 200
N_EVAL = 30
BATCH_PER_ROUND = 40     # 每轮采样问题数
ROUNDS = 3
MAX_WORKERS = 8
OPD_EPOCHS = 2
OPD_LR = 1e-4


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_correct(question, student_plan, student_raw):
    """学生合法则逐段评判，非法则纠正格式。"""
    if student_plan is not None:
        r = refine_plan(question, student_plan)
    else:
        r = refine_raw(question, student_raw)
    return r


def main():
    load_dotenv(ROOT / ".env")
    swanlab.login(os.getenv("SWANLAB_API_KEY"))

    questions = load_jsonl(QUESTIONS_FILE)
    opd_qs = questions[N_SFT + N_EVAL:]
    print(f"OPD 问题池 {len(opd_qs)} 条")

    current_adapter = str(SFT_CKPT)
    swanlab.init(
        project="retrieval-planner",
        experiment_name="opd_loop",
        config={"batch_per_round": BATCH_PER_ROUND, "rounds": ROUNDS,
                "opd_epochs": OPD_EPOCHS, "opd_lr": OPD_LR},
    )

    cursor = 0
    for rnd in range(1, ROUNDS + 1):
        print(f"\n========== OPD 第 {rnd}/{ROUNDS} 轮 ==========")
        batch = opd_qs[cursor:cursor + BATCH_PER_ROUND]
        cursor += BATCH_PER_ROUND
        if len(batch) < BATCH_PER_ROUND:
            print("OPD 问题不足，结束。")
            break

        # ① On-policy 采样（批量）
        student = StudentModel(adapter_path=current_adapter)
        questions = [q["question"] for q in batch]
        gen_results = student.generate_batch(questions, batch_size=8)
        samples = [
            {"question": q, "student_plan": r[0],
             "student_raw": r[1], "student_errors": r[2]}
            for q, r in zip(questions, gen_results)
        ]
        student.close()
        valid = sum(1 for s in samples if s["student_plan"] is not None)
        valid_rate = valid / len(samples)
        print(f"① 采样完成：合法 {valid}/{len(samples)}，合法率 {valid_rate:.2%}")

        # ② 老师逐段评判（并发）
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(teacher_correct, s["question"],
                              s["student_plan"], s["student_raw"]): s
                    for s in samples}
            for fut in as_completed(futs):
                s = futs[fut]
                r = fut.result()
                s["critique"] = r["critique"]
                s["teacher_plan"] = r["plan"]
        corrected = sum(1 for s in samples if s["teacher_plan"] is not None)
        print(f"② 老师修正完成：得到 {corrected} 份优质规划")

        # ③ 保存本轮完整记录 + 构造训练样本
        round_file = ROOT / "data" / f"opd_round_{rnd}.jsonl"
        with round_file.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        train_rows = [{"question": s["question"], "plan": s["teacher_plan"]}
                      for s in samples if s["teacher_plan"] is not None]
        print(f"③ 训练样本 {len(train_rows)} 条，记录 -> {round_file}")

        # ④ LoRA 续训
        next_adapter = str(OPD_DIR / f"round_{rnd}")
        run_lora_train(
            train_rows, next_adapter,
            init_adapter=current_adapter,
            epochs=OPD_EPOCHS, lr=OPD_LR, grad_accum=4,
            log_fn=swanlab.log, tag=f"opd_r{rnd}",
        )
        current_adapter = next_adapter
        swanlab.log({f"opd/student_valid_rate": valid_rate,
                     f"opd/round": rnd})
        print(f"④ 第 {rnd} 轮完成，当前 adapter: {current_adapter}")

    swanlab.finish()
    print("\nOPD 循环全部结束。")


if __name__ == "__main__":
    main()
