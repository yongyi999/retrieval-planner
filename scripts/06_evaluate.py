# -*- coding: utf-8 -*-
"""在固定评估集上对比：基座 / SFT / OPD 最终版 的 JSON 合法率与老师质量评分。"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import swanlab
from dotenv import load_dotenv

from configs.paths import EVAL_QUESTIONS_FILE, SFT_CKPT, OPD_DIR, EVAL_DIR
from utils.inference import StudentModel
from utils.deepseek import chat_json

SCORE_SYS = (
    "你是严格的检索规划质量评审。请为给定研究问题的检索规划打分（1~10 分），"
    "从以下维度综合：是否覆盖关键子问题、搜索词是否具体可执行、步骤顺序与依赖是否合理、"
    "是否存在冗余。只输出 JSON：{\"score\": 数字, \"reason\": \"简短理由\"}。"
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_one(question, plan):
    from configs.schema import plan_to_json
    user = f"研究问题：{question}\n检索规划：{plan_to_json(plan)}"
    d = chat_json([{"role": "system", "content": SCORE_SYS},
                   {"role": "user", "content": user}],
                  temperature=0.0, max_tokens=300)
    if d and isinstance(d.get("score"), (int, float)):
        return min(10, max(1, float(d["score"])))
    return None


def eval_checkpoint(adapter_path, name, questions, max_workers=8):
    print(f"\n=== 评估 {name} ({adapter_path or 'base'}) ===")
    student = StudentModel(adapter_path=adapter_path)
    qs = [q["question"] for q in questions]
    gen = student.generate_batch(qs, batch_size=8, temperature=0.3)
    out = [{"question": q, "plan": r[0], "errors": r[2]}
           for q, r in zip(qs, gen)]
    student.close()

    valid = [o for o in out if o["plan"] is not None]
    valid_rate = len(valid) / len(out)

    # 对合法规划并发打分
    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_one, o["question"], o["plan"]): o for o in valid}
        for fut in as_completed(futs):
            s = fut.result()
            if s is not None:
                scores.append(s)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    n_steps = [len(o["plan"]["steps"]) for o in valid]
    avg_steps = sum(n_steps) / len(n_steps) if n_steps else 0.0

    result = {"name": name, "valid_rate": valid_rate, "valid": len(valid),
              "total": len(out), "avg_score": avg_score, "avg_steps": avg_steps}
    print(f"合法率 {valid_rate:.2%} ({len(valid)}/{len(out)}) | "
          f"平均分 {avg_score:.2f} | 平均步数 {avg_steps:.1f}")
    return result, out


def main():
    load_dotenv(ROOT / ".env")
    swanlab.login(os.getenv("SWANLAB_API_KEY"))
    questions = load_jsonl(EVAL_QUESTIONS_FILE)
    print(f"评估集 {len(questions)} 条")

    # 找到 OPD 最终轮
    opd_final = None
    if OPD_DIR.exists():
        rounds = sorted(OPD_DIR.glob("round_*"),
                        key=lambda p: int(p.name.split("_")[1]))
        if rounds:
            opd_final = str(rounds[-1])

    checkpoints = [
        (None, "base"),
        (str(SFT_CKPT) if SFT_CKPT.exists() else None, "sft"),
        (opd_final, "opd_final"),
    ]
    checkpoints = [c for c in checkpoints if c[1] == "base" or c[0]]

    swanlab.init(project="retrieval-planner", experiment_name="evaluation")
    summary = []
    for path, name in checkpoints:
        result, out = eval_checkpoint(path, name, questions)
        summary.append(result)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with (EVAL_DIR / f"{name}_outputs.jsonl").open("w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        swanlab.log({f"eval/{name}_valid_rate": result["valid_rate"],
                     f"eval/{name}_score": result["avg_score"]})

    with (EVAL_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    swanlab.finish()
    print("\n评估完成，汇总：")
    for r in summary:
        print(f"  {r['name']:10s} 合法率 {r['valid_rate']:.2%} 平均分 {r['avg_score']:.2f}")


if __name__ == "__main__":
    main()
