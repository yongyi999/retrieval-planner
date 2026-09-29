# -*- coding: utf-8 -*-
"""交互式启动检索规划器：输入研究问题，输出检索规划 JSON。
用法： .venv\\Scripts\\python.exe run.py
"""
import json
from utils.inference import StudentModel

# 最终模型（三轮 OPD）。想换成基线可改为 "outputs/sft_baseline"
ADAPTER = "outputs/opd/round_3"


def main():
    print("加载模型中，请稍候 ...", flush=True)
    m = StudentModel(adapter_path=ADAPTER)
    print("模型已就绪。输入研究问题后回车，得到检索规划；输入 q 退出。\n")

    while True:
        try:
            q = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            # 空行不退出，继续等待输入
            continue
        if q.lower() in ("q", "quit", "exit"):
            break

        print("正在生成检索规划，请稍候 10~30 秒 ...", flush=True)
        plan, raw, errors = m.generate(q, temperature=0.3)
        if plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            print("生成失败：", errors)
            print(raw[:500])
        print()

    print("已退出。")


if __name__ == "__main__":
    main()
