# -*- coding: utf-8 -*-
"""合成研究问题池：多领域、需要多步检索的深度研究问题，写入 data/questions.jsonl。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.deepseek import chat_json

# 领域 -> 子主题 / 切入角度（用于引导多样性）
DOMAINS = {
    "人工智能与前沿科技": [
        "大模型训练与推理优化", "AI Agent 与工作流", "多模态模型", "RAG 与长上下文",
        "端侧/小模型部署", "AI 编程与开发者工具", "AI 芯片与算力", "AI 安全与对齐",
    ],
    "商业与行业研究": [
        "市场格局与竞争", "公司商业模式", "供应链与成本结构", "出海与全球化",
        "新消费品牌", "平台经济", "制造业升级", "企业服务 SaaS",
    ],
    "经济金融与政策": [
        "货币政策与利率", "财政与税收", "房地产与地方债", "资本市场改革",
        "汇率与跨境资本", "产业政策", "养老金与社保", "银行与不良资产",
    ],
    "科学与工程": [
        "生物技术与基因", "新材料", "半导体制造", "航天与卫星",
        "量子计算", "机器人与自动化", "脑科学", "基础物理进展",
    ],
    "能源环境与气候": [
        "新能源发电与并网", "储能与电池", "电动汽车产业链", "氢能",
        "碳市场与减排", "极端天气与气候", "核电", "矿业与关键矿产",
    ],
    "历史人文与社会": [
        "历史事件的多因素分析", "考古新发现", "文化比较", "语言与文字",
        "宗教与哲学", "人口与家庭结构", "城市化与迁徙", "技术史",
    ],
    "健康与医学科普": [
        "常见病机制与循证治疗", "药物原理与副作用", "疫苗与公共卫生",
        "营养与代谢", "运动与康复", "心理健康", "老龄化疾病", "医学影像与诊断",
    ],
    "教育职业与发展": [
        "升学与专业选择", "职业路径与技能", "留学与海外就业", "在线教育",
        "考证与资格", "科研职业", "远程办公", "收入与行业回报",
    ],
    "法律法规与治理": [
        "劳动与就业法规", "知识产权", "数据合规与隐私", "消费者权益",
        "公司治理", "税务合规", "互联网监管", "国际贸易规则",
    ],
    "生活消费与社会现象": [
        "大宗消费品选购", "保险与理财", "旅游与出行规划", "家居与家电",
        "数字产品评测", "婚育与家庭决策", "城市生活成本", "公共服务体验",
    ],
}

SYS = (
    "你是一个研究课题设计专家。请围绕给定领域和子主题，生成一批【高质量、需要多步网络检索才能回答】"
    "的中文研究问题。好问题的标准：需要拆解、比较、综合多个信息源，或需要追踪最新进展，"
    "不能是一句话就能查到的简单事实（例如“某公司成立于哪一年”就是坏问题）。\n"
    "问题要具体、有现实调研价值，措辞多样，覆盖不同子主题和难度。\n"
    "只输出 JSON：{\"questions\": [\"问题1\", \"问题2\", ...]}。"
)

OUT_FILE = ROOT / "data" / "questions.jsonl"


def main(n_per_call: int = 25, calls_per_domain: int = 2):
    if OUT_FILE.exists():
        OUT_FILE.unlink()
    total = 0
    seen = set()
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for domain, topics in DOMAINS.items():
            got_domain = 0
            for call in range(calls_per_domain):
                topic_hint = topics
                user = (
                    f"领域：{domain}\n可参考的子主题：{', '.join(topic_hint)}\n"
                    f"请生成 {n_per_call} 个互不重复的研究问题（第 {call+1} 批，"
                    f"角度尽量与本批其他问题不同）。"
                )
                data = chat_json(
                    [{"role": "system", "content": SYS},
                     {"role": "user", "content": user}],
                    temperature=0.95, max_tokens=2200,
                )
                if not data or "questions" not in data:
                    print(f"[warn] {domain} 第{call+1}批生成失败")
                    continue
                for q in data["questions"]:
                    q = q.strip()
                    if not q or q in seen or len(q) < 8:
                        continue
                    seen.add(q)
                    total += 1
                    got_domain += 1
                    f.write(json.dumps(
                        {"id": total, "domain": domain, "question": q},
                        ensure_ascii=False) + "\n")
            print(f"{domain}: 本领域 {got_domain} 条，累计 {total}")
    print(f"完成，共 {total} 条 -> {OUT_FILE}")


if __name__ == "__main__":
    main()
