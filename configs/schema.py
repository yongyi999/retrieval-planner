# -*- coding: utf-8 -*-
"""检索规划器的 JSON schema、prompt 模板与校验/解析工具。"""
import json
import re
from typing import Any, Dict, List

# ----------------------------------------------------------------------------
# 规划 JSON 的标准结构（示例）
# {
#   "goal": "研究目标的一句话概括",
#   "steps": [
#     {
#       "id": 1,
#       "action": "search",
#       "query": "给搜索引擎的具体查询词",
#       "purpose": "这一步要获取什么信息 / 解决什么子问题",
#       "depends_on": []          # 无依赖为 []；否则填依赖步骤的 id
#     }
#   ]
# }
# ----------------------------------------------------------------------------

VALID_ACTIONS = ("search",)
MIN_STEPS = 2
MAX_STEPS = 8

SYSTEM_PROMPT = (
    "你是一个专业的【检索规划器】。给定一个需要通过网络检索才能回答的研究问题，"
    "你要把它拆解成若干有序、可执行的搜索步骤，并输出一份结构化的检索规划。\n"
    "要求：\n"
    "1. 只输出一个 JSON 对象，不要输出 markdown 代码块、解释或多余文字；\n"
    "2. steps 中的每个搜索步骤都要具体、可执行，query 应是适合直接放进搜索引擎的查询词；"
    "后面的步骤可以依赖前面步骤的结果（用 depends_on 声明）；\n"
    f"3. 步骤数量控制在 {MIN_STEPS}~{MAX_STEPS} 步；\n"
    "4. JSON 必须严格符合下面的格式：\n"
    "{\n"
    '  "goal": "研究目标的一句话概括",\n'
    '  "steps": [\n'
    '    {"id": 1, "action": "search", "query": "搜索词", "purpose": "这一步要获取什么", "depends_on": []}\n'
    "  ]\n"
    "}"
)


def build_user_prompt(question: str) -> str:
    """把研究问题包装成给规划器的 user 消息。"""
    return f"研究问题：{question}\n请输出检索规划 JSON。"


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出中抽取 JSON 对象（兼容 ```json 代码块包裹的情况）。"""
    text = text.strip()
    # 优先尝试直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 去除 markdown 代码块
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            text = candidate
    # 兜底：抓取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("无法从输出中解析 JSON")


def validate_plan(plan: Dict[str, Any]) -> List[str]:
    """校验规划，返回错误信息列表；为空表示合法。"""
    errors: List[str] = []
    if not isinstance(plan, dict):
        return ["规划不是 JSON 对象"]
    goal = plan.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        errors.append("缺少非空字段 goal")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps 必须是非空数组")
        return errors
    if not (MIN_STEPS <= len(steps) <= MAX_STEPS):
        errors.append(f"步骤数量 {len(steps)} 超出 {MIN_STEPS}~{MAX_STEPS}")
    seen_ids = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"第 {i+1} 步不是对象")
            continue
        sid = step.get("id")
        if not isinstance(sid, int) or sid <= 0:
            errors.append(f"第 {i+1} 步缺少正整数 id")
        else:
            if sid in seen_ids:
                errors.append(f"步骤 id {sid} 重复")
            seen_ids.add(sid)
        if step.get("action") not in VALID_ACTIONS:
            errors.append(f"步骤 {sid} 的 action 非法：{step.get('action')}")
        if not isinstance(step.get("query"), str) or not step["query"].strip():
            errors.append(f"步骤 {sid} 缺少非空 query")
        if not isinstance(step.get("purpose"), str) or not step["purpose"].strip():
            errors.append(f"步骤 {sid} 缺少非空 purpose")
        deps = step.get("depends_on", [])
        if not isinstance(deps, list):
            errors.append(f"步骤 {sid} 的 depends_on 必须是数组")
        else:
            for d in deps:
                if not isinstance(d, int) or d <= 0:
                    errors.append(f"步骤 {sid} 的依赖 id 非法：{d}")
                elif isinstance(sid, int) and d >= sid:
                    errors.append(f"步骤 {sid} 不能依赖同步骤或更后的步骤 {d}")
    return errors


def plan_to_json(plan: Dict[str, Any]) -> str:
    """规范化输出：紧凑、可复现的 JSON 字符串。"""
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
