# -*- coding: utf-8 -*-
"""老师（DeepSeek）能力：①直接生成最优规划；②OPD 中逐段评判并修正学生规划。"""
import json
from typing import Any, Dict, Optional

from utils.deepseek import chat_json
from configs.schema import (
    SYSTEM_PROMPT, build_user_prompt, validate_plan, plan_to_json,
)

# 标准规划的 JSON 格式说明（用于老师输出）
_PLAN_FORMAT = (
    '{"goal": "研究目标", "steps": [{"id": 1, "action": "search", '
    '"query": "搜索词", "purpose": "这一步要获取什么", "depends_on": []}]}'
)

REFINE_SYS = (
    "你是资深的【检索规划评审专家】。给你一个研究问题和一名学生写的检索规划，"
    "学生的规划可能存在错误，例如：搜索词太宽泛或太具体、遗漏关键子问题、步骤顺序不合理、"
    "依赖关系错误、步骤冗余、或某一步无法支撑最终问题的回答。\n"
    "请你：1）逐段（逐步）评判学生规划，明确指出每一处问题；"
    "2）在学生规划基础上，产出一份高质量、可执行的检索规划。\n"
    "只输出一个 JSON 对象，格式为：\n"
    '{"critique": "逐段点评（可分点）", "plan": ' + _PLAN_FORMAT + "}"
)


def generate_plan(question: str, *, temperature: float = 0.3) -> Optional[Dict[str, Any]]:
    """老师直接为问题生成最优规划，校验通过后返回；否则 None。"""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    data = chat_json(msgs, temperature=temperature, max_tokens=1600)
    if data and not validate_plan(data):
        return data
    return None


def refine_plan(
    question: str, student_plan: Dict[str, Any], *, temperature: float = 0.3
) -> Dict[str, Optional[Any]]:
    """OPD：老师逐段评判学生规划并给出修正版。
    返回 {"critique": str, "plan": dict|None}。"""
    user = (
        f"研究问题：{question}\n"
        f"学生写的检索规划：{plan_to_json(student_plan)}\n"
        "请逐段评判并输出修正后的规划。"
    )
    data = chat_json(
        [{"role": "system", "content": REFINE_SYS},
         {"role": "user", "content": user}],
        temperature=temperature, max_tokens=2400,
    )
    if not data:
        return {"critique": None, "plan": None}
    critique = data.get("critique")
    plan = data.get("plan")
    if isinstance(plan, dict) and validate_plan(plan):
        plan = None
    return {"critique": critique, "plan": plan}


RAW_REFINE_SYS = (
    "你是资深的【检索规划评审专家】。学生被要求为研究问题输出检索规划 JSON，"
    "但他的输出不是合法、规范的 JSON（可能格式错误、字段缺失、夹带解释文字或代码块）。\n"
    "请指出其问题，并直接给出一份符合要求的检索规划。\n"
    "只输出一个 JSON 对象，格式为：\n"
    '{"critique": "问题点评", "plan": ' + _PLAN_FORMAT + "}"
)


def refine_raw(question: str, student_text: str, *, temperature: float = 0.3):
    """学生输出无法解析为合法 JSON 时，让老师纠正格式并给出规划。"""
    user = (
        f"研究问题：{question}\n"
        f"学生的原始输出：\n{student_text}\n"
        "请指出问题并给出正确的检索规划 JSON。"
    )
    data = chat_json(
        [{"role": "system", "content": RAW_REFINE_SYS},
         {"role": "user", "content": user}],
        temperature=temperature, max_tokens=2400,
    )
    if not data:
        return {"critique": None, "plan": None}
    critique = data.get("critique")
    plan = data.get("plan")
    if isinstance(plan, dict) and validate_plan(plan):
        plan = None
    return {"critique": critique, "plan": plan}
