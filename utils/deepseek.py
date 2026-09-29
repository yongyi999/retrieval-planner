# -*- coding: utf-8 -*-
"""DeepSeek（老师）API 封装：从 .env 读取 key，提供 JSON 模式调用与重试。"""
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("未在 .env 中找到 DEEPSEEK_API_KEY")

# 加入项目根目录以 import configs
sys.path.insert(0, str(ROOT))
from configs.paths import DEEPSEEK_BASE_URL, DEEPSEEK_TEACHER  # noqa: E402

client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)


def chat_json(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: str = DEEPSEEK_TEACHER,
    retries: int = 4,
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """以 JSON 模式调用 DeepSeek，返回解析后的 dict；失败返回 None。"""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = resp.choices[0].message.content
            import json
            return json.loads(content)
        except Exception as e:  # 限流 / 网络 / 解析错误
            last_err = e
            wait = 2 ** attempt
            time.sleep(wait)
    print(f"[chat_json] 调用失败: {last_err}", file=sys.stderr)
    return None
