# -*- coding: utf-8 -*-
"""学生模型推理：加载基座 + LoRA，支持批量生成检索规划（左 padding，速度快）。"""
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from configs.paths import MODEL_DIR
from configs.schema import (
    SYSTEM_PROMPT, build_user_prompt, extract_json, validate_plan,
)


class StudentModel:
    def __init__(self, adapter_path: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # generate 需要左 padding
        self.tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True)
        if adapter_path and os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            print(f"[student] 加载 adapter: {adapter_path}", flush=True)
        model.cuda()
        model.eval()
        self.model = model

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _parse(self, raw: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        try:
            plan = extract_json(raw)
            errors = validate_plan(plan)
            return (plan if not errors else None), errors
        except Exception as e:
            return None, [f"JSON 解析失败: {e}"]

    @torch.no_grad()
    def generate_batch(
        self, questions: List[str], *, batch_size: int = 8,
        temperature: float = 0.7, top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> List[Tuple[Optional[Dict[str, Any]], str, List[str]]]:
        """批量生成，返回与 questions 等长的 (plan|None, raw, errors) 列表。"""
        results: List[Optional[Tuple]] = [None] * len(questions)
        prompts = [self._build_prompt(q) for q in questions]

        for start in range(0, len(questions), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True).to("cuda")
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=top_p, pad_token_id=self.tokenizer.eos_token_id,
            )
            in_len = inputs["input_ids"].shape[1]
            for j in range(len(batch_prompts)):
                raw = self.tokenizer.decode(
                    gen[j][in_len:], skip_special_tokens=True)
                plan, errors = self._parse(raw)
                results[start + j] = (plan, raw, errors)
        return results

    @torch.no_grad()
    def generate(self, question: str, **kwargs) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
        return self.generate_batch([question], batch_size=1, **kwargs)[0]

    def close(self):
        del self.model
        torch.cuda.empty_cache()
