# -*- coding: utf-8 -*-
"""公共训练模块：加载基座、挂载/续训 LoRA、构造只对规划部分计 loss 的样本、训练循环。
SFT（步骤0）与 OPD（步骤1）均复用 run_lora_train。"""
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

from configs.paths import MODEL_DIR
from configs.schema import SYSTEM_PROMPT, build_user_prompt, plan_to_json


# ---------------------------------------------------------------- 模型加载
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()
    return model


def attach_lora(model, init_adapter: Optional[str] = None,
                r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """init_adapter 为空：新建 LoRA；否则加载已有 adapter 继续训练（OPD 续训）。"""
    target = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    if init_adapter is not None and os.path.exists(init_adapter):
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        print(f"[lora] 续训已有 adapter: {init_adapter}")
    else:
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target, task_type="CAUSAL_LM", bias="none",
        )
        model = get_peft_model(model, cfg)
        print("[lora] 新建 LoRA")
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------- 数据构造
def encode_example(question: str, plan: Dict, tokenizer, max_len: int = 1024):
    """构造一条训练样本：prompt 部分 mask 掉，只对 assistant 规划 tokens 计 loss。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    answer_text = plan_to_json(plan)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > max_len:  # 截断左侧 prompt，保留完整答案
        cut = len(input_ids) - max_len
        input_ids = input_ids[cut:]
        labels = labels[cut:]
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            # 右 padding
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ---------------------------------------------------------------- 训练循环
def run_lora_train(
    rows: List[Dict],
    output_dir: str,
    *,
    init_adapter: Optional[str] = None,
    epochs: int = 3,
    lr: float = 2e-4,
    micro_batch: int = 1,
    grad_accum: int = 8,
    max_len: int = 1024,
    warmup_ratio: float = 0.05,
    log_fn: Optional[Callable] = None,
    tag: str = "train",
):
    """rows: [{"question":..., "plan": {...}}, ...]，返回保存的 adapter 目录。"""
    tokenizer = load_tokenizer()
    model = load_base_model()
    model = attach_lora(model, init_adapter=init_adapter)

    enc = [encode_example(r["question"], r["plan"], tokenizer, max_len) for r in rows]
    loader = DataLoader(
        enc, batch_size=micro_batch, shuffle=True,
        collate_fn=Collator(tokenizer.pad_token_id), num_workers=0)

    total_steps = max(1, (len(loader) * epochs) // grad_accum)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * warmup_ratio), total_steps)

    model.train()
    step, micro, running = 0, 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += out.loss.item()
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                avg = running / grad_accum
                running = 0.0
                msg = f"[{tag}] epoch {epoch+1}/{epochs} step {step}/{total_steps} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e}"
                print(msg)
                if log_fn:
                    log_fn({f"{tag}/loss": avg, f"{tag}/lr": sched.get_last_lr()[0],
                            f"{tag}/epoch": epoch + 1})

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{tag}] LoRA 已保存 -> {output_dir}")
    # 释放显存
    del model, optim, sched
    torch.cuda.empty_cache()
    return output_dir
