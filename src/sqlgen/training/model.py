from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from sqlgen.config import LoraParams, ModelConfig


def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, revision=cfg.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: ModelConfig, *, bf16: bool = True):
    return AutoModelForCausalLM.from_pretrained(
        cfg.base,
        revision=cfg.revision,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )


def build_lora(cfg: LoraParams) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
