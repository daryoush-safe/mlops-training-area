from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

from sqlgen.config import Params, load_params
from sqlgen.prompts.presenter import Presentation, build_messages, parse_output
from sqlgen.training.model import load_base_model, load_tokenizer

log = logging.getLogger(__name__)


class Presenter:
    def __init__(self, params: Params | None = None, *, model=None, tokenizer=None) -> None:
        self.params = params or load_params()
        self.cfg = self.params.presenter
        model_cfg = self.params.model_for(self.cfg.base_model)
        self.tokenizer = tokenizer or load_tokenizer(model_cfg)
        self.model = model or load_base_model(model_cfg)
        self.model.eval()

    def _generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = out[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion[0], skip_special_tokens=True)

    def present(self, question: str, columns: Sequence[str], rows: Sequence[Any]) -> Presentation:
        messages = build_messages(
            question,
            columns,
            rows,
            max_rows=self.cfg.max_preview_rows,
            max_cols=self.cfg.max_preview_cols,
        )
        text = self._generate(messages)
        return parse_output(text, columns, allowed_types=self.cfg.chart_types)
