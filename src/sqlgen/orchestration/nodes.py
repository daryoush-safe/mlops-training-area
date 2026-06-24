from __future__ import annotations

import gc
import logging
import os
import threading
from contextlib import contextmanager
import sqlglot

import torch

from sqlgen.config import ModelConfig, Params, load_params
from sqlgen.data.io import read_json
from sqlgen.data.schema import DBSchema, serialize_schema
from sqlgen.orchestration.state import InferenceState
from sqlgen.prompts import pruner as pruner_prompt
from sqlgen.prompts import sqlgen as sqlgen_prompt
from sqlgen.training.model import load_base_model, load_tokenizer
from sqlgen.data.prune import ExtractionError, extract_used_schema

log = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


class InferencePipeline:

    def __init__(self, params: Params | None = None) -> None:
        self.params = params or load_params()
        self.max_retries = self.params.inference.max_retries
        
        self.low_vram = _env_flag("SQLGEN_LOW_VRAM", self.params.inference.low_vram)
        
        self._generate_lock = threading.Lock()

        self.schemas: dict[str, DBSchema] = {
            db_id: DBSchema.model_validate(raw)
            for db_id, raw in read_json(self.params.paths.interim_dir / "schemas.json").items()
        }

        self.pruner_tokenizer = load_tokenizer(self.params.models.pruner)
        self.sqlgen_tokenizer = load_tokenizer(self.params.models.sqlgen)

        self.pruner_model = None if self.low_vram else self._load_model(self.params.models.pruner)
        self.sqlgen_model = None if self.low_vram else self._load_model(self.params.models.sqlgen)

    @staticmethod
    def _load_model(cfg: ModelConfig):
        model = load_base_model(cfg)
        model.eval()
        return model

    def is_ready(self) -> bool:
        """Cheap readiness check, suitable for a future k8s readiness probe."""
        return bool(self.schemas) and self.pruner_tokenizer is not None and self.sqlgen_tokenizer is not None

    @contextmanager
    def _model(self, which: str):
        """Yields the requested model, loading it on demand in low_vram mode.

        Outside low_vram mode this just yields the already-resident model. In
        low_vram mode it loads fresh weights on every call and frees them again
        on exit, so pruner and sqlgen are never both on the GPU simultaneously.
        """
        cached = self.pruner_model if which == "pruner" else self.sqlgen_model
        if cached is not None:
            yield cached
            return

        cfg = self.params.models.pruner if which == "pruner" else self.params.models.sqlgen
        log.info("low_vram: loading %s model on demand", which)
        model = self._load_model(cfg)
        try:
            yield model
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate(self, model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with self._generate_lock, torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = out[:, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(completion[0], skip_special_tokens=True)

    # --- graph nodes -----------------------------------------------------

    def parse_request(self, state: InferenceState) -> InferenceState:
        if not state.get("question", "").strip() or not state.get("db_id", "").strip():
            return {"status": "failed", "error": "question and db_id are required"}
        return {"attempts": 0, "status": "running", "error": None}

    def load_schema(self, state: InferenceState) -> InferenceState:
        db_schema = self.schemas.get(state["db_id"])
        if db_schema is None:
            return {"status": "failed", "error": f"unknown db_id {state['db_id']!r}"}
        schema_cfg = self.params.serialization
        schema_text = serialize_schema(
            db_schema, style=schema_cfg.style, include_types=schema_cfg.include_types
        )
        return {"schema": schema_text}

    def prune_schema(self, state: InferenceState) -> InferenceState:
        messages = pruner_prompt.build_messages(state["question"], state["schema"])
        try:
            with self._model("pruner") as model:
                text = self._generate(
                    model,
                    self.pruner_tokenizer,
                    messages,
                    max_new_tokens=self.params.inference.prune_max_new_tokens,
                )
        except RuntimeError as e:
            log.exception("pruner generation failed for db_id=%s", state["db_id"])
            return {"status": "failed", "error": f"schema pruning generation failed: {e}"}

        try:
            _tables, columns = pruner_prompt.parse_output(text)
        except pruner_prompt.OutputParseError as e:
            return {"status": "failed", "error": f"schema pruning failed to parse: {e}"}

        db_schema = self.schemas[state["db_id"]]
        used_columns = {t.lower(): {c.lower() for c in cols} for t, cols in columns.items()}
        prune_cfg = self.params.prune
        pruned = db_schema.prune(
            used_columns,
            include_primary_keys=prune_cfg.include_primary_keys,
            include_foreign_keys=prune_cfg.include_foreign_keys,
        )
        schema_cfg = self.params.serialization
        pruned_text = serialize_schema(
            pruned, style=schema_cfg.style, include_types=schema_cfg.include_types
        )
        return {"pruned_schema": pruned_text}

    def generate_sql(self, state: InferenceState) -> InferenceState:
        messages = sqlgen_prompt.build_messages(
            state["question"],
            state["pruned_schema"],
            prev_sql=state.get("sql"),
            error=state.get("error"),
        )
        try:
            with self._model("sqlgen") as model:
                text = self._generate(
                    model,
                    self.sqlgen_tokenizer,
                    messages,
                    max_new_tokens=self.params.inference.sql_max_new_tokens,
                )
        except RuntimeError as e:
            log.exception("sqlgen generation failed for db_id=%s", state["db_id"])
            return {"sql": None, "error": f"sql generation failed: {e}"}

        try:
            sql = sqlgen_prompt.extract_sql(text)
        except sqlgen_prompt.OutputParseError as e:
            return {"sql": None, "error": f"sql generation failed to parse: {e}"}
        return {"sql": sql, "error": None}

    def validate_sql(self, state: InferenceState) -> InferenceState:
        sql = state.get("sql")
        if not sql:
            return {"error": state.get("error") or "no SQL was generated"}

        dialect = self.params.prune.dialect
        try:
            sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            return {"error": f"SQL failed to parse: {e}"}

        # No live DB to execute against, so validate against the schema instead:
        # every table/column the query touches must resolve in the full schema
        # (not just the pruned one -- the model may legitimately need a column
        # the pruner dropped).
        db_schema = self.schemas[state["db_id"]]
        try:
            extract_used_schema(sql, db_schema, dialect=dialect, enable_fallback=False)
        except ExtractionError as e:
            return {"error": f"SQL references tables/columns not in schema: {e}"}

        return {"error": None, "status": "success"}

    def retry_or_finish(self, state: InferenceState) -> InferenceState:
        if state.get("error") is None:
            return {"status": "success"}
        attempts = state.get("attempts", 0) + 1
        if attempts < self.max_retries:
            log.info(
                "retrying sql generation for db_id=%s attempt=%d/%d: %s",
                state["db_id"],
                attempts,
                self.max_retries,
                state["error"],
            )
            return {"attempts": attempts, "status": "retry"}
        return {"attempts": attempts, "status": "failed"}
