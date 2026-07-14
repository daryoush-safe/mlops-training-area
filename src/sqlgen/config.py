from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_zip: Path = Path("spider_data.zip")
    raw_dir: Path = Path("data/raw/spider")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed/schema_pruner")
    reports_dir: Path = Path("reports")


class SplitsConfig(BaseModel):
    train: list[str] = ["train_spider.json", "train_others.json"]
    val: list[str] = ["dev.json"]


class PruneConfig(BaseModel):
    dialect: str = "sqlite"
    include_primary_keys: bool = True
    include_foreign_keys: bool = True
    enable_fallback: bool = True


class SerializationConfig(BaseModel):
    # "compact": `table(col1, col2, ...)` | "verbose": CREATE TABLE-like block with types, PKs, FKs
    style: str = "verbose"
    include_types: bool = True


class ValidationConfig(BaseModel):
    max_parse_failure_rate: float = 0.02
    max_fallback_rate: float = 0.05
    min_examples_per_split: int = 100


class ModelConfig(BaseModel):
    base: str
    revision: str = "main"
    mirror_bucket: str = "models"
    mirror_prefix: str = "base"
    load_in_4bit: bool = True

    @property
    def mirror_key(self) -> str:
        """Object key prefix for this (base, revision) inside the bucket."""
        return f"{self.mirror_prefix}/{self.base}/{self.revision}"

    @property
    def mirror_uri(self) -> str:
        return f"s3://{self.mirror_bucket}/{self.mirror_key}"


class LoraParams(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class TrainConfig(BaseModel):
    seed: int = 42
    epochs: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True  # trade ~25% speed for much lower activation memory
    logging_steps: int = 20
    limit: int | None = None  # cap examples per split for smoke runs
    output_dir: Path = Path("models/schema_pruner")
    lora: LoraParams = Field(default_factory=LoraParams)


class MlflowConfig(BaseModel):
    experiment: str = "schema-pruner"
    registered_model: str = "schema-pruner-qwen2.5-coder-1.5b"


class EvalGatesConfig(BaseModel):
    min_table_recall: float = 0.95
    min_column_recall: float = 0.90
    max_unparseable_rate: float = 0.01


class EvalConfig(BaseModel):
    max_new_tokens: int = 256
    num_samples: int | None = None  # None = full val split
    batch_size: int = 8
    gates: EvalGatesConfig = Field(default_factory=EvalGatesConfig)


class ModelRegistry(BaseModel):
    pruner: ModelConfig = Field(
        default_factory=lambda: ModelConfig(base="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    )
    sqlgen: ModelConfig = Field(
        default_factory=lambda: ModelConfig(base="Qwen/Qwen2.5-Coder-3B-Instruct")
    )


class InferenceConfig(BaseModel):
    max_retries: int = 2
    prune_max_new_tokens: int = 256
    generator_max_new_tokens: int = 320
    low_vram: bool = False
    history_turns: int = 8


class PresenterConfig(BaseModel):
    base_model: str = "sqlgen"
    max_new_tokens: int = 320
    max_preview_rows: int = 20
    max_preview_cols: int = 12
    chart_types: list[str] = ["bar", "line", "pie", "scatter", "area", "histogram"]


class CheckpointConfig(BaseModel):
    host: str = "localhost"
    port: int = 5433
    database: str = "langgraph_checkpoints"
    sslmode: str = "disable"
    user_env: str = "CHECKPOINT_DB_USER"
    password_env: str = "CHECKPOINT_DB_PASSWORD"
    pool_max_size: int = 20

    def dsn(self) -> str:
        override = os.environ.get("CHECKPOINT_DB_URI")
        if override:
            return override
        user = os.environ.get(self.user_env, "postgres")
        password = os.environ.get(self.password_env, "")
        auth = quote(user, safe="")
        if password:
            auth = f"{auth}:{quote(password, safe='')}"
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"


class Params(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    prune: PruneConfig = Field(default_factory=PruneConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelRegistry = Field(default_factory=ModelRegistry)
    presenter: PresenterConfig = Field(default_factory=PresenterConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    def model_for(self, name: str) -> ModelConfig:
        registry = dict(self.models)
        try:
            return registry[name]
        except KeyError:
            raise KeyError(f"unknown model {name!r}; known models: {sorted(registry)}") from None


def load_params(path: str | Path = "params.yaml") -> Params:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Params.model_validate(data)
