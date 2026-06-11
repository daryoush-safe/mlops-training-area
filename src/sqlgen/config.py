from __future__ import annotations

from pathlib import Path

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


class Params(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    prune: PruneConfig = Field(default_factory=PruneConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def load_params(path: str | Path = "params.yaml") -> Params:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Params.model_validate(data)
