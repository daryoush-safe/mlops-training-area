from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExampleScore:
    table_precision: float
    table_recall: float
    table_f1: float
    column_precision: float
    column_recall: float
    column_f1: float
    exact_match: bool


def _prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def qualified_columns(columns: dict[str, list[str]]) -> set[str]:
    return {f"{t}.{c}".lower() for t, cols in columns.items() for c in cols}


def score_example(
    gold_tables: list[str],
    gold_columns: dict[str, list[str]],
    pred_tables: list[str],
    pred_columns: dict[str, list[str]],
) -> ExampleScore:
    gold_t = {t.lower() for t in gold_tables}
    pred_t = {t.lower() for t in pred_tables}
    gold_c = qualified_columns(gold_columns)
    pred_c = qualified_columns(pred_columns)
    tp, tr, tf = _prf(gold_t, pred_t)
    cp, cr, cf = _prf(gold_c, pred_c)
    return ExampleScore(
        table_precision=tp,
        table_recall=tr,
        table_f1=tf,
        column_precision=cp,
        column_recall=cr,
        column_f1=cf,
        exact_match=gold_t == pred_t and gold_c == pred_c,
    )


def aggregate(scores: list[ExampleScore], n_unparseable: int = 0) -> dict[str, float]:
    n_total = len(scores) + n_unparseable
    if n_total == 0:
        raise ValueError("no examples to aggregate")

    def mean(attr: str) -> float:
        return sum(getattr(s, attr) for s in scores) / n_total

    return {
        "table_precision": mean("table_precision"),
        "table_recall": mean("table_recall"),
        "table_f1": mean("table_f1"),
        "column_precision": mean("column_precision"),
        "column_recall": mean("column_recall"),
        "column_f1": mean("column_f1"),
        "exact_match": sum(s.exact_match for s in scores) / n_total,
        "unparseable_rate": n_unparseable / n_total,
        "n_examples": float(n_total),
    }
