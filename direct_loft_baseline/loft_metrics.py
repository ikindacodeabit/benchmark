"""Standalone implementation of the LOFT RAG metrics."""

from __future__ import annotations

import ast
import collections
import re
import string
import unicodedata
from statistics import fmean
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFD", value).lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value, flags=re.UNICODE)
    value = "".join(character for character in value if character not in set(string.punctuation))
    return " ".join(value.split())


def extract_prediction(model_output: str, answer_prefix: str = "final answer") -> list[str]:
    def escape_inner_quotes(value: str) -> str:
        return re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9])", r"\1\'\2", value)

    lines = model_output.replace("*", "").replace("`", "").strip().split("\n")
    for line in lines:
        if "[" not in line or "]" not in line:
            continue
        candidate = line[line.find("[") : line.rfind("]") + 1].strip()
        try:
            parsed = ast.literal_eval(escape_inner_quotes(candidate))
        except Exception:
            continue
        return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]

    for line in lines:
        prefix_index = line.lower().find(answer_prefix.lower())
        if prefix_index < 0:
            continue
        candidate = line[prefix_index + len(answer_prefix) :].strip().lstrip(":").strip()
        if candidate:
            return [candidate]
    return []


def token_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    prediction_tokens = normalize_answer(prediction).split()
    common = collections.Counter(gold_tokens) & collections.Counter(prediction_tokens)
    shared = sum(common.values())
    if shared == 0:
        return 0.0
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    precision = shared / len(prediction_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def multi_value_subspan(gold: list[str], predicted: list[str]) -> float:
    scores = np.zeros((len(gold), len(predicted)))
    for gold_index, gold_item in enumerate(gold):
        for prediction_index, predicted_item in enumerate(predicted):
            if gold_item in predicted_item or predicted_item in gold_item:
                scores[gold_index, prediction_index] = 1
    row_indices, column_indices = linear_sum_assignment(-scores)
    aligned = np.zeros(len(gold))
    for row_index, column_index in zip(row_indices, column_indices):
        aligned[row_index] = scores[row_index, column_index]
    return float(all(aligned))


def score_records(records: list[dict[str, Any]], task: str) -> dict[str, float | int]:
    if not records:
        return {"num_samples": 0, "em": 0.0, "subspan_em": 0.0}

    multi_value = task.startswith("qampari") or task.startswith("quest")
    exact_scores: list[float] = []
    subspan_scores: list[float] = []
    secondary_scores: list[float] = []

    for record in records:
        gold = [normalize_answer(str(item)) for item in record.get("answers", []) if item is not None]
        predictions = extract_prediction(
            str(record.get("predicted_answer", "")),
            str(record.get("answer_prefix", "Final Answer: ")).lower(),
        )
        predicted = [normalize_answer(item) for item in predictions]
        if not gold or not predicted:
            exact_scores.append(0.0)
            subspan_scores.append(0.0)
            secondary_scores.append(0.0)
            continue

        if multi_value:
            exact_scores.append(float(set(gold) == set(predicted)))
            subspan_scores.append(multi_value_subspan(gold, predicted))
            secondary_scores.append(len(set(gold).intersection(predicted)) / len(gold))
        else:
            answer = predicted[0]
            exact_scores.append(max(float(item == answer) for item in gold))
            subspan_scores.append(max(float(item in answer) for item in gold))
            secondary_scores.append(max(token_f1(item, answer) for item in gold))

    metrics: dict[str, float | int] = {
        "em": fmean(exact_scores),
        "subspan_em": fmean(subspan_scores),
        "num_samples": len(records),
    }
    metrics["coverage" if multi_value else "f1"] = fmean(secondary_scores)
    return metrics
