"""Metrics for exact synthetic key/value retrieval."""

import re
from collections.abc import Iterable

import pandas as pd


CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f]")
VALUE_IDENTIFIER = re.compile(r"(?<![a-z0-9])(?:v_)?([a-f0-9]{12})(?![a-z0-9])")


def _normalize(value: object) -> str:
    return CONTROL_CHARACTERS.sub("", str(value)).strip().casefold()


def _references(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [_normalize(value)]
    return [_normalize(reference) for reference in value]


def _canonical_values(value: str) -> set[str]:
    """Extract value identifiers while ignoring an optional V_ prefix."""
    return set(VALUE_IDENTIFIER.findall(value))


def _string_match(prediction: str, reference: str) -> bool:
    reference_values = _canonical_values(reference)
    if reference_values:
        return bool(reference_values.intersection(_canonical_values(prediction)))
    return reference in prediction


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Report strict exact match and answer containment percentages."""
    if len(df) == 0:
        raise ValueError("Cannot score an empty synthetic-KV dataframe")

    exact_matches = 0
    string_matches = 0
    for prediction, answer in zip(df["predicted_answer"], df["answer"]):
        normalized_prediction = _normalize(prediction)
        references = _references(answer)
        exact_matches += int(normalized_prediction in references)
        string_matches += int(
            any(_string_match(normalized_prediction, reference) for reference in references)
        )

    sample_count = len(df)
    task_name = str(df["task"].iloc[0])
    return {
        task_name: {
            "exact_match": round(100.0 * exact_matches / sample_count, 2),
            "string_match": round(100.0 * string_matches / sample_count, 2),
            "num_samples": sample_count,
        }
    }
