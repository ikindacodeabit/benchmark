# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run any supported task/memory-budget matrix with one model load."""

import argparse

from evaluate import EvaluationConfig, EvaluationRunner, build_config
from matrix_constants import MATRIX_PROFILES, MatrixProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="evaluate_config.yaml")
    parser.add_argument("--profile", choices=sorted(MATRIX_PROFILES), required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-memory-budgets",
        type=int,
        default=None,
        help="Run only the first N budgets; the baseline is still included.",
    )
    selection.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the profile's baseline and no budgeted configurations.",
    )
    selection.add_argument(
        "--configuration-id",
        type=int,
        default=None,
        help="Run baseline ID 0 or one budget ID 1..N for a Slurm array.",
    )
    selection.add_argument(
        "--memory-budget",
        nargs=2,
        metavar=("VALUE", "UNIT"),
        help="Run one budget from the selected profile without its baseline.",
    )
    return parser.parse_args()


def validate_worker(worker_id: int, worker_count: int) -> None:
    if worker_count < 1:
        raise ValueError("worker-count must be positive")
    if not 0 <= worker_id < worker_count:
        raise ValueError("worker-id must be between 0 and worker-count - 1")


def validate_tasks(config: EvaluationConfig, profile: MatrixProfile) -> list[str]:
    if config.dataset != profile.dataset:
        raise ValueError(
            f"Profile expects dataset={profile.dataset!r}, got {config.dataset!r}"
        )
    if not isinstance(config.data_dir, list) or not config.data_dir:
        raise ValueError("Matrix evaluation requires data_dir to be a non-empty task list")

    tasks = config.data_dir
    if profile.task_suffix is not None:
        invalid_tasks = [task for task in tasks if not task.endswith(profile.task_suffix)]
        if invalid_tasks:
            raise ValueError(
                f"Profile requires tasks ending in {profile.task_suffix!r}; "
                f"invalid tasks: {invalid_tasks}"
            )
    if profile.exact_tasks is not None and tuple(tasks) != profile.exact_tasks:
        raise ValueError(
            f"Profile requires data_dir={list(profile.exact_tasks)!r}, got {tasks!r}"
        )
    if profile.required_fraction is not None and config.fraction != profile.required_fraction:
        raise ValueError(
            f"Profile requires fraction={profile.required_fraction}, got {config.fraction}"
        )
    if profile.required_press_name is not None and config.press_name != profile.required_press_name:
        raise ValueError(
            f"Profile requires press_name={profile.required_press_name!r}, "
            f"got {config.press_name!r}"
        )
    if (
        profile.required_model_basename is not None
        and config.model.rstrip("/").split("/")[-1]
        != profile.required_model_basename
    ):
        raise ValueError(
            f"Profile requires model basename={profile.required_model_basename!r}, "
            f"got {config.model!r}"
        )
    if profile.require_non_quantized and (config.int8 or config.int4 or config.fp8):
        raise ValueError("Profile requires a non-quantized model configuration")
    if profile.validate_native_rope:
        expected_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        if config.model_kwargs.get("rope_scaling", "missing") is not None:
            raise ValueError("Profile requires rope_scaling: null")
        if config.model_kwargs.get("rope_parameters") != expected_parameters:
            raise ValueError(f"Profile requires rope_parameters={expected_parameters!r}")
    if profile.yarn_factor is not None:
        expected_yarn = {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": profile.yarn_factor,
            "original_max_position_embeddings": 32768,
        }
        if config.model_kwargs.get("rope_scaling") != expected_yarn:
            raise ValueError(f"Profile requires rope_scaling={expected_yarn!r}")
    if (
        profile.expected_max_position_embeddings is not None
        and config.model_kwargs.get("max_position_embeddings")
        != profile.expected_max_position_embeddings
    ):
        raise ValueError(
            "Profile requires max_position_embeddings="
            f"{profile.expected_max_position_embeddings}"
        )
    return tasks


def select_configurations(
    args: argparse.Namespace, profile: MatrixProfile
) -> tuple[list[tuple[float, str]], bool]:
    """Select profile budgets and whether its baseline should run."""
    budgets = list(profile.memory_budgets)
    if args.baseline_only:
        return [], True
    if args.configuration_id is not None:
        if not 0 <= args.configuration_id <= len(budgets):
            raise ValueError(
                f"configuration-id must be between 0 and {len(budgets)}"
            )
        if args.configuration_id == 0:
            return [], True
        return [budgets[args.configuration_id - 1]], False
    if args.memory_budget is not None:
        value_text, unit = args.memory_budget
        try:
            selected_budget = (float(value_text), unit.upper())
        except ValueError as error:
            raise ValueError(f"Invalid memory budget value: {value_text!r}") from error
        if selected_budget not in budgets:
            raise ValueError(
                f"Budget {selected_budget!r} is not defined for this profile: {budgets}"
            )
        return [selected_budget], False
    if args.max_memory_budgets is not None:
        return budgets[: args.max_memory_budgets], True
    return budgets, True


def main() -> None:
    args = parse_args()
    validate_worker(args.worker_id, args.worker_count)
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("max-tasks must be positive")
    if args.max_memory_budgets is not None and args.max_memory_budgets < 1:
        raise ValueError("max-memory-budgets must be positive")

    config = build_config(args.config_file, fraction=args.fraction)

    profile = MATRIX_PROFILES[args.profile]
    tasks = validate_tasks(config, profile)
    assigned_tasks = tasks[args.worker_id :: args.worker_count]
    if args.max_tasks is not None:
        assigned_tasks = assigned_tasks[: args.max_tasks]
    if not assigned_tasks:
        raise ValueError(
            f"Worker {args.worker_id}/{args.worker_count} was assigned no tasks"
        )

    memory_budgets, include_baseline = select_configurations(args, profile)

    print(
        f"Profile {args.profile}; worker {args.worker_id}/{args.worker_count}; "
        f"tasks: {', '.join(assigned_tasks)}; budgets: {memory_budgets}"
    )
    runner = EvaluationRunner(config)
    runner.run_memory_budget_matrix(
        tasks=assigned_tasks,
        memory_budgets=memory_budgets,
        baseline_compression_ratio=profile.baseline_compression_ratio,
        include_baseline=include_baseline,
        baseline_press_name=profile.baseline_press_name,
    )


if __name__ == "__main__":
    main()
