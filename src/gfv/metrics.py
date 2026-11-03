"""Grading metrics, per seed and as mean ± std across seeds.

`calculate_metrics` is the function from the qlora notebooks
(notebooks/multi_agent_vote/), so records from this pipeline and from those
notebooks are scored identically. Both write the same record fields:
`dataset_index`, `ground_truth`, `prediction`, `valid`,
`true_final_score`, `predicted_final_score` and `generation_seconds`.
"""
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

METRIC_COLUMNS = [
    "invalid_output_rate",
    "exact_match_accuracy",
    "per_criterion_accuracy",
    "precision",
    "recall",
    "f1",
    "final_score_exact_accuracy",
    "final_score_within_one_accuracy",
    "final_score_mae",
    "final_score_rmse",
    "final_score_pearson",
    "final_score_qwk",
]


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Invalid outputs count as failures for exact-match and final-score accuracy.
    Criterion-level metrics, MAE, RMSE, Pearson and QWK use valid outputs only."""
    criterion_true, criterion_pred = [], []
    exact_correct = invalid_count = final_score_exact = final_score_within_one = 0
    final_score_errors, y_true_scores, y_pred_scores = [], [], []

    for record in records:
        true_scores = record["ground_truth"]
        pred_scores = record["prediction"] if record["valid"] else None
        exact_correct += int(record["valid"] and pred_scores == true_scores)
        invalid_count += int(not record["valid"])

        if record["valid"]:
            criterion_true.extend(true_scores)
            criterion_pred.extend(pred_scores)
            error = abs(record["predicted_final_score"] - record["true_final_score"])
            final_score_errors.append(error)
            final_score_exact += int(error == 0)
            final_score_within_one += int(error <= 1)
            y_true_scores.append(record["true_final_score"])
            y_pred_scores.append(record["predicted_final_score"])

    total = len(records)
    elapsed_seconds = sum(float(record.get("generation_seconds", 0.0)) for record in records)

    pearson_corr = float("nan")
    if len(y_true_scores) > 1:
        corr_matrix = np.corrcoef(y_true_scores, y_pred_scores)
        if not np.isnan(corr_matrix[0, 1]):
            pearson_corr = float(corr_matrix[0, 1])

    qwk = float("nan")
    if y_true_scores:
        try:
            qwk = float(cohen_kappa_score(
                [int(round(s)) for s in y_true_scores],
                [int(round(s)) for s in y_pred_scores],
                weights="quadratic",
            ))
        except Exception:
            pass

    return {
        "examples": total,
        "invalid_output_rate": invalid_count / total if total else 0.0,
        "exact_match_accuracy": exact_correct / total if total else 0.0,
        "per_criterion_accuracy": accuracy_score(criterion_true, criterion_pred) if criterion_true else 0.0,
        "precision": precision_score(criterion_true, criterion_pred, zero_division=0) if criterion_true else 0.0,
        "recall": recall_score(criterion_true, criterion_pred, zero_division=0) if criterion_true else 0.0,
        "f1": f1_score(criterion_true, criterion_pred, zero_division=0) if criterion_true else 0.0,
        "final_score_exact_accuracy": final_score_exact / total if total else 0.0,
        "final_score_within_one_accuracy": final_score_within_one / total if total else 0.0,
        "final_score_mae": float(np.mean(final_score_errors)) if final_score_errors else float("nan"),
        "final_score_rmse": float(np.sqrt(np.mean(np.square(final_score_errors)))) if final_score_errors else float("nan"),
        "final_score_pearson": pearson_corr,
        "final_score_qwk": qwk,
        "generation_minutes": elapsed_seconds / 60,
        "mean_generation_seconds": elapsed_seconds / total if total else float("nan"),
    }


def grader_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """The same record scored on the grading agent's own marks, before any vote."""
    seconds = record.get("agent_seconds", {})
    return {
        **record,
        "valid": record["grader_valid"],
        "prediction": record["grader_prediction"],
        "predicted_final_score": record["grader_final_score"],
        "generation_seconds": seconds.get("grader", record.get("generation_seconds", 0.0)),
    }


def systems_for(records: List[Dict[str, Any]]) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """'single' is the grading agent alone. 'vote' (the final grade) is added when the run voted."""
    systems = {"single": grader_view if "grader_prediction" in records[0] else (lambda record: record)}
    if "resolution" in records[0]:
        systems["vote"] = lambda record: record
    return systems


def agent_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How often each agent ran and changed something, and the time it took."""
    frame = pd.DataFrame(records)
    stats: Dict[str, Any] = {"examples": len(frame)}
    if "grader_valid" in frame:
        stats["grader_valid_rate"] = frame["grader_valid"].mean()
    if "resolution" in frame:
        stats.update({
            "evaluator_valid_rate": frame["evaluator_valid"].mean(),
            "agreement_rate": frame["agents_agreed"].mean(),
            "regrade_rate": frame["regraded"].mean(),
            "final_changed_rate": frame["final_changed"].mean(),
            **{
                f"resolution_{name}": int((frame["resolution"] == name).sum())
                for name in ("agree", "majority", "two_valid", "single_valid", "invalid")
            },
        })
    if "feedback" in frame:
        stats["feedback_coverage"] = frame["feedback"].notna().mean()
        valid = frame["feedback_valid"].dropna().astype(bool)
        stats["feedback_valid_rate"] = valid.mean() if len(valid) else float("nan")
    if "verifier_initial" in frame:
        initial = frame["verifier_initial"].dropna()
        final = frame["verifier"].dropna()
        stats.update({
            "verifier_valid_rate": initial.map(lambda v: v["valid"]).mean() if len(initial) else float("nan"),
            "verifier_pass_rate_initial": _pass_rate(initial),
            "verifier_pass_rate_final": _pass_rate(final),
            "feedback_revised_rate": (frame["feedback_revisions"].fillna(0) > 0).mean(),
            "feedback_revisions_mean": frame["feedback_revisions"].fillna(0).mean(),
        })
    stats["generation_minutes_total"] = frame["generation_seconds"].sum() / 60
    if "agent_seconds" in frame:
        agents = sorted({agent for seconds in frame["agent_seconds"] for agent in seconds})
        for agent in agents:
            stats[f"generation_minutes_{agent}"] = frame["agent_seconds"].map(lambda s: s.get(agent, 0.0)).sum() / 60
    return stats


def _pass_rate(verdicts: pd.Series) -> float:
    readable = [v["is_consistent"] for v in verdicts if v.get("valid")]
    return float(np.mean(readable)) if readable else float("nan")


def evaluate_records(records_by_index: Dict[int, Dict[str, Any]], test_indices_by_seed: Dict[int, List[int]]):
    """Scores each seed's test rows. Returns (metrics_by_seed, seed_summary, agent_stats) DataFrames."""
    metric_rows, stat_rows = [], []
    for seed, test_indices in test_indices_by_seed.items():
        missing = [index for index in test_indices if index not in records_by_index]
        if missing:
            raise ValueError(f"Seed {seed}: {len(missing)} test rows have no record yet.")
        seed_records = [records_by_index[index] for index in test_indices]
        for name, view in systems_for(seed_records).items():
            metric_rows.append({"split_seed": seed, "system": name,
                                **calculate_metrics([view(record) for record in seed_records])})
        stat_rows.append({"split_seed": seed, **agent_statistics(seed_records)})

    metrics_by_seed = pd.DataFrame(metric_rows)
    seed_summary = metrics_by_seed.groupby("system", sort=False)[METRIC_COLUMNS].agg(["mean", "std"])
    agent_stats = pd.DataFrame(stat_rows).set_index("split_seed")
    agent_stats.loc["mean"] = agent_stats.mean(numeric_only=True)
    return metrics_by_seed, seed_summary, agent_stats


def write_reports(records_by_index, test_indices_by_seed, output_dir: str):
    metrics_by_seed, seed_summary, agent_stats = evaluate_records(records_by_index, test_indices_by_seed)
    metrics_by_seed.to_csv(os.path.join(output_dir, "metrics_by_seed.csv"), index=False)
    seed_summary.to_csv(os.path.join(output_dir, "metrics_seed_summary.csv"))
    agent_stats.to_csv(os.path.join(output_dir, "agent_stats_by_seed.csv"))
    print(seed_summary.xs("mean", axis=1, level=1).T.to_string(float_format="%.4f"), flush=True)
    return metrics_by_seed, seed_summary, agent_stats


def load_records(path: str) -> Dict[int, Dict[str, Any]]:
    """Records by dataset_index. An interrupted run may repeat a row; the last copy wins."""
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete line {line_number} in {path}", flush=True)
                continue
            records[int(record["dataset_index"])] = record
    return records
