"""Grading metrics, per seed and as mean ± std across seeds.

`calculate_metrics` follows the qlora repo's Rubrics notebooks
(`evaluate_current_model` and `summarize_question_metrics` in
Rubrics/Multi-seed/ and Rubrics/Qlora Comparison/): the same metrics, in the
same order, with the same rules for invalid outputs. The 12 metrics shared
with the multi-agent notebooks (notebooks/multi_agent_vote/) are computed as
they compute them, so rescoring their JSONL output reproduces their
metrics_by_seed.csv.

Records need `dataset_index`, `ground_truth`, `prediction`, `valid`,
`true_final_score`, `predicted_final_score` and `raw_output`; both this
pipeline and the notebooks write them. `evaluate_records` adds each row's
`question` from the dataset.
"""
import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

# The Rubrics notebooks' column order: their 7-series metrics, then the question-level ones.
METRIC_COLUMNS = [
    "examples",
    "valid_examples",
    "recovered_output_rate",
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
    "question_count",
    "question_macro_exact_match_accuracy",
    "question_macro_per_criterion_accuracy",
    "question_macro_criterion_f1",
]


def is_recovered(raw_output: Optional[str], valid: bool) -> bool:
    """The Rubrics notebooks' `recovered`: a valid grade that had to be found inside the
    output because the whole output was not a JSON array (e.g. extra text around it)."""
    if not valid or raw_output is None:
        return False
    try:
        json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return True
    return False


def question_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """`summarize_question_metrics` from the Rubrics notebooks: every question has equal weight,
    and an invalid output fails all of its criteria."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["question"].strip()].append(record)
    per_question = []
    for answers in grouped.values():
        tp = fp = fn = criterion_correct = criterion_total = 0
        for answer in answers:
            truth = answer["ground_truth"]
            criterion_total += len(truth)
            if not answer["valid"]:
                # Missing positive decisions count as FN, missing negative decisions as FP.
                fn += sum(label == 1 for label in truth)
                fp += sum(label == 0 for label in truth)
                continue
            for actual, predicted in zip(truth, answer["prediction"]):
                criterion_correct += int(actual == predicted)
                tp += int(actual == predicted == 1)
                fp += int(actual == 0 and predicted == 1)
                fn += int(actual == 1 and predicted == 0)
        denominator = 2 * tp + fp + fn
        per_question.append({
            "exact_match_accuracy": sum(
                bool(a["valid"] and a["prediction"] == a["ground_truth"]) for a in answers
            ) / len(answers),
            "per_criterion_accuracy": criterion_correct / criterion_total,
            # A perfectly graded all-negative question has F1 = 1 rather than undefined F1.
            "criterion_f1": 2 * tp / denominator if denominator else 1.0,
        })
    count = len(per_question)

    def macro(column):
        return sum(row[column] for row in per_question) / count if count else 0.0

    return {
        "question_count": count,
        "question_macro_exact_match_accuracy": macro("exact_match_accuracy"),
        "question_macro_per_criterion_accuracy": macro("per_criterion_accuracy"),
        "question_macro_criterion_f1": macro("criterion_f1"),
    }


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Invalid outputs count as failures for exact-match and final-score accuracy.
    Criterion-level metrics, MAE, RMSE, Pearson and QWK use valid outputs only."""
    criterion_true, criterion_pred = [], []
    exact_correct = invalid_count = recovered_count = final_score_exact = final_score_within_one = 0
    final_score_errors, y_true_scores, y_pred_scores = [], [], []

    for record in records:
        true_scores = record["ground_truth"]
        pred_scores = record["prediction"] if record["valid"] else None
        exact_correct += int(record["valid"] and pred_scores == true_scores)
        invalid_count += int(not record["valid"])
        recovered_count += int(record["recovered"] if "recovered" in record
                               else is_recovered(record.get("raw_output"), record["valid"]))

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
    precision, recall, f1, _ = (
        precision_recall_fscore_support(criterion_true, criterion_pred, average="binary", zero_division=0)
        if criterion_true else (float("nan"),) * 4
    )
    if len(y_true_scores) > 1 and len(set(y_true_scores)) > 1 and len(set(y_pred_scores)) > 1:
        pearson = float(np.corrcoef(y_true_scores, y_pred_scores)[0, 1])
    else:
        pearson = float("nan")
    qwk = (float(cohen_kappa_score(y_true_scores, y_pred_scores, weights="quadratic"))
           if y_true_scores else float("nan"))

    return {
        "examples": total,
        "valid_examples": total - invalid_count,
        "recovered_output_rate": recovered_count / total if total else 0.0,
        "invalid_output_rate": invalid_count / total if total else 0.0,
        "exact_match_accuracy": exact_correct / total if total else 0.0,
        "per_criterion_accuracy": accuracy_score(criterion_true, criterion_pred) if criterion_true else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "final_score_exact_accuracy": final_score_exact / total if total else 0.0,
        "final_score_within_one_accuracy": final_score_within_one / total if total else 0.0,
        "final_score_mae": float(np.mean(final_score_errors)) if final_score_errors else float("nan"),
        "final_score_rmse": (float(np.sqrt(np.mean(np.square(final_score_errors))))
                             if final_score_errors else float("nan")),
        "final_score_pearson": pearson,
        "final_score_qwk": qwk,
        **question_metrics(records),
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
        "recovered": is_recovered(record.get("raw_output"), record["grader_valid"]),
        "generation_seconds": seconds.get("grader", record.get("generation_seconds", 0.0)),
    }


def vote_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """The final grade after the vote. It counts as recovered when any grade that went into
    the vote (grader, evaluator, re-grade) was recovered."""
    return {
        **record,
        "recovered": any(
            is_recovered(record.get(raw), bool(record.get(valid)))
            for raw, valid in (("raw_output", "grader_valid"), ("evaluator_raw_output", "evaluator_valid"),
                               ("regrade_raw_output", "regrade_valid"))
        ),
    }


def systems_for(records: List[Dict[str, Any]]) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """'single' is the grading agent alone. 'vote' (the final grade) is added when the run voted."""
    systems = {"single": grader_view if "grader_prediction" in records[0] else (lambda record: record)}
    if "resolution" in records[0]:
        systems["vote"] = vote_view
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
            "verifier_unreadable_rate_initial": _unreadable_rate(initial),
            "verifier_unreadable_rate_final": _unreadable_rate(final),
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


def verdict_passed(verdict: Optional[Dict[str, Any]]) -> Optional[bool]:
    """True only for a readable verdict that accepts the feedback. A verdict that could not be
    read counts as a fail: dropping it would favour feedback that confuses the verifier.
    None when there is no verdict (the row had no feedback)."""
    if not verdict:
        return None
    return bool(verdict.get("valid") and verdict.get("is_consistent") is True)


def _pass_rate(verdicts: pd.Series) -> float:
    passed = [verdict_passed(v) for v in verdicts if v]
    return float(np.mean(passed)) if passed else float("nan")


def _unreadable_rate(verdicts: pd.Series) -> float:
    unreadable = [not v.get("valid") for v in verdicts if v]
    return float(np.mean(unreadable)) if unreadable else float("nan")


def evaluate_records(records_by_index: Dict[int, Dict[str, Any]], test_indices_by_seed: Dict[int, List[int]],
                     samples: List[Any]):
    """Scores each seed's test rows. `samples` is the dataset (indexed by dataset_index), for
    each row's question. Returns (metrics_by_seed, seed_summary, agent_stats) DataFrames."""
    metric_rows, stat_rows = [], []
    for seed, test_indices in test_indices_by_seed.items():
        missing = [index for index in test_indices if index not in records_by_index]
        if missing:
            raise ValueError(f"Seed {seed}: {len(missing)} test rows have no record yet.")
        seed_records = [{**records_by_index[index], "question": samples[index].question}
                        for index in test_indices]
        for name, view in systems_for(seed_records).items():
            metric_rows.append({"split_seed": seed, "system": name,
                                **calculate_metrics([view(record) for record in seed_records])})
        stat_rows.append({"split_seed": seed, **agent_statistics(seed_records)})

    metrics_by_seed = pd.DataFrame(metric_rows)
    seed_summary = metrics_by_seed.groupby("system", sort=False)[METRIC_COLUMNS].agg(["mean", "std"])
    agent_stats = pd.DataFrame(stat_rows).set_index("split_seed")
    agent_stats.loc["mean"] = agent_stats.mean(numeric_only=True)
    return metrics_by_seed, seed_summary, agent_stats


def write_reports(records_by_index, test_indices_by_seed, samples, output_dir: str):
    metrics_by_seed, seed_summary, agent_stats = evaluate_records(records_by_index, test_indices_by_seed, samples)
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
