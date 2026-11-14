"""Compares finished runs on the same test rows.

Every comparison is paired: both systems are scored on the same rows of one
seed's test split. For each pair and seed it reports

- the difference B - A of exact match, criterion F1, final-score MAE and QWK,
  with a 95% bootstrap interval that resamples questions (rows of one question
  are not independent);
- McNemar's exact test on exact match (or, for feedback, on the verifier's pass).

Grading systems come from runs that voted (`single` and `vote` for each).
Feedback systems come from runs that verified (`initial` = before any rewrite,
`final` = after). Feedback is judged by the pipeline's own verifier here; the
27B judge is a separate step.
"""
import itertools
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.gfv.data import Sample
from src.gfv.metrics import (METRIC_COLUMNS, agent_statistics, evaluate_records, load_records, systems_for,
                             verdict_passed)

BOOTSTRAP_METRICS = ["exact_match_accuracy", "f1", "final_score_mae", "final_score_qwk"]
TIMESTAMP = re.compile(r"^\d{8}_\d{6}$")

Records = Dict[int, Dict[str, Any]]


# --------------------------------------------------------------------------- runs

def run_name(path: str) -> str:
    """The config name: outputs/comparison/<name>/ or outputs/<name>/<timestamp>/."""
    path = os.path.normpath(path)
    name = os.path.basename(path)
    return os.path.basename(os.path.dirname(path)) if TIMESTAMP.match(name) else name


def load_runs(paths: Sequence[str]) -> Dict[str, Records]:
    runs = {}
    for path in paths:
        records = load_records(os.path.join(path, "records.jsonl"))
        if not records:
            raise FileNotFoundError(f"No records in {path}")
        name = run_name(path)
        if name in runs:
            raise ValueError(f"Two runs are both called {name}; pass only one of them.")
        runs[name] = records
    return runs


def grading_systems(runs: Dict[str, Records]) -> Dict[str, Records]:
    """'<run>:single' and '<run>:vote' for every run that voted."""
    systems = {}
    for name, records in runs.items():
        sample = next(iter(records.values()))
        if "resolution" not in sample:
            continue
        for system, view in systems_for([sample]).items():
            systems[f"{name}:{system}"] = {index: view(record) for index, record in records.items()}
    return systems


def feedback_systems(runs: Dict[str, Records]) -> Dict[str, Dict[int, Optional[bool]]]:
    """'<run>:initial' and '<run>:final' verifier verdicts for every run that verified.
    A verdict the verifier's answer could not be read for counts as a fail; it is None
    only when the row had no feedback."""
    systems = {}
    for name, records in runs.items():
        if "verifier_initial" not in next(iter(records.values())):
            continue
        systems[f"{name}:initial"] = {i: verdict_passed(r["verifier_initial"]) for i, r in records.items()}
        systems[f"{name}:final"] = {i: verdict_passed(r["verifier"]) for i, r in records.items()}
    return systems


# ------------------------------------------------------------- fast paired metrics

def quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same result as sklearn's cohen_kappa_score(weights="quadratic")."""
    if len(y_true) == 0:
        return float("nan")
    labels, inverse = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    k, n = len(labels), len(y_true)
    confusion = np.bincount(inverse[:n] * k + inverse[n:], minlength=k * k).reshape(k, k).astype(float)
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / confusion.sum()
    weights = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    denominator = (weights * expected).sum()
    return float("nan") if denominator == 0 else float(1 - (weights * confusion).sum() / denominator)


def grading_arrays(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    valid = np.array([bool(r["valid"]) for r in records])
    tp, fp, fn = [], [], []
    for r in records:
        if r["valid"]:
            pairs = list(zip(r["ground_truth"], r["prediction"]))
            tp.append(sum(t == 1 and p == 1 for t, p in pairs))
            fp.append(sum(t == 0 and p == 1 for t, p in pairs))
            fn.append(sum(t == 1 and p == 0 for t, p in pairs))
        else:
            tp.append(0), fp.append(0), fn.append(0)
    pred = np.array([r["predicted_final_score"] if r["valid"] else np.nan for r in records], dtype=float)
    true = np.array([r["true_final_score"] for r in records], dtype=float)
    return {
        "valid": valid,
        "exact": np.array([bool(r["valid"]) and r["prediction"] == r["ground_truth"] for r in records]),
        "tp": np.array(tp), "fp": np.array(fp), "fn": np.array(fn),
        "true": true, "pred": pred,
    }


def grading_metrics(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    """calculate_metrics' exact match, F1, MAE and QWK on the rows `idx` (repeats allowed)."""
    valid = idx[arrays["valid"][idx]]
    tp, fp, fn = (arrays[key][valid].sum() for key in ("tp", "fp", "fn"))
    true, pred = arrays["true"][valid], arrays["pred"][valid]
    return {
        "exact_match_accuracy": float(arrays["exact"][idx].mean()) if len(idx) else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "final_score_mae": float(np.abs(pred - true).mean()) if len(valid) else float("nan"),
        "final_score_qwk": quadratic_kappa(np.rint(true).astype(int), np.rint(pred).astype(int)),
    }


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> Tuple[int, int, float]:
    """Rows only A got right, rows only B got right, and the exact two-sided p-value."""
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((b_correct & ~a_correct).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return a_only, b_only, p


def bootstrap(groups: List[np.ndarray], stat_a: Callable, stat_b: Callable,
              metrics: List[str], n_boot: int, seed: int) -> Dict[str, Tuple[float, float]]:
    """95% percentile interval of stat_b - stat_a, resampling whole questions."""
    rng = np.random.default_rng(seed)
    diffs = {metric: [] for metric in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        a, b = stat_a(idx), stat_b(idx)
        for metric in metrics:
            diffs[metric].append(b[metric] - a[metric])
    return {
        metric: tuple(np.nanpercentile(values, [2.5, 97.5])) if np.isfinite(values).any() else (np.nan, np.nan)
        for metric, values in diffs.items()
    }


def question_groups(indices: List[int], samples: List[Sample]) -> List[np.ndarray]:
    positions: Dict[str, List[int]] = {}
    for position, index in enumerate(indices):
        positions.setdefault(samples[index].question.strip(), []).append(position)
    return [np.array(group) for group in positions.values()]


# --------------------------------------------------------------------- comparisons

def compare_grading(systems: Dict[str, Records], test_indices_by_seed: Dict[int, List[int]],
                    samples: List[Sample], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(systems.items(), 2):
        for split_seed, indices in test_indices_by_seed.items():
            arrays_a = grading_arrays([a[i] for i in indices])
            arrays_b = grading_arrays([b[i] for i in indices])
            everything = np.arange(len(indices))
            point_a, point_b = grading_metrics(arrays_a, everything), grading_metrics(arrays_b, everything)
            intervals = bootstrap(
                question_groups(indices, samples),
                lambda idx: grading_metrics(arrays_a, idx), lambda idx: grading_metrics(arrays_b, idx),
                BOOTSTRAP_METRICS, n_boot, seed,
            )
            a_only, b_only, p = mcnemar(arrays_a["exact"], arrays_b["exact"])
            for metric in BOOTSTRAP_METRICS:
                low, high = intervals[metric]
                is_exact = metric == "exact_match_accuracy"
                rows.append({
                    "a": name_a, "b": name_b, "split_seed": split_seed, "metric": metric,
                    "a_value": point_a[metric], "b_value": point_b[metric],
                    "diff_b_minus_a": point_b[metric] - point_a[metric],
                    "ci_low": low, "ci_high": high,
                    "ci_excludes_0": bool(low > 0 or high < 0),
                    "only_a_right": a_only if is_exact else None,
                    "only_b_right": b_only if is_exact else None,
                    "mcnemar_p": p if is_exact else None,
                })
    return pd.DataFrame(rows)


def compare_feedback(systems: Dict[str, Dict[int, Optional[bool]]], test_indices_by_seed: Dict[int, List[int]],
                     samples: List[Sample], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(systems.items(), 2):
        for split_seed, indices in test_indices_by_seed.items():
            # Rows without feedback in either system are left out.
            kept = [i for i in indices if a.get(i) is not None and b.get(i) is not None]
            pass_a = np.array([a[i] for i in kept], dtype=bool)
            pass_b = np.array([b[i] for i in kept], dtype=bool)

            def rate(values):
                return lambda idx: {"verifier_pass_rate": float(values[idx].mean()) if len(idx) else float("nan")}

            low, high = (np.nan, np.nan)
            if kept:
                low, high = bootstrap(question_groups(kept, samples), rate(pass_a), rate(pass_b),
                                      ["verifier_pass_rate"], n_boot, seed)["verifier_pass_rate"]
            a_only, b_only, p = mcnemar(pass_a, pass_b)
            rows.append({
                "a": name_a, "b": name_b, "split_seed": split_seed, "metric": "verifier_pass_rate",
                "paired_rows": len(kept),
                "a_value": pass_a.mean() if kept else np.nan, "b_value": pass_b.mean() if kept else np.nan,
                "diff_b_minus_a": (pass_b.mean() - pass_a.mean()) if kept else np.nan,
                "ci_low": low, "ci_high": high, "ci_excludes_0": bool(low > 0 or high < 0),
                "only_a_passed": a_only, "only_b_passed": b_only, "mcnemar_p": p,
            })
    return pd.DataFrame(rows)


def summaries(runs: Dict[str, Records], test_indices_by_seed: Dict[int, List[int]], samples: List[Sample]
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-system metric means and stds across seeds, per-run agent statistics, and every
    system's metrics per seed.

    Grading metrics come only from runs that voted, when there are any. A run without a
    vote only repeats the single grader of a grading run (the feedback runs use the
    QLoRA grader of cmp_grading_qlora), so its column would be a duplicate. Agent
    statistics cover every run."""
    voted = {name for name, records in runs.items() if "resolution" in next(iter(records.values()))}
    metric_frames, stat_frames = [], []
    for name, records in runs.items():
        metrics_by_seed, _, agent_stats = evaluate_records(records, test_indices_by_seed, samples)
        if not voted or name in voted:
            metrics_by_seed.insert(0, "run", name)
            metric_frames.append(metrics_by_seed)
        stats = agent_stats.loc[["mean"]].copy()
        stats.insert(0, "run", name)
        # Seconds per row over every row the run processed, not per seed.
        all_rows = agent_statistics(list(records.values()))
        stats["rows_processed"] = len(records)
        stats["seconds_per_row"] = all_rows["generation_minutes_total"] * 60 / len(records)
        stat_frames.append(stats.reset_index(drop=True))
    metrics = pd.concat(metric_frames)
    grouped = metrics.groupby(["run", "system"], sort=False)[METRIC_COLUMNS]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std")).reset_index()
    return summary, pd.concat(stat_frames, ignore_index=True), metrics.reset_index(drop=True)


# -------------------------------------------------------------------------- report

def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


COUNT_METRICS = {"examples", "valid_examples", "question_count"}


def metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.0f}" if metric in COUNT_METRICS else f"{value:.4f}"


def metric_tables(summary: pd.DataFrame, metrics_by_seed: pd.DataFrame) -> List[str]:
    """Every metric in the Rubrics notebooks' order, one column per system: the average across
    seeds (their Table 2), then each seed (their Table 1). With one seed, only the latter."""
    lines = []
    if metrics_by_seed.split_seed.nunique() > 1:
        systems = list(dict.fromkeys(summary.run + ":" + summary.system))
        average = pd.DataFrame({"metric": METRIC_COLUMNS})
        for system, row in zip(systems, summary.itertuples(index=False)):
            average[system] = [
                f"{metric_value(m, getattr(row, m + '_mean'))} ± {metric_value(m, getattr(row, m + '_std'))}"
                for m in METRIC_COLUMNS
            ]
        lines += ["## Grading: every metric, mean ± std across seeds", "", markdown_table(average), ""]
    labelled = metrics_by_seed.assign(column=metrics_by_seed.run + ":" + metrics_by_seed.system)
    for split_seed, rows in labelled.groupby("split_seed", sort=True):
        table = pd.DataFrame({"metric": METRIC_COLUMNS})
        for row in rows.itertuples(index=False):
            table[row.column] = [metric_value(m, getattr(row, m)) for m in METRIC_COLUMNS]
        lines += [f"## Grading: seed {split_seed}", "", markdown_table(table), ""]
    return lines[:-1]


def write_report(output_dir: str, summary: pd.DataFrame, agent_stats: pd.DataFrame,
                 grading: pd.DataFrame, feedback: pd.DataFrame, n_boot: int,
                 metrics_by_seed: pd.DataFrame):
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, "grading_summary.csv"), index=False)
    metrics_by_seed.to_csv(os.path.join(output_dir, "grading_by_seed.csv"), index=False)
    agent_stats.to_csv(os.path.join(output_dir, "agent_stats.csv"), index=False)
    grading.to_csv(os.path.join(output_dir, "grading_paired.csv"), index=False)
    feedback.to_csv(os.path.join(output_dir, "feedback_paired.csv"), index=False)

    lines = [
        "# Multi-agent comparison",
        "",
        "Each comparison is paired: both systems are scored on the same test rows of one seed. "
        "`diff` is B − A; higher is better except for MAE. The interval is a 95% bootstrap "
        f"({n_boot} resamples of whole questions); `*` marks an interval that excludes 0.",
        "",
        "The grading metrics are those of the qlora Rubrics notebooks, in their order. "
        "`recovered_output_rate` is the share of valid grades found inside extra text; for a vote, "
        "any recovered input grade counts. The `question_macro_*` metrics give every question equal "
        "weight, and an invalid output fails all of its criteria.",
        "",
        *metric_tables(summary, metrics_by_seed),
    ]
    if not grading.empty:
        table = grading.assign(
            result=grading.apply(
                lambda r: f"{r.diff_b_minus_a:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
                          + ("*" if r.ci_excludes_0 else ""), axis=1)
        ).pivot_table(index=["a", "b", "metric"], columns="split_seed", values="result", aggfunc="first")
        table.columns = [f"seed {seed}" for seed in table.columns]
        exact = grading[grading.metric == "exact_match_accuracy"]
        mcnemar_table = exact.assign(
            result=exact.apply(lambda r: f"{int(r.only_a_right)} vs {int(r.only_b_right)}, p={r.mcnemar_p:.3f}", axis=1)
        ).pivot_table(index=["a", "b"], columns="split_seed", values="result", aggfunc="first")
        mcnemar_table.columns = [f"seed {seed}" for seed in mcnemar_table.columns]
        lines += [
            "", "## Grading: paired differences", "", markdown_table(table.reset_index()),
            "", "## Grading: McNemar on exact match (rows only A got right vs only B)", "",
            markdown_table(mcnemar_table.reset_index()),
        ]
    if not feedback.empty:
        table = feedback.assign(
            result=feedback.apply(
                lambda r: f"{r.a_value:.3f} → {r.b_value:.3f} ({r.diff_b_minus_a:+.3f} "
                          f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]{'*' if r.ci_excludes_0 else ''}, "
                          f"McNemar p={r.mcnemar_p:.3f})", axis=1)
        ).pivot_table(index=["a", "b"], columns="split_seed", values="result", aggfunc="first")
        table.columns = [f"seed {seed}" for seed in table.columns]
        lines += [
            "", "## Feedback: verifier pass rate, A → B", "",
            "Judged by the pipeline's own verifier (base Qwen3.5-9B), so this is not yet an independent "
            "measure of feedback quality; score `feedback_initial` and `feedback` with the 27B judge for that. "
            "A verdict that could not be read counts as a fail; the table below gives how often that happened.",
            "", markdown_table(table.reset_index()),
        ]
    cost_columns = ["run", "rows_processed", "seconds_per_row"] + [
        c for c in agent_stats.columns if c.startswith("generation_minutes_")
    ] + [c for c in ("verifier_pass_rate_initial", "verifier_pass_rate_final",
                     "verifier_unreadable_rate_initial", "verifier_unreadable_rate_final", "feedback_revised_rate")
         if c in agent_stats.columns]
    lines += ["", "## Agents and cost (mean over seeds; minutes per seed's test split)", "",
              markdown_table(agent_stats[cost_columns])]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
