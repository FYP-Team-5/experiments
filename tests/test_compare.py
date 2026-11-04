import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from src.gfv.compare import (compare_feedback, compare_grading, feedback_systems, grading_arrays,
                             grading_metrics, grading_systems, load_runs, mcnemar, quadratic_kappa,
                             run_name, summaries, write_report)
from src.gfv.data import load_jsonl, question_splits
from src.gfv.metrics import METRIC_COLUMNS, calculate_metrics, evaluate_records, load_records
from src.gfv.pipeline import run_pipeline
from tests.test_pipeline import FakeLLM, make_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_RECORDS = os.path.join(
    ROOT, "notebooks", "multi_agent_vote", "multi_agent_results",
    "multi_agent_rubrics_nf4_v1_do_sample_true__qwen3_5_9b__non-thinking.jsonl",
)


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_quadratic_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y_true = rng.integers(0, 6, 40)
        y_pred = np.clip(y_true + rng.integers(-2, 3, 40), 0, 7)
        assert quadratic_kappa(y_true, y_pred) == pytest.approx(
            cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    gapped = (np.array([0, 3, 3, 5]), np.array([0, 3, 5, 5]))
    assert quadratic_kappa(*gapped) == pytest.approx(cohen_kappa_score(*gapped, weights="quadratic"))


def test_fast_metrics_match_calculate_metrics():
    samples = load_jsonl("data/processed/data.jsonl")
    records = [{**r, "question": samples[i].question} for i, r in load_records(NOTEBOOK_RECORDS).items()]
    expected = calculate_metrics(records)
    actual = grading_metrics(grading_arrays(records), np.arange(len(records)))
    for metric, value in actual.items():
        assert value == pytest.approx(expected[metric]), metric


def test_mcnemar():
    a = np.array([True, True, True, False, False])
    b = np.array([True, False, False, False, True])
    assert mcnemar(a, b)[:2] == (2, 1)
    assert mcnemar(a, a) == (0, 0, 1.0)


def test_run_name():
    assert run_name(os.path.join("outputs", "comparison", "cmp_grading_base")) == "cmp_grading_base"
    assert run_name(os.path.join("outputs", "cmp_grading_base", "20261103_120000")) == "cmp_grading_base"


def test_compare_pipeline_runs(tmp_path):
    grading_dir, feedback_dir = tmp_path / "cmp_grading_fake", tmp_path / "cmp_feedback_fake"
    run_pipeline(make_config(feedback={"enabled": False}, verification={"enabled": False}),
                 str(grading_dir), llm_factory=FakeLLM)
    run_pipeline(make_config(voting={"enabled": False}), str(feedback_dir), llm_factory=FakeLLM)

    samples = load_jsonl("data/processed/data_small.jsonl")
    seeds = {seed: [s.index for s in question_splits(samples, seed)["test"]] for seed in (42, 123)}
    runs = load_runs([str(grading_dir), str(feedback_dir)])

    assert list(grading_systems(runs)) == ["cmp_grading_fake:single", "cmp_grading_fake:vote"]
    grading = compare_grading(grading_systems(runs), seeds, samples, n_boot=50)
    assert set(grading.metric) == {"exact_match_accuracy", "f1", "final_score_mae", "final_score_qwk"}

    assert list(feedback_systems(runs)) == ["cmp_feedback_fake:initial", "cmp_feedback_fake:final"]
    feedback = compare_feedback(feedback_systems(runs), seeds, samples, n_boot=50)
    # The fake verifier rejects every first draft and accepts every rewrite.
    assert (feedback.a_value == 0).all() and (feedback.b_value == 1).all()

    summary, agent_stats, metrics_by_seed = summaries(runs, seeds, samples)
    assert set(summary.run) == {"cmp_grading_fake", "cmp_feedback_fake"}
    write_report(str(tmp_path / "report"), summary, agent_stats, grading, feedback, 50, metrics_by_seed)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "## Grading: paired differences" in report and "## Feedback: verifier pass rate" in report
    assert pd.read_csv(tmp_path / "report" / "grading_paired.csv").shape[0] == len(grading)


def test_unreadable_verdict_is_a_fail_in_the_comparison():
    unreadable = {"valid": False, "is_consistent": None}
    runs = {"run": {
        0: {"verifier_initial": unreadable, "verifier": {"valid": True, "is_consistent": True}},
        1: {"verifier_initial": None, "verifier": None},
    }}
    systems = feedback_systems(runs)
    assert systems["run:initial"] == {0: False, 1: None}
    assert systems["run:final"] == {0: True, 1: None}


def test_rescoring_reproduces_the_notebook_metrics():
    """The 12 metrics the multi-agent notebook reports are unchanged by following the Rubrics notebooks."""
    samples = load_jsonl("data/processed/data.jsonl")
    seeds = {seed: [s.index for s in question_splits(samples, seed)["test"]] for seed in (42, 123, 2026)}
    metrics_by_seed, _, _ = evaluate_records(load_records(NOTEBOOK_RECORDS), seeds, samples)
    expected = pd.read_csv(NOTEBOOK_RECORDS.replace("__qwen3_5_9b__non-thinking.jsonl", "__metrics_by_seed.csv"))
    expected["system"] = expected.model.map(lambda m: "single" if "grader only" in m else "vote")
    shared = ["examples", "invalid_output_rate", "exact_match_accuracy", "per_criterion_accuracy", "precision",
              "recall", "f1", "final_score_exact_accuracy", "final_score_within_one_accuracy", "final_score_mae",
              "final_score_rmse", "final_score_pearson", "final_score_qwk"]
    merged = metrics_by_seed.merge(expected, on=["split_seed", "system"], suffixes=("", "_notebook"))
    assert len(merged) == 6
    for metric in shared:
        assert np.allclose(merged[metric], merged[metric + "_notebook"]), metric
    assert list(metrics_by_seed.columns[2:2 + len(METRIC_COLUMNS)]) == METRIC_COLUMNS


def test_rubrics_metrics_on_hand_made_records():
    def record(question, truth, prediction, raw):
        valid = prediction is not None
        return {"question": question, "ground_truth": truth, "prediction": prediction, "valid": valid,
                "raw_output": raw, "true_final_score": float(sum(truth)),
                "predicted_final_score": float(sum(prediction)) if valid else None}

    records = [
        record("Q1", [1, 0], [1, 0], "[1, 0]"),
        record("Q1", [1, 1], [1, 0], "Marks: [1, 0]"),   # valid, but found inside extra text
        record("Q2", [0, 1], None, "no idea"),           # invalid: fails both criteria
    ]
    metrics = calculate_metrics(records)
    assert metrics["examples"] == 3 and metrics["valid_examples"] == 2
    assert metrics["recovered_output_rate"] == pytest.approx(1 / 3)
    assert metrics["question_count"] == 2
    # Q1: 1 of 2 exact, 3 of 4 criteria right, F1 = 2*2/(2*2+0+1); Q2: all wrong, F1 = 0.
    assert metrics["question_macro_exact_match_accuracy"] == pytest.approx((0.5 + 0) / 2)
    assert metrics["question_macro_per_criterion_accuracy"] == pytest.approx((0.75 + 0) / 2)
    assert metrics["question_macro_criterion_f1"] == pytest.approx((0.8 + 0) / 2)
