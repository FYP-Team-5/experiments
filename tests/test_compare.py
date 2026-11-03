import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from src.gfv.compare import (compare_feedback, compare_grading, feedback_systems, grading_arrays,
                             grading_metrics, grading_systems, load_runs, mcnemar, quadratic_kappa,
                             run_name, summaries, write_report)
from src.gfv.data import load_jsonl, question_splits
from src.gfv.metrics import calculate_metrics, load_records
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
    records = list(load_records(NOTEBOOK_RECORDS).values())
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

    summary, agent_stats = summaries(runs, seeds)
    assert set(summary.run) == {"cmp_grading_fake", "cmp_feedback_fake"}
    write_report(str(tmp_path / "report"), summary, agent_stats, grading, feedback, 50)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "## Grading: paired differences" in report and "## Feedback: verifier pass rate" in report
    assert pd.read_csv(tmp_path / "report" / "grading_paired.csv").shape[0] == len(grading)
