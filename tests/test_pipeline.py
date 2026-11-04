"""Runs the whole pipeline with a fake model, so no GPU or download is needed.

    python -m pytest tests
"""
import glob
import json
import os

import pandas as pd
import pytest

from src.gfv.pipeline import build_stages, run_pipeline
from src.gfv.stages.voting import resolve_grades

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


class OutOfMemoryError(Exception):
    """Same class name as torch.OutOfMemoryError, which is all the pipeline checks."""


class FakeLLM:
    """Greedy grades mark every criterion met; sampled grades (the evaluator) mark the first one not met.
    The verifier rejects feedback until it has been revised once."""

    def __init__(self, cfg, fail_batches_larger_than=None):
        self.cfg = cfg
        self.fail_batches_larger_than = fail_batches_larger_than
        with open("prompts/grading/grading_v1.txt", encoding="utf-8") as handle:
            self.grading_system = handle.read().strip()

    def generate_batch(self, message_lists, decoding, enable_thinking=False):
        assert enable_thinking is False
        if self.fail_batches_larger_than and len(message_lists) > self.fail_batches_larger_than:
            raise OutOfMemoryError()
        replies = []
        for messages in message_lists:
            system = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user = messages[-1]["content"]
            if system == self.grading_system:
                marks = [1] * len(user.split("Criteria:\n", 1)[1].splitlines())
                if decoding.get("do_sample"):
                    marks[0] = 0
                replies.append(json.dumps(marks))
            elif "[GENERATED FEEDBACK START]" in user:
                ok = "Revised feedback" in user
                verdict = {
                    "is_consistent": ok,
                    "confidence_score": 0.9,
                    "detected_issues": [] if ok else [
                        {"issue_type": "HALLUCINATED_EVIDENCE", "severity": "HIGH", "description": "Made up."}
                    ],
                    "verification_summary": "Checked.",
                }
                replies.append("```json\n" + json.dumps(verdict) + "\n```")
            elif "rejected your previous feedback" in user:
                assert "HALLUCINATED_EVIDENCE: Made up." in user
                replies.append("Revised feedback.")
            else:
                replies.append("First feedback.")
        return replies, 0.01 * len(message_lists)


def make_config(**stage_overrides):
    stages = {
        "grading": {"system_prompt_path": "prompts/grading/grading_v1.txt",
                    "decoding": {"do_sample": False, "max_new_tokens": 128}},
        "voting": {"enabled": True, "decoding": {"do_sample": True, "temperature": 0.7}},
        "feedback": {"enabled": True,
                     "system_prompt_path": "prompts/feedback/key_points_system_v1.txt",
                     "template_path": "prompts/feedback/key_points_v1.txt"},
        "verification": {"enabled": True, "max_regenerations": 2,
                         "template_path": "prompts/verification/consistency_v2.txt"},
    }
    for name, override in stage_overrides.items():
        stages[name] = {**stages[name], **override}
    base = {"model": {"base_model_id": "fake/base"}}
    return {
        "experiment": {"seeds": [42, 123], "batch_size": 4, "enable_thinking": False},
        "data": {"input_path": "data/processed/data_small.jsonl"},
        "models": {"grading": {"model": {"base_model_id": "fake/base", "adapter_id": "fake/grader"}},
                   "feedback": base, "verification": base},
        "stages": stages,
    }


def read_records(output_dir):
    with open(os.path.join(output_dir, "records.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_full_pipeline_records_and_metrics(tmp_path):
    loaded = []

    def factory(cfg):
        loaded.append(cfg)
        return FakeLLM(cfg)

    run_pipeline(make_config(), str(tmp_path), llm_factory=factory)

    # Grading adapter + one shared base model for feedback and verification.
    assert len(loaded) == 2
    records = read_records(tmp_path)
    assert records and len({r["dataset_index"] for r in records}) == len(records)
    for record in records:
        assert record["grader_prediction"] == [1] * len(record["ground_truth"])
        assert record["resolution"] == "majority" and record["regraded"]
        assert record["prediction"][0] == 0 and record["final_changed"]
        assert record["feedback_initial"] == "First feedback."
        assert record["feedback"] == "Revised feedback."
        assert record["feedback_revisions"] == 1
        assert record["verifier_initial"]["is_consistent"] is False
        assert record["verifier"]["is_consistent"] is True
        assert len(record["verifier_attempts"]) == 2
        assert record["generation_seconds"] == pytest.approx(sum(record["agent_seconds"].values()))

    metrics = pd.read_csv(tmp_path / "metrics_by_seed.csv")
    assert set(metrics["system"]) == {"single", "vote"}
    assert set(metrics["split_seed"]) == {42, 123}
    stats = pd.read_csv(tmp_path / "agent_stats_by_seed.csv", index_col=0)
    assert stats.loc["mean", "verifier_pass_rate_initial"] == 0
    assert stats.loc["mean", "verifier_pass_rate_final"] == 1
    assert stats.loc["mean", "verifier_unreadable_rate_final"] == 0


def test_resume_skips_completed_rows(tmp_path):
    run_pipeline(make_config(), str(tmp_path), llm_factory=FakeLLM)
    first = read_records(tmp_path)

    def must_not_load(cfg):
        raise AssertionError("A finished run should not load any model.")

    run_pipeline(make_config(), str(tmp_path), llm_factory=must_not_load)
    assert read_records(tmp_path) == first


def test_out_of_memory_halves_the_batch(tmp_path):
    run_pipeline(make_config(), str(tmp_path),
                 llm_factory=lambda cfg: FakeLLM(cfg, fail_batches_larger_than=1))
    assert read_records(tmp_path)


def test_stages_can_be_switched_off(tmp_path):
    config = make_config(voting={"enabled": False}, feedback={"enabled": False},
                         verification={"enabled": False})
    run_pipeline(config, str(tmp_path), llm_factory=FakeLLM)
    record = read_records(tmp_path)[0]
    assert "resolution" not in record and "feedback" not in record
    assert record["prediction"] == record["grader_prediction"]
    metrics = pd.read_csv(tmp_path / "metrics_by_seed.csv")
    assert set(metrics["system"]) == {"single"}


def test_max_regenerations_zero_only_audits(tmp_path):
    run_pipeline(make_config(verification={"max_regenerations": 0}), str(tmp_path), llm_factory=FakeLLM)
    record = read_records(tmp_path)[0]
    assert record["feedback"] == "First feedback." and record["feedback_revisions"] == 0
    assert record["verifier"]["is_consistent"] is False


class TruncatedVerifierLLM(FakeLLM):
    """The verifier reasons in prose and runs out of tokens before the JSON."""

    def generate_batch(self, message_lists, decoding, enable_thinking=False):
        replies, seconds = super().generate_batch(message_lists, decoding, enable_thinking)
        return [
            "Let me check the rubric first. The student says" if "[GENERATED FEEDBACK START]" in m[-1]["content"]
            else reply
            for m, reply in zip(message_lists, replies)
        ], seconds


def test_unreadable_verdict_counts_as_fail(tmp_path):
    run_pipeline(make_config(), str(tmp_path), llm_factory=TruncatedVerifierLLM)
    record = read_records(tmp_path)[0]
    # No notes to revise with, so the feedback is kept as it was.
    assert record["verifier"]["valid"] is False and record["feedback_revisions"] == 0
    stats = pd.read_csv(tmp_path / "agent_stats_by_seed.csv", index_col=0)
    assert stats.loc["mean", "verifier_pass_rate_final"] == 0
    assert stats.loc["mean", "verifier_unreadable_rate_initial"] == 1


def test_resolve_grades():
    ok = lambda scores: {"valid": True, "scores": scores, "error": None}
    bad = {"valid": False, "scores": None, "error": "invalid"}
    assert resolve_grades(ok([1, 0]), ok([1, 0]))["resolution"] == "agree"
    assert resolve_grades(ok([1, 0, 1]), ok([0, 0, 1]), ok([1, 1, 0]))["scores"] == [1, 0, 1]
    assert resolve_grades(bad, ok([1, 1]), ok([1, 0]))["scores"] == [1, 0]
    assert resolve_grades(bad, bad, bad)["valid"] is False


@pytest.mark.parametrize("config_path", sorted(
    path for path in glob.glob(os.path.join(ROOT, "configs", "experiments", "*.yaml")) if os.path.getsize(path)
))
def test_experiment_configs_build(config_path):
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from scripts.run_stage import load_and_merge_config

    config = OmegaConf.to_container(load_and_merge_config(os.path.relpath(config_path, ROOT)), resolve=True)
    assert os.path.exists(config["data"]["input_path"])
    stages = build_stages(config, FakeLLM)
    assert "grading" in stages
    for stage_cfg in config["stages"].values():
        for key in ("system_prompt_path", "template_path"):
            if stage_cfg.get(key):
                assert os.path.exists(stage_cfg[key]), stage_cfg[key]
