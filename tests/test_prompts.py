"""The prompts must match what each adapter was trained on, character for character."""
import os

import pytest

from src.gfv.data import load_jsonl
from src.gfv.parsing import parse_feedback, parse_json_object, parse_prediction
from src.gfv.prompts import PromptLoader
from src.gfv.stages.feedback import FeedbackStage
from src.gfv.stages.grading import GradingStage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def samples():
    return load_jsonl(os.path.join(ROOT, "data", "processed", "data.jsonl"))


# SYSTEM_PROMPT from qlora/Rubrics/Qlora Comparison/qlora (48r,96a).ipynb (the grading adapter).
GRADING_TRAINING_SYSTEM_PROMPT = (
    "You are a strict grading assistant.\n\n"
    "Your task is to evaluate whether a student's answer fulfills each grading criterion.\n\n"
    "For each criterion:\n"
    "- Output 1 if the criterion is fulfilled.\n"
    "- Output 0 if the criterion is not fulfilled.\n\n"
    "Evaluate each criterion independently.\n\n"
    "Return ONLY a valid JSON array containing 0 or 1.\n"
    "The output array must contain exactly one value for each criterion, "
    "in the same order as the criteria provided.\n\n"
    "Do not include explanations.\n"
    "Do not include markdown.\n"
    "Do not include any text before or after the JSON array."
)

# STUDENT_SYSTEM_PROMPT and build_row_prompt from feedback/qlora_feedback_qwen3.5-9B.ipynb (the feedback adapter).
FEEDBACK_TRAINING_SYSTEM_PROMPT = (
    "You are a computer science tutor. Write concise, specific, encouraging feedback to the student on their short answer. "
    "The marker's decision for each key point is given; help the student understand it. "
    "Acknowledge what is correct, explain anything missing or incorrect, help the student improve, "
    "and never mention marks, key points or the reference answer."
)


def feedback_training_prompt(sample, scores):
    marks = [score >= maximum for score, maximum in zip(scores, sample.criteria_max_scores)]
    key_points = [
        f"{number}. {'[MET] ' if met else '[NOT MET] '}{criterion}"
        for number, (criterion, met) in enumerate(zip(sample.criteria, marks), start=1)
    ]
    return "\n\n".join([
        f"Question:\n{sample.question.strip()}",
        f"Reference answer:\n{sample.desired_answer.strip()}",
        f"Student answer:\n{sample.student_answer.strip() or '(no answer given)'}",
        "Key points:\n" + "\n".join(key_points),
        f"Key points met: {sum(marks)} of {len(sample.criteria)}",
    ])


def test_grading_prompt_matches_adapter_training():
    stage = GradingStage(None, "prompts/grading/grading_v1.txt", {})
    assert stage.system_prompt == GRADING_TRAINING_SYSTEM_PROMPT


def test_feedback_prompt_matches_adapter_training(samples):
    stage = FeedbackStage(None, "prompts/feedback/key_points_v1.txt", {},
                          "prompts/feedback/key_points_system_v1.txt")
    unstripped = [s for s in samples if s.student_answer != s.student_answer.strip()][:20]
    empty = [s for s in samples if not s.student_answer.strip()]
    multi = [s for s in samples if len(s.criteria) > 2][:20]
    for sample in unstripped + empty + multi:
        for scores in (sample.criteria_scores, [1 - x for x in sample.criteria_scores]):
            system, user = stage.messages(sample, scores)
            assert system == {"role": "system", "content": FEEDBACK_TRAINING_SYSTEM_PROMPT}
            assert user == {"role": "user", "content": feedback_training_prompt(sample, scores)}


def test_every_prompt_template_fills():
    stage_templates = [
        "prompts/feedback/answer_rubric_v1.txt",
        "prompts/feedback/answer_rubric_grade_v1.txt",
        "prompts/feedback/key_points_v1.txt",
        "prompts/feedback/graded_criteria_v1.txt",
    ]
    sample = load_jsonl("data/processed/data_small.jsonl")[0]
    for path in stage_templates:
        FeedbackStage(None, path, {}).messages(sample, sample.criteria_scores)

    from src.gfv.stages.verification import VerificationStage

    prompt = VerificationStage(None, "prompts/verification/consistency_v1.txt", {}).messages(
        sample, sample.criteria_scores, "Nice work."
    )[0]["content"]
    assert '"is_consistent": true | false' in prompt and "Nice work." in prompt


def test_format_prompt_reports_missing_values():
    with pytest.raises(KeyError, match="score"):
        PromptLoader.format_prompt("Score: {score}", rubric="r")
    assert PromptLoader.format_prompt('{"a": 1} {x}', x=2) == '{"a": 1} 2'


def test_parsers():
    assert parse_prediction("[1, 0]", 2)["scores"] == [1, 0]
    assert parse_prediction("<think>[1]</think>\n[0, 1]", 2)["scores"] == [0, 1]
    assert not parse_prediction("[1, 0, 1]", 2)["valid"]
    assert parse_feedback('{"feedback": "Good."}') == {"feedback_valid": True, "feedback": "Good."}
    assert parse_feedback("- point one\n- point two")["feedback"] == "- point one\n- point two"
    assert parse_json_object('Audit:\n```json\n{"is_consistent": true}\n```') == {"is_consistent": True}
    assert parse_json_object('{"is_consistent": false}') == {"is_consistent": False}
    assert parse_json_object("no json here") is None
