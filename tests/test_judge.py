"""src/gfv/judge.py is copied from the feedback repo's notebooks. The first tests compare it with
them when the feedback repo is next to this one (as in the project folder); the rest run the
judge's bookkeeping on made-up records and judgements."""
import ast
import json
import os
import re

import pandas as pd
import pytest

from src.gfv import judge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEEDBACK_REPO = os.path.join(ROOT, "..", "feedback")
TEACHER = os.path.join(FEEDBACK_REPO, "feedback_teacher_qwen3.5-27B.ipynb")
STUDENT = os.path.join(FEEDBACK_REPO, "qlora_feedback_qwen3.5-9B.ipynb")
DATA_CSV = os.path.join(ROOT, "data", "training_data_moh_eng_rubrics.csv")


def notebook_definitions(path, names):
    """Executes only the named top-level assignments and functions of a notebook's code cells."""
    with open(path, encoding="utf-8") as handle:
        cells = ["".join(c["source"]) for c in json.load(handle)["cells"] if c["cell_type"] == "code"]
    namespace = {"re": re, "json": json, "pd": pd, "annotations": None}
    for source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:  # cells with IPython "!" commands
            continue
        for node in tree.body:
            targets = ([node.name] if isinstance(node, ast.FunctionDef)
                       else [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign)
                       else [])
            if set(targets) & set(names):
                exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    missing = set(names) - set(namespace)
    assert not missing, f"{missing} not found in {path}"
    return namespace


needs_feedback_repo = pytest.mark.skipif(not os.path.exists(TEACHER), reason="feedback repo not next to this one")


@needs_feedback_repo
def test_judge_prompt_and_checks_match_the_teacher_notebook():
    teacher = notebook_definitions(TEACHER, ["JUDGE_SYSTEM_PROMPT", "JUDGE_KEYS", "parse_judgement",
                                             "criterion_marks", "build_row_prompt", "build_judge_messages"])
    assert judge.JUDGE_SYSTEM_PROMPT == teacher["JUDGE_SYSTEM_PROMPT"]
    assert judge.JUDGE_KEYS == teacher["JUDGE_KEYS"]

    df = judge.load_dataset(DATA_CSV)
    for index in list(range(0, len(df), 97)) + [988, 989]:
        row = df.loc[index]
        assert judge.gold_marks(row) == teacher["criterion_marks"](row)
        assert (judge.build_judge_messages(row, judge.gold_marks(row), "Some feedback.")
                == teacher["build_judge_messages"](row, "Some feedback."))

    for text in ['{"covers_missing_points": true, "respects_met_points": true, "faithful_to_student": false, '
                 '"factually_correct": true, "issues": "x"}', "no json", '{"covers_missing_points": "yes"}',
                 'Reasoning... {"covers_missing_points": true, "respects_met_points": true, '
                 '"faithful_to_student": true, "factually_correct": true, "issues": ""} done']:
        assert judge.parse_judgement(text) == teacher["parse_judgement"](text)


@needs_feedback_repo
def test_format_checks_match_the_student_notebook():
    student = notebook_definitions(STUDENT, ["GRADING_TERMS", "MARKDOWN", "NON_ENGLISH", "format_issues"])
    assert judge.GRADING_TERMS.pattern == student["GRADING_TERMS"].pattern
    assert judge.MARKDOWN.pattern == student["MARKDOWN"].pattern
    for character in ["　", "中", "鿿", "가", "힯", "a", "é", "’"]:
        assert bool(judge.NON_ENGLISH.search(character)) == bool(student["NON_ENGLISH"].search(character))
    texts = ["Short.", "word " * 250, "Your score shows you missed the reference answer here, " * 2,
             "**Bold** feedback that is long enough to pass the length check for sure, really it is.",
             "- a bullet point that is long enough to pass the length check for sure, really it is.",
             "Plain prose feedback that explains the idea clearly and is long enough to pass every check."]
    for text in texts:
        for truncated in (False, True):
            assert judge.format_issues(text, truncated) == student["format_issues"](text, truncated)


def make_dataset():
    return pd.DataFrame({
        "question": ["Q1?", "Q1?", "Q2?"],
        "desired_answer": ["A1", "A1", "A2"],
        "student_answer": ["s0", "", "s2"],
        "criteria": [["c1", "c2"], ["c1", "c2"], ["d1"]],
        "criteria_score": [[1, 0], [0, 0], [1]],
        "criteria_max_score": [[1, 1], [1, 1], [1]],
    })


def record(prediction, initial, final, verified):
    verdict = {"valid": True, "is_consistent": verified}
    return {"prediction": prediction, "feedback_initial": initial, "feedback": final,
            "verifier_initial": verdict, "verifier": verdict}


def test_items_judge_each_distinct_prompt_once():
    df = make_dataset()
    runs = {
        # Row 0: grader agrees with the dataset; row 1: grader says c1 met, dataset says not; row 2: no feedback.
        "base": {0: record([1, 0], "fb a", "fb a", True), 1: record([1, 0], "fb b", "fb b2", False),
                 2: {"prediction": None, "feedback": None, "feedback_initial": None}},
        "adapter": {0: record([1, 0], "fb c", "fb c", True), 1: record([1, 0], "fb d", "fb d", True)},
    }
    items = judge.build_items(runs, df)
    assert len(items) == 2 * 2 * 2 * 2  # runs x rows with feedback x stages x marks
    assert {item["row_index"] for item in items} == {0, 1}
    row1_base_final = [i for i in items if i["system"] == "base:final" and i["row_index"] == 1]
    assert {i["marks"] for i in row1_base_final} == {"given", "gold"}
    assert all(i["marks_differ"] for i in row1_base_final)
    given = next(i for i in row1_base_final if i["marks"] == "given")
    assert "1. [MET] c1" in given["messages"][1]["content"]
    assert "(no answer given)" in given["messages"][1]["content"]
    # Unchanged rewrites and agreeing marks give the same prompt: base row 0 needs 1 call (not 4),
    # base row 1 needs 4 (rewritten, marks differ), adapter row 0 needs 1 and adapter row 1 needs 2.
    assert len({item["key"] for item in items}) == 8


def test_results_tables_and_report(tmp_path):
    df = make_dataset()
    runs = {"base": {0: record([1, 0], "fb a", "fb a", True), 1: record([1, 0], "fb b", "fb b2", False)},
            "adapter": {0: record([1, 0], "fb c", "fb c", True), 1: record([1, 0], "fb d", "fb d", True)}}
    items = judge.build_items(runs, df)
    passing = json.dumps({key: True for key in judge.JUDGE_KEYS} | {"issues": ""})
    failing = json.dumps({key: key != "faithful_to_student" for key in judge.JUDGE_KEYS} | {"issues": "invented claim"})
    judgements = {}
    for item in items:
        text = failing if "fb b" in item["messages"][1]["content"] else passing
        if item["system"] == "adapter:final" and item["row_index"] == 1 and item["marks"] == "gold":
            text = "unreadable"
        judgements[item["key"]] = judge.judgement_record(item["key"], text, "stop")
    results = judge.results_frame(items, judgements)
    assert len(results) == len(items)
    tables = judge.summary_tables(results)
    by_system = tables["by_system"].set_index(["marks", "system"])
    assert by_system.loc[("given", "base:initial"), "passed"] == 0.5
    assert by_system.loc[("gold", "adapter:final"), "parsed"] == 0.5
    agreement = tables["verifier_agreement"].set_index("system")
    assert agreement.loc["base:initial", "agreement"] == 1.0
    failures = tables["failures"].set_index(["marks", "system"])
    # base row 1 fails faithfulness only; adapter:final row 1 (gold) is unreadable.
    assert failures.loc[("given", "base:initial")][["failed", "unreadable", "faithful_to_student",
                                                    "factually_correct", "one_check_only"]].tolist() == [1, 0, 1, 0, 1]
    assert failures.loc[("gold", "adapter:final")][["failed", "unreadable", "faithful_to_student"]].tolist() == [1, 1, 0]

    class Sample:
        def __init__(self, question):
            self.question = question

    samples = [Sample(q) for q in df.question]
    paired = judge.paired_tables(results, samples, split_seed=42, n_boot=50)
    assert set(paired.marks) == {"given", "gold"}
    assert len(paired) == 2 * 6  # 4 systems -> 6 pairs, per marks
    path = judge.write_report(str(tmp_path), results, tables, paired, 50)
    report = open(path, encoding="utf-8").read()
    assert "## Pass rates" in report and "## Why feedback fails" in report
    assert os.path.exists(tmp_path / "judge_paired.csv") and os.path.exists(tmp_path / "judge_failures.csv")
