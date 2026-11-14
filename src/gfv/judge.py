"""The feedback repo's 27B judge, for the feedback this pipeline writes.

Copied from the feedback repo so the results compare with its reference (base Qwen3.5-9B
79.7%, feedback adapter 89.7% on the seed-42 test rows):

- the judge prompt, its four checks, the row prompt and the parsing from
  `feedback/feedback_teacher_qwen3.5-27B.ipynb` (cells 6 and 10);
- the format checks from `feedback/qlora_feedback_qwen3.5-9B.ipynb` (`format_issues`).

`tests/test_judge.py` checks the copies against those notebooks when the feedback repo sits
next to this one.

The pipeline writes feedback for the marks its grader gave, which are not always the
dataset's. Each feedback is judged twice: against the marks it was written for (`given`, how
good the feedback is) and against the dataset's marks (`gold`, the end-to-end result, which
the reference used). Where the two agree the prompts are identical and are judged once.

`notebooks/judge_feedback_27b.ipynb` builds the items, judges them with vLLM, then calls
`results_frame`, `summary_tables`, `paired_tables` and `write_report`.
"""
import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

JUDGE_MODEL = "Qwen/Qwen3.5-27B-FP8"
JUDGE_THINKING = True  # the teacher notebook: "without reasoning the judge misreads student answers"
MAX_MODEL_LEN = 16384
MAX_JUDGE_TOKENS = 8192

# --- feedback_teacher_qwen3.5-27B.ipynb, cell 6 ---------------------------------------------

JUDGE_SYSTEM_PROMPT = """You audit tutor feedback written for a student's short answer. You are given the question, a reference answer, the student's answer, the marker's MET/NOT MET decision for each key point, and the feedback. Judge the substance of the feedback, not its wording or style.

Checks:
- covers_missing_points: every NOT MET point is addressed with an explanation of the correct idea (true if no point is NOT MET).
- respects_met_points: it never says or implies that a MET point is missing. Correcting a specific incorrect claim the student actually wrote is allowed.
- faithful_to_student: it never credits the student with a specific claim, term or detail that is absent from, or contradicted by, the student's answer. Paraphrasing is fine, and so is affirming the student's general idea and then stating it more precisely.
- factually_correct: its technical content is correct and consistent with the reference answer. Calling a technically valid student point wrong only because the reference answer does not list it is a factual error.

Reply with only this JSON object:
{"covers_missing_points": true or false, "respects_met_points": true or false, "faithful_to_student": true or false, "factually_correct": true or false, "issues": "the main problem in under 20 words, or an empty string"}"""

JUDGE_KEYS = [
    "covers_missing_points",
    "respects_met_points",
    "faithful_to_student",
    "factually_correct",
]


def gold_marks(row: pd.Series) -> List[bool]:
    """The teacher notebook's `criterion_marks`: the dataset's MET/NOT MET per key point."""
    return [score >= maximum for score, maximum in zip(row["criteria_score"], row["criteria_max_score"])]


def build_row_prompt(row: pd.Series, marks: List[bool]) -> str:
    """The teacher notebook's `build_row_prompt`, with the marks passed in."""
    key_points = [f"{number}. {'[MET] ' if met else '[NOT MET] '}{criterion}"
                  for number, (criterion, met) in enumerate(zip(row["criteria"], marks), start=1)]
    sections = [
        f"Question:\n{row['question'].strip()}",
        f"Reference answer:\n{row['desired_answer'].strip()}",
        f"Student answer:\n{row['student_answer'].strip() or '(no answer given)'}",
        "Key points:\n" + "\n".join(key_points),
        f"Key points met: {sum(marks)} of {len(row['criteria'])}",
    ]
    return "\n\n".join(sections)


def build_judge_messages(row: pd.Series, marks: List[bool], feedback: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{build_row_prompt(row, marks)}\n\nFeedback to audit:\n{feedback}"},
    ]


# --- feedback_teacher_qwen3.5-27B.ipynb, cell 10 --------------------------------------------

def parse_judgement(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(verdict.get(key), bool) for key in JUDGE_KEYS):
        return None
    return verdict


# --- qlora_feedback_qwen3.5-9B.ipynb, cell 17 (the "format checks" of its results table) ----

GRADING_TERMS = re.compile(
    r"\b(criteri(on|a)|rubric|checklist|key points|not met|scor(e|es|ed|ing)|grad(e|ed|ing)|marker|"
    r"(full|no|lose|lost|earn|earned|award|awarded) marks?|(reference|model|desired|expected|sample) answer)\b",
    re.IGNORECASE,
)
MARKDOWN = re.compile(r"\*\*|__|^\s*#|^\s*[-*•]\s|^\s*\d+[.)]\s", re.MULTILINE)
NON_ENGLISH = re.compile(r"[　-鿿가-힯]")


def format_issues(text: str, truncated: bool) -> List[str]:
    issues = []
    words = len(text.split())
    if truncated:
        issues.append("truncated")
    if words < 15:
        issues.append("too short")
    if words > 200:
        issues.append("too long")
    if GRADING_TERMS.search(text):
        issues.append("grading language")
    if MARKDOWN.search(text):
        issues.append("markdown")
    if NON_ENGLISH.search(text):
        issues.append("non-English")
    return issues


# --- data -----------------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """The teacher notebook's `load_and_validate_dataset`: row position is the row index, the
    same as `dataset_index` in this pipeline's records."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for column in ["question", "desired_answer", "student_answer"]:
        df[column] = df[column].str.replace("\r\n", "\n").str.replace("\r", "\n")
    df["criteria"] = df["criteria[]"].map(json.loads)
    df["criteria_score"] = df["criteria_score[]"].map(json.loads)
    df["criteria_max_score"] = df["criteria_max_score[]"].map(json.loads)
    return df


def outcome(marks: List[bool]) -> str:
    """The teacher notebook's outcome bands, for the marks the judge is shown."""
    met = sum(marks)
    return "none met" if met == 0 else "all met" if met == len(marks) else "partial"


def prompt_key(messages: List[Dict[str, str]]) -> str:
    """Identifies one judge call. The same prompt is judged once, whichever system it came from."""
    payload = json.dumps([JUDGE_MODEL, JUDGE_THINKING, messages], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_items(runs: Dict[str, Dict[int, Dict[str, Any]]], df: pd.DataFrame,
                count_tokens: Optional[Callable[[str], int]] = None,
                max_feedback_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
    """One item per run, stage (feedback before and after rewrites), marks (given and gold) and row.

    `runs` maps a short name to a feedback run's records by dataset index. Rows whose grade
    was invalid have no feedback and are left out. With `count_tokens`, feedback that used all
    of the feedback stage's `max_feedback_tokens` counts as truncated."""
    from src.gfv.metrics import verdict_passed

    items = []
    for run, records in runs.items():
        for index, record in sorted(records.items()):
            if not record.get("feedback"):
                continue
            row = df.loc[index]
            marks_by_mode = {"given": [score == 1 for score in record["prediction"]], "gold": gold_marks(row)}
            for stage, feedback_key, verdict_key in (("initial", "feedback_initial", "verifier_initial"),
                                                     ("final", "feedback", "verifier")):
                feedback = record[feedback_key]
                truncated = bool(count_tokens and max_feedback_tokens
                                 and count_tokens(feedback) >= max_feedback_tokens - 2)
                for mode, marks in marks_by_mode.items():
                    messages = build_judge_messages(row, marks, feedback)
                    items.append({
                        "system": f"{run}:{stage}", "run": run, "stage": stage, "marks": mode,
                        "row_index": int(index), "outcome": outcome(marks),
                        "marks_differ": marks_by_mode["given"] != marks_by_mode["gold"],
                        "key": prompt_key(messages), "messages": messages,
                        "format_issues": format_issues(feedback, truncated),
                        "verifier_9b_passed": verdict_passed(record.get(verdict_key)),
                    })
    return items


# --- judgements -----------------------------------------------------------------------------

def load_judgements(path: str) -> Dict[str, Dict[str, Any]]:
    judgements = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    judgements[record["key"]] = record
    return judgements


def judgement_record(key: str, text: str, finish_reason: str) -> Dict[str, Any]:
    verdict = parse_judgement(text)
    return {"key": key, "verdict": verdict, "finish_reason": finish_reason,
            "raw_judgement": None if verdict is not None else text}


def results_frame(items: List[Dict[str, Any]], judgements: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        judgement = judgements.get(item["key"])
        if judgement is None:
            continue
        verdict = judgement["verdict"]
        checks = {key: bool(verdict and verdict[key]) for key in JUDGE_KEYS}
        rows.append({
            **{k: item[k] for k in ("system", "run", "stage", "marks", "row_index", "outcome", "marks_differ",
                                    "verifier_9b_passed")},
            "parsed": verdict is not None,
            **checks,
            # An unreadable judgement fails, as in the teacher notebook.
            "passed": verdict is not None and all(checks.values()),
            "format_clean": not item["format_issues"],
            "format_issues": "; ".join(item["format_issues"]),
            "issues": (verdict or {}).get("issues", "") if verdict else "unparsable judgement",
        })
    return pd.DataFrame(rows)


def summary_tables(results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    columns = JUDGE_KEYS + ["passed", "parsed", "format_clean"]
    by_system = results.groupby(["marks", "system"], sort=False).agg(
        rows=("row_index", "size"), **{column: (column, "mean") for column in columns}).reset_index()
    by_outcome = results.pivot_table(index=["marks", "outcome"], columns="system", values="passed",
                                     aggfunc="mean", sort=False).reset_index()
    # The 9B verifier saw the marks the feedback was written for, so compare it with `given`.
    rows = []
    given = results[(results.marks == "given") & results.verifier_9b_passed.notna()]
    for system, group in given.groupby("system", sort=False):
        judge, verifier = group.passed.astype(bool).to_numpy(), group.verifier_9b_passed.astype(bool).to_numpy()
        rows.append({"system": system, "rows": len(group),
                     "both_pass": (judge & verifier).mean(), "only_27b_passes": (judge & ~verifier).mean(),
                     "only_9b_passes": (~judge & verifier).mean(), "both_fail": (~judge & ~verifier).mean(),
                     "agreement": (judge == verifier).mean()})
    return {"by_system": by_system, "by_outcome": by_outcome, "verifier_agreement": pd.DataFrame(rows),
            "failures": failure_counts(results)}


def failure_counts(results: pd.DataFrame) -> pd.DataFrame:
    """Failed rows per marks and system, split into unreadable judgements and the checks that
    failed. A row can fail several checks, so the check columns can add up to more than `failed`."""
    rows = []
    for (mode, system), group in results.groupby(["marks", "system"], sort=False):
        read = group[group.parsed]
        failed_checks = ~read[JUDGE_KEYS].astype(bool)
        rows.append({"marks": mode, "system": system, "rows": len(group),
                     "failed": int((~group.passed.astype(bool)).sum()),
                     "unreadable": int((~group.parsed.astype(bool)).sum()),
                     **{key: int(failed_checks[key].sum()) for key in JUDGE_KEYS},
                     "one_check_only": int((failed_checks.sum(axis=1) == 1).sum())})
    return pd.DataFrame(rows)


def paired_tables(results: pd.DataFrame, samples, split_seed: int, n_boot: int = 2000) -> pd.DataFrame:
    """Pass rate of every pair of systems on the same rows, per marks: the comparison report's
    bootstrap by question and McNemar test."""
    from src.gfv.compare import compare_feedback

    frames = []
    for mode, group in results.groupby("marks", sort=False):
        systems = {system: dict(zip(rows.row_index, rows.passed)) for system, rows in group.groupby("system", sort=False)}
        indices = sorted(group.row_index.unique())
        frame = compare_feedback(systems, {split_seed: indices}, samples, n_boot)
        frame.insert(0, "marks", mode)
        frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    paired["metric"] = "judge_pass_rate"
    return paired


REFERENCE = ("The feedback repo's own result on these 389 rows, with feedback written for and judged against "
             "the dataset's marks: base Qwen3.5-9B passes all four checks on 79.7% and the format checks on "
             "44.2%; the feedback adapter 89.7% and 100%.")


def write_report(output_dir: str, results: pd.DataFrame, tables: Dict[str, pd.DataFrame],
                 paired: pd.DataFrame, n_boot: int) -> str:
    from src.gfv.compare import markdown_table

    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, "judge_results.csv"), index=False)
    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"judge_{name}.csv"), index=False)
    paired.to_csv(os.path.join(output_dir, "judge_paired.csv"), index=False)

    def rounded(frame):
        return frame.round(3)

    lines = [
        "# Feedback judged by Qwen3.5-27B",
        "",
        f"The feedback repo's judge (`{JUDGE_MODEL}`, thinking on) with its four checks, on the feedback from "
        "`cmp_feedback_base` and `cmp_feedback_adapter`. `initial` is the feedback before any rewrite, `final` "
        "after the 9B verifier's rewrites. `passed` needs all four checks; an unreadable judgement fails. "
        "`format_clean` is the feedback notebook's format check.",
        "",
        "- **given**: judged against the marks the feedback was written for (the QLoRA grader's). This measures "
        "the feedback.",
        "- **gold**: judged against the dataset's marks. Feedback that follows a wrong grade fails, so this is the "
        "end-to-end result, grading errors included.",
        "",
        REFERENCE,
        "",
        "## Pass rates",
        "",
        markdown_table(rounded(tables["by_system"])),
        "",
        "## Pass rate by outcome (of the marks shown to the judge)",
        "",
        markdown_table(rounded(tables["by_outcome"])),
        "",
        f"## Paired differences (B − A, 95% interval from {n_boot} resamples of whole questions; McNemar)",
        "",
    ]
    for mode, group in paired.groupby("marks", sort=False):
        lines += [f"**{mode}**", "", "| a | b | pass rate | diff [95% CI] | McNemar |", "|---|---|---|---|---|"]
        for row in group.itertuples(index=False):
            star = "*" if row.ci_excludes_0 else ""
            lines.append(f"| {row.a} | {row.b} | {row.a_value:.3f} → {row.b_value:.3f} | "
                         f"{row.diff_b_minus_a:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]{star} | "
                         f"{row.only_a_passed} vs {row.only_b_passed}, p={row.mcnemar_p:.3f} |")
        lines.append("")
    lines += [
        "## Agreement with the pipeline's 9B verifier (given marks)",
        "",
        "Share of rows by the two verdicts. Low agreement means the 9B verifier's pass rates in the comparison "
        "report should not be read as feedback quality.",
        "",
        markdown_table(rounded(tables["verifier_agreement"])),
        "",
        "## Why feedback fails",
        "",
        "Rows that failed: unreadable judgements (the judge ran out of tokens while thinking), then the number of "
        "readable judgements failing each check. A row can fail several checks; `one_check_only` counts rows that "
        "failed exactly one. The judge's own notes on each row are in `judge_results.csv` (`issues`).",
        "",
        markdown_table(tables["failures"]),
        "",
    ]
    path = os.path.join(output_dir, "report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
