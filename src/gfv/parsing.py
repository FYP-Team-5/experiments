"""Output parsers shared by every stage.

`parse_prediction` is copied from
notebooks/multi_agent_vote/Multi_Agent_Rubrics_Qwen35_9B.ipynb, so both
multi-agent pipelines accept and reject exactly the same grades.
`parse_feedback` is adapted from the same notebook: it keeps line breaks
(the experiments prompts allow bullet points) and treats plain-text
feedback as valid (only the notebook's prompt asked for JSON).
"""
import json
import re
from typing import Any, Dict, List, Optional


def final_segment(raw_output: str) -> str:
    """Text after the last </think>, if the model produced a thinking trace."""
    return raw_output.strip().rsplit("</think>", 1)[-1].strip()


def parse_prediction(raw_output: str, expected_count: int) -> Dict[str, Any]:
    """Return a valid binary array of `expected_count` marks, tolerating a preceding thinking trace."""
    text = raw_output.strip()

    candidates = []
    for candidate_text in (final_segment(text), text):
        try:
            candidates.append(json.loads(candidate_text))
        except json.JSONDecodeError:
            pass

    for match in re.findall(r"\[\s*[01](?:\s*,\s*[01])*\s*\]", text):
        candidates.append(json.loads(match))

    for candidate in reversed(candidates):
        if not isinstance(candidate, list):
            continue
        try:
            scores = [int(value) for value in candidate]
        except (TypeError, ValueError):
            continue
        if len(scores) == expected_count and all(value in (0, 1) for value in scores):
            return {"valid": True, "scores": scores, "error": None}

    return {
        "valid": False,
        "scores": None,
        "error": f"No binary JSON array of length {expected_count} was found.",
    }


def parse_feedback(raw_output: str) -> Dict[str, Any]:
    """Always return plain-text feedback, whether the model wrote valid JSON, broken JSON, or plain text.

    `feedback_valid` is False only when the model tried to write a
    {"feedback": ...} object and broke it. Plain-text feedback is valid.
    """
    text = final_segment(raw_output)
    # Drop ```json ... ``` fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    # 1. Valid JSON (whole text, or the outermost {...}).
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("feedback"), str) and parsed["feedback"].strip():
            return {"feedback_valid": True, "feedback": parsed["feedback"].strip()}

    # 2. Broken JSON wrapper (invalid escapes, unescaped quotes, or cut off at the token limit):
    #    take everything after "feedback": " and strip the closing "} if it is there.
    match = re.search(r'"feedback"\s*:\s*"(.*)', text, flags=re.DOTALL)
    if match:
        feedback = re.sub(r'"?\s*}?\s*$', "", match.group(1))
        feedback = feedback.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        return {"feedback_valid": False, "feedback": feedback.strip()}

    # 3. Plain text: use as is.
    return {"feedback_valid": bool(text.strip()), "feedback": text.strip()}


def parse_json_object(raw_output: str) -> Optional[Dict[str, Any]]:
    """The first JSON object found in a ```json fence, the whole text, or the outermost {...}."""
    text = final_segment(raw_output)
    candidates: List[str] = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
