from typing import Any, Dict, List, Optional, Tuple
from src.gfv.data import Sample
from src.gfv.parsing import parse_json_object
from src.gfv.prompts import PromptLoader
from src.gfv.stages.feedback import prompt_fields


class VerificationStage:
    """Checks that the score and the feedback are consistent with each other, the answer and the rubric."""

    def __init__(self, llm, prompt_path: str, decoding_params: dict,
                 system_prompt_path: Optional[str] = None, enable_thinking: bool = False):
        self.llm = llm
        self.template = PromptLoader.load_prompt_template(prompt_path)
        self.system_prompt = (
            PromptLoader.load_prompt_template(system_prompt_path).strip() if system_prompt_path else None
        )
        self.decoding_params = decoding_params
        self.enable_thinking = enable_thinking

    def messages(self, sample: Sample, criterion_scores: List[int], feedback: str) -> List[Dict[str, str]]:
        prompt = PromptLoader.format_prompt(
            self.template, **prompt_fields(sample, criterion_scores), generated_feedback=feedback
        ).strip()
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    @staticmethod
    def parse(raw_output: str) -> Dict[str, Any]:
        """`valid` is False when no verdict could be read; `is_consistent` is then None, not False,
        so a parsing failure is never counted as a rejected feedback."""
        parsed = parse_json_object(raw_output)
        if parsed is None or not isinstance(parsed.get("is_consistent"), bool):
            return {
                "valid": False,
                "is_consistent": None,
                "confidence_score": None,
                "detected_issues": [{
                    "issue_type": "PARSING_ERROR",
                    "severity": "HIGH",
                    "description": "Failed to parse structured JSON from verification agent.",
                }],
                "verification_summary": None,
                "raw_output": raw_output,
            }
        issues = parsed.get("detected_issues") or []
        return {
            "valid": True,
            "is_consistent": parsed["is_consistent"],
            "confidence_score": parsed.get("confidence_score"),
            "detected_issues": issues if isinstance(issues, list) else [issues],
            "verification_summary": parsed.get("verification_summary"),
            "raw_output": raw_output,
        }

    def run_batch(self, samples: List[Sample], criterion_score_lists: List[List[int]],
                  feedbacks: List[str]) -> Tuple[List[Dict[str, Any]], float]:
        raw_outputs, elapsed = self.llm.generate_batch(
            [self.messages(sample, scores, feedback)
             for sample, scores, feedback in zip(samples, criterion_score_lists, feedbacks)],
            self.decoding_params,
            self.enable_thinking,
        )
        return [self.parse(raw) for raw in raw_outputs], elapsed

    def run(self, sample: Sample, generated_scores: List[int], generated_feedback: str) -> dict:
        results, _ = self.run_batch([sample], [generated_scores], [generated_feedback])
        return results[0]


def revision_notes(verdict: Dict[str, Any]) -> List[str]:
    """The verifier's issues, as notes for the feedback agent's next attempt."""
    notes = []
    for issue in verdict.get("detected_issues") or []:
        if isinstance(issue, dict):
            label = issue.get("issue_type", "ISSUE")
            notes.append(f"{label}: {issue.get('description', '').strip()}".strip())
        else:
            notes.append(str(issue))
    return notes or [verdict.get("verification_summary") or "The feedback was judged inconsistent."]
