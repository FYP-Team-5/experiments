from typing import Dict, Any, List, Optional, Tuple
from src.gfv.data import Sample
from src.gfv.parsing import parse_feedback
from src.gfv.prompts import PromptLoader
from src.gfv.stages.grading import weighted_score


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def prompt_fields(sample: Sample, criterion_scores: List[int]) -> Dict[str, str]:
    """Placeholders available to feedback and verification templates.

    Text is stripped and an empty answer becomes "(no answer given)", as in the
    feedback adapter's training prompt (feedback/qlora_feedback_qwen3.5-9B.ipynb).
    """
    marks = [bool(score) for score in criterion_scores]
    score = weighted_score(criterion_scores, sample.criteria_max_scores)
    return {
        "question": sample.question.strip(),
        "desired_answer": sample.desired_answer.strip(),
        "student_answer": sample.student_answer.strip() or "(no answer given)",
        "rubric": sample.rubric,
        # Feedback adapter format: "1. [MET] ..." / "2. [NOT MET] ..."
        "key_points": "\n".join(
            f"{number}. {'[MET] ' if met else '[NOT MET] '}{criterion}"
            for number, (criterion, met) in enumerate(zip(sample.criteria, marks), start=1)
        ),
        # Multi-agent notebook format: "1. [met] ..." / "2. [not met] ..."
        "graded_criteria": "\n".join(
            f"{number}. [{'met' if met else 'not met'}] {criterion}"
            for number, (criterion, met) in enumerate(zip(sample.criteria, marks), start=1)
        ),
        "met_count": str(sum(marks)),
        "criteria_count": str(len(marks)),
        "score": format_number(score),
        "max_score": format_number(sum(sample.criteria_max_scores)),
    }


class FeedbackStage:
    """Execution logic for generating student feedback on the grade from the grading stage."""

    def __init__(self, llm, template_path: str, decoding_params: Dict[str, Any],
                 system_prompt_path: Optional[str] = None, enable_thinking: bool = False):
        self.llm = llm
        self.template = PromptLoader.load_prompt_template(template_path)
        self.system_prompt = (
            PromptLoader.load_prompt_template(system_prompt_path).strip() if system_prompt_path else None
        )
        self.decoding_params = decoding_params
        self.enable_thinking = enable_thinking

    def messages(self, sample: Sample, criterion_scores: List[int],
                 revision_notes: Optional[List[str]] = None) -> List[Dict[str, str]]:
        user_prompt = PromptLoader.format_prompt(self.template, **prompt_fields(sample, criterion_scores)).strip()
        if revision_notes:
            user_prompt += (
                "\n\nA reviewer rejected your previous feedback for these reasons:\n"
                + "\n".join(f"- {note}" for note in revision_notes)
                + "\nWrite new feedback that fixes them."
            )
        messages = [{"role": "user", "content": user_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def run_batch(self, samples: List[Sample], criterion_score_lists: List[List[int]],
                  revision_notes: Optional[List[Optional[List[str]]]] = None
                  ) -> Tuple[List[Dict[str, Any]], float]:
        revision_notes = revision_notes or [None] * len(samples)
        raw_outputs, elapsed = self.llm.generate_batch(
            [self.messages(sample, scores, notes)
             for sample, scores, notes in zip(samples, criterion_score_lists, revision_notes)],
            self.decoding_params,
            self.enable_thinking,
        )
        return [{**parse_feedback(raw), "feedback_raw_output": raw} for raw in raw_outputs], elapsed

    def run(self, sample: Sample, criterion_scores: List[int]) -> str:
        results, _ = self.run_batch([sample], [criterion_scores])
        return results[0]["feedback"]
