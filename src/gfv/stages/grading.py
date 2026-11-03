from typing import List, Dict, Any, Optional, Tuple
from src.gfv.data import Sample
from src.gfv.parsing import parse_prediction
from src.gfv.prompts import PromptLoader


def make_prompt(system_prompt: str, question: str, desired_answer: str, student_answer: str, criteria: List[str]) -> List[Dict[str, str]]:
    """Builds the exact prompt structure used during model training."""
    criteria_text = "\n".join(f"{i}. {text}" for i, text in enumerate(criteria, 1))
    user_text = (
        f"Question:\n{question}\n\n"
        f"Desired answer:\n{desired_answer}\n\n"
        f"Student answer:\n{student_answer}\n\n"
        f"Criteria:\n{criteria_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def weighted_score(criterion_scores: Optional[List[int]], criterion_maximums: List[int]) -> Optional[float]:
    if criterion_scores is None:
        return None
    return float(sum(score * maximum for score, maximum in zip(criterion_scores, criterion_maximums)))


class GradingStage:
    """Execution stage for evaluating student answers against a list of rubric criteria.

    The voting stage reuses this class as its evaluation agent, with its own
    model and decoding settings.
    """

    def __init__(self, llm, prompt_path: str, decoding_params: Dict[str, Any], enable_thinking: bool = False):
        self.llm = llm
        self.system_prompt = PromptLoader.load_prompt_template(prompt_path).strip()
        self.decoding_params = decoding_params
        self.enable_thinking = enable_thinking

    def messages(self, sample: Sample) -> List[Dict[str, str]]:
        # The raw (unstripped) text is what the grading adapter was trained on.
        return make_prompt(
            system_prompt=self.system_prompt,
            question=sample.question,
            desired_answer=sample.desired_answer,
            student_answer=sample.student_answer,
            criteria=sample.criteria,
        )

    def run_batch(self, samples: List[Sample]) -> Tuple[List[Dict[str, Any]], float]:
        """Grades every sample. Each result has `valid`, `scores`, `error`, `final_score` and `raw_output`."""
        raw_outputs, elapsed = self.llm.generate_batch(
            [self.messages(sample) for sample in samples], self.decoding_params, self.enable_thinking
        )
        results = []
        for sample, raw_output in zip(samples, raw_outputs):
            parsed = parse_prediction(raw_output, len(sample.criteria))
            results.append({
                **parsed,
                "final_score": weighted_score(parsed["scores"], sample.criteria_max_scores),
                "raw_output": raw_output,
            })
        return results, elapsed

    def run(self, sample: Sample) -> Dict[str, Any]:
        """Runs the model and extracts the criteria evaluation array [0, 1, 0, ...]."""
        results, _ = self.run_batch([sample])
        return results[0]
