import json
import re
from typing import List, Dict, Any, Union
from src.gfv.llm import LLMWrapper
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


class GradingStage:
    """Execution stage for evaluating student answers against a list of rubric criteria."""

    def __init__(self, llm: LLMWrapper, prompt_path: str, decoding_params: Dict[str, Any]):
        self.llm = llm
        self.system_prompt = PromptLoader.load_prompt_template(prompt_path)
        self.decoding_params = decoding_params

    def format_input_prompt(self, sample) -> str:
        """
        Converts the sample's data into the model's chat template string.
        """
        # Retrieve criteria as a list (from metadata or parsed rubric string)
        if isinstance(sample.rubric, list):
            criteria = sample.rubric
        elif getattr(sample, "metadata", None) and "raw_criteria" in sample.metadata:
            criteria = sample.metadata["raw_criteria"]
        else:
            # Fallback: split rubric text into lines if passed as a formatted string
            criteria = [line.strip("- ") for line in sample.rubric.split("\n") if line.strip()]

        messages = make_prompt(
            system_prompt=self.system_prompt,
            question=sample.question,
            desired_answer=sample.desired_answer,
            student_answer=sample.student_answer,
            criteria=criteria
        )

        # Apply model tokenizer's chat template if available, otherwise join text
        if hasattr(self.llm.tokenizer, "apply_chat_template"):
            return self.llm.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            return f"System: {self.system_prompt}\n\nUser:\n{messages[1]['content']}\n\nAssistant:"

    def run(self, sample) -> Dict[str, Any]:
        """
        Runs the model and extracts the criteria evaluation array [0, 1, 0, ...].
        """
        formatted_prompt = self.format_input_prompt(sample)
        raw_output = self.llm.generate(formatted_prompt, self.decoding_params)

        # Parse JSON array output safely
        scores = self._parse_json_array(raw_output)

        return {
            "raw_output": raw_output,
            "criteria_scores": scores,
            "total_score": sum(scores) if isinstance(scores, list) else 0.0
        }

    @staticmethod
    def _parse_json_array(text: str) -> Union[List[int], None]:
        """Extracts and parses a JSON list of 0s and 1s from the raw response string."""
        text = text.strip()
        
        # 1. Direct JSON parse
        try:
            res = json.loads(text)
            if isinstance(res, list):
                return [int(x) for x in res]
        except Exception:
            pass

        # 2. Regex fallback if model included markdown formatting or text around array
        match = re.search(r"\[\s*([01]\s*,\s*)*[01]\s*\]", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return []