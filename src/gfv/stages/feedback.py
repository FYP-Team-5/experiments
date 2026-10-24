from typing import Dict, Any
from src.gfv.llm import LLMWrapper
from src.gfv.prompts import PromptLoader
from src.gfv.data import Sample

class FeedbackStage:
    """Execution logic for generating student feedback."""
    
    def __init__(self, llm: LLMWrapper, prompt_path: str, decoding_params: Dict[str, Any]):
        self.llm = llm
        self.template = PromptLoader.load_prompt_template(prompt_path)
        self.decoding_params = decoding_params

    def run(self, sample: Sample) -> str:
        prompt = PromptLoader.format_prompt(
            self.template,
            question=sample.question,
            rubric=sample.rubric,
            student_answer=sample.student_answer
        )
        return self.llm.generate(prompt, self.decoding_params)
