import json
from src.gfv.llm import LLMWrapper
from src.gfv.prompts import PromptLoader

class VerificationStage:
    def __init__(self, llm: LLMWrapper, prompt_path: str, decoding_params: dict):
        self.llm = llm
        self.template = PromptLoader.load_prompt_template(prompt_path)
        self.decoding_params = decoding_params

    def run(self, sample, generated_score: float, generated_feedback: str) -> dict:
        prompt = PromptLoader.format_prompt(
            self.template,
            question=sample.question,
            rubric=sample.rubric,
            student_answer=sample.student_answer,
            score=generated_score,
            generated_feedback=generated_feedback
        )
        
        raw_output = self.llm.generate(prompt, self.decoding_params)
        
        # Parse JSON block from verification output
        try:
            json_str = raw_output.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except Exception:
            return {
                "is_consistent": False,
                "confidence_score": 0.0,
                "detected_issues": [{
                    "issue_type": "PARSING_ERROR",
                    "severity": "HIGH",
                    "description": "Failed to parse structured JSON from verification agent."
                }],
                "raw_output": raw_output
            }