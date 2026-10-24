import os
import json
from typing import Dict, Any
from src.gfv.data import load_test_split
from src.gfv.llm import LLMWrapper
from src.gfv.stages.feedback import FeedbackStage

def run_pipeline(config: Dict[str, Any], output_run_dir: str):
    """Orchestrates stage execution exclusively on the test split for the given experiment config."""
    input_path = config["data"]["input_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found at: {input_path}")

    # 1. Load ONLY the test split using deterministic seed 42
    split_seed = config.get("seed", 42)
    samples = load_test_split(input_path, seed=split_seed)
    
    # 2. Initialize LLM wrapper
    if "models" in config and isinstance(config["models"], dict) and "feedback" in config["models"]:
        model_cfg = config["models"]["feedback"]
    else:
        raise KeyError("Feedback model config not found")

    llm = LLMWrapper(model_cfg)
    
    # 3. Resolve prompt path
    prompt_path = config["prompt"].get("feedback_path") or config["prompt"].get("path")
    if not prompt_path or not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")

    stage = FeedbackStage(
        llm=llm,
        prompt_path=prompt_path,
        decoding_params=config.get("decoding", {})
    )
    
    # 4. Process test samples and stream generations to JSONL output
    generations_file = os.path.join(output_run_dir, "feedback_generations.jsonl")
    with open(generations_file, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, 1):
            generated_text = stage.run(sample)
            record = {
                "id": sample.id,
                "question": sample.question,
                "student_answer": sample.student_answer,
                "prompt_used": prompt_path,
                "generated_output": generated_text,
                "ground_truth_score": sample.ground_truth_score
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            
            print(f"Processed {idx}/{len(samples)}: {sample.id}", flush=True)

    print(f"Pipeline finished. Output written to {generations_file}", flush=True)