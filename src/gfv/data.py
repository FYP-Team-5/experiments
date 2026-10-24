# src/gfv/data.py
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Sample:
    id: str
    question: str
    desired_answer: str
    student_answer: str
    rubric: str
    ground_truth_score: Optional[float] = None
    ground_truth_feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def load_jsonl(filepath: str) -> List[Sample]:
    """Loads all samples from a JSONL file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            samples.append(
                Sample(
                    id=item["id"],
                    question=item["question"],
                    desired_answer=item.get("desired_answer", ""),
                    student_answer=item["student_answer"],
                    rubric=item["rubric"],
                    ground_truth_score=item.get("ground_truth_score"),
                    ground_truth_feedback=item.get("ground_truth_feedback", ""),
                    metadata=item.get("metadata"),
                )
            )
    return samples


def load_test_split(filepath: str, seed: int = 42, split_ratio: tuple = (0.80, 0.10, 0.10)) -> List[Sample]:
    """
    Loads samples from JSONL and returns ONLY the test split using 
    deterministic question-level grouping to prevent data leakage.
    """
    all_samples = load_jsonl(filepath)

    # Extract ordered question keys
    question_keys = [s.question.strip() for s in all_samples]
    unique_questions = list(dict.fromkeys(question_keys))

    # Shuffle unique question IDs with fixed seed
    rng = random.Random(seed)
    rng.shuffle(unique_questions)

    # Compute 80 / 10 / 10 boundary cutoffs
    n_questions = len(unique_questions)
    train_end = int(n_questions * split_ratio[0])
    validation_end = train_end + int(n_questions * split_ratio[1])

    # Isolate test questions (the remaining 10%)
    test_questions = set(unique_questions[validation_end:])

    # Select test samples
    test_samples = [s for s in all_samples if s.question.strip() in test_questions]

    print(
        f"Loaded test split: {len(test_samples)} samples / "
        f"{len(test_questions)} unique questions (Seed: {seed})"
    )
    return test_samples