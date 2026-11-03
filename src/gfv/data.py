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
    # Row position in the JSONL file. It equals `dataset_index` (the CSV row
    # position) in the qlora notebooks, so records from both can be joined.
    index: int = -1

    @property
    def criteria(self) -> List[str]:
        return list(self.metadata["raw_criteria"])

    @property
    def criteria_scores(self) -> List[int]:
        return [int(x) for x in self.metadata["criteria_scores"]]

    @property
    def criteria_max_scores(self) -> List[int]:
        return [int(x) for x in self.metadata["criteria_max_scores"]]


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
                    index=len(samples),
                )
            )
    return samples


def question_splits(samples: List[Sample], seed: int = 42,
                    split_ratio: tuple = (0.80, 0.10, 0.10)) -> Dict[str, List[Sample]]:
    """
    Deterministic question-level 80/10/10 split, identical to the one in the
    qlora notebooks: unique stripped questions are shuffled with `seed`, and
    every row goes to the split of its question.
    """
    question_keys = [s.question.strip() for s in samples]
    unique_questions = list(dict.fromkeys(question_keys))

    rng = random.Random(seed)
    rng.shuffle(unique_questions)

    n_questions = len(unique_questions)
    train_end = int(n_questions * split_ratio[0])
    validation_end = train_end + int(n_questions * split_ratio[1])
    split_of = {}
    for position, question in enumerate(unique_questions):
        split_of[question] = (
            "train" if position < train_end
            else "validation" if position < validation_end
            else "test"
        )

    splits = {"train": [], "validation": [], "test": []}
    for sample, question in zip(samples, question_keys):
        splits[split_of[question]].append(sample)
    return splits


def load_test_split(filepath: str, seed: int = 42, split_ratio: tuple = (0.80, 0.10, 0.10)) -> List[Sample]:
    """
    Loads samples from JSONL and returns ONLY the test split using
    deterministic question-level grouping to prevent data leakage.
    """
    test_samples = question_splits(load_jsonl(filepath), seed, split_ratio)["test"]
    n_questions = len({s.question.strip() for s in test_samples})
    print(
        f"Loaded test split: {len(test_samples)} samples / "
        f"{n_questions} unique questions (Seed: {seed})"
    )
    return test_samples
