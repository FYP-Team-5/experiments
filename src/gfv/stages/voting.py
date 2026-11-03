"""Grade voting: the multi-agent method from notebooks/multi_agent_vote/.

An evaluation agent grades the answer again. If its marks differ from the
grader's (or either output is invalid), it grades a third time, and each
criterion takes the most common of the three marks.

The notebook wrote feedback on the grader's marks and rewrote it when the vote
changed the grade. The pipeline here writes feedback once, on the final marks.
With greedy feedback decoding the final feedback is the same, and it costs
less.
"""
from typing import Any, Dict, List, Optional, Tuple
from src.gfv.data import Sample
from src.gfv.stages.grading import GradingStage, weighted_score


def resolve_grades(grader: Dict[str, Any], evaluator: Dict[str, Any],
                   regrade: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Combine parse_prediction results into the final criterion array."""
    if grader["valid"] and evaluator["valid"] and grader["scores"] == evaluator["scores"]:
        return {"valid": True, "scores": grader["scores"], "error": None, "resolution": "agree"}

    valid_arrays = [
        parsed["scores"] for parsed in (grader, evaluator, regrade)
        if parsed is not None and parsed["valid"]
    ]
    if len(valid_arrays) == 3:
        # Most common mark for each criterion; three binary votes always have a majority.
        scores = [int(sum(votes) >= 2) for votes in zip(*valid_arrays)]
        resolution = "majority"
    elif len(valid_arrays) == 2:
        # No majority is possible, so a criterion is met only if both valid grades agree it is.
        scores = [int(all(votes)) for votes in zip(*valid_arrays)]
        resolution = "two_valid"
    elif len(valid_arrays) == 1:
        scores = valid_arrays[0]
        resolution = "single_valid"
    else:
        return {
            "valid": False, "scores": None, "resolution": "invalid",
            "error": "No agent produced a valid criterion array.",
        }
    return {"valid": True, "scores": scores, "error": None, "resolution": resolution}


class VotingStage:
    def __init__(self, evaluator: GradingStage):
        self.evaluator = evaluator

    def run_batch(self, samples: List[Sample], grader: List[Dict[str, Any]]
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """Returns, per sample, the evaluator and re-grade results and the resolved grade,
        plus the seconds each agent spent per sample."""
        n = len(samples)
        seconds = {"evaluator": [0.0] * n, "regrade": [0.0] * n}

        evaluator, elapsed = self.evaluator.run_batch(samples)
        seconds["evaluator"] = [elapsed / n] * n

        regrade: List[Optional[Dict[str, Any]]] = [None] * n
        disputed = [
            p for p in range(n)
            if not (grader[p]["valid"] and evaluator[p]["valid"] and grader[p]["scores"] == evaluator[p]["scores"])
        ]
        if disputed:
            results, elapsed = self.evaluator.run_batch([samples[p] for p in disputed])
            for p, result in zip(disputed, results):
                regrade[p] = result
                seconds["regrade"][p] = elapsed / len(disputed)

        outcomes = []
        for p, sample in enumerate(samples):
            final = resolve_grades(grader[p], evaluator[p], regrade[p])
            final["final_score"] = weighted_score(final["scores"], sample.criteria_max_scores)
            outcomes.append({"evaluator": evaluator[p], "regrade": regrade[p], "final": final})
        return outcomes, seconds
