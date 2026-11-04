"""Grade -> (vote) -> feedback -> (verify, then revise rejected feedback).

Every stage after grading can be switched off in the config, so the same
pipeline runs both multi-agent methods:

- stages.voting: the grade-voting method from notebooks/multi_agent_vote/
- stages.verification: the feedback-verification method of this repo

Each row in any seed's test split is processed once and written to
records.jsonl. Each seed is then scored on its own test rows, as in the
notebooks.
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional

from src.gfv.data import Sample, load_jsonl, question_splits
from src.gfv.llm import LLMWrapper, clear_cuda_cache, is_out_of_memory
from src.gfv.metrics import load_records, write_reports
from src.gfv.stages.feedback import FeedbackStage
from src.gfv.stages.grading import GradingStage
from src.gfv.stages.verification import VerificationStage, revision_notes
from src.gfv.stages.voting import VotingStage

AGENTS = ("grader", "evaluator", "regrade", "feedback", "verifier")


def resolve_seeds(config: Dict[str, Any]) -> List[int]:
    experiment = config.get("experiment", {})
    if experiment.get("seeds"):
        return [int(seed) for seed in experiment["seeds"]]
    return [int(experiment.get("seed", 42))]


class ModelCache:
    """Loads each distinct model config once, so stages that share a model share its weights."""

    def __init__(self, factory: Callable[[dict], Any]):
        self.factory = factory
        self.models: Dict[str, Any] = {}

    def get(self, model_cfg: dict):
        key = json.dumps(model_cfg, sort_keys=True)
        if key not in self.models:
            self.models[key] = self.factory(model_cfg)
        return self.models[key]


def build_stages(config: Dict[str, Any], llm_factory: Callable[[dict], Any]) -> Dict[str, Any]:
    models = config.get("models") or {}
    stages_cfg = config.get("stages") or {}
    thinking = bool(config.get("experiment", {}).get("enable_thinking", False))
    cache = ModelCache(llm_factory)

    def model_for(role: str, fallback: Optional[str] = None) -> Any:
        model_cfg = models.get(role) or (models.get(fallback) if fallback else None)
        if not isinstance(model_cfg, dict):
            raise KeyError(f"No model config for the '{role}' stage (set models.{role}).")
        return cache.get(model_cfg)

    def enabled(name: str) -> bool:
        return bool(stages_cfg.get(name, {}).get("enabled", name in ("grading", "feedback")))

    grading_cfg = stages_cfg["grading"]
    stages: Dict[str, Any] = {
        "grading": GradingStage(
            model_for("grading"), grading_cfg["system_prompt_path"], grading_cfg.get("decoding", {}), thinking
        )
    }
    if enabled("voting"):
        voting_cfg = stages_cfg["voting"]
        stages["voting"] = VotingStage(GradingStage(
            model_for("voting", fallback="grading"),
            voting_cfg.get("system_prompt_path", grading_cfg["system_prompt_path"]),
            voting_cfg.get("decoding", {}),
            thinking,
        ))
    if enabled("feedback"):
        feedback_cfg = stages_cfg["feedback"]
        stages["feedback"] = FeedbackStage(
            model_for("feedback"), feedback_cfg["template_path"], feedback_cfg.get("decoding", {}),
            feedback_cfg.get("system_prompt_path"), thinking,
        )
    if enabled("verification"):
        if "feedback" not in stages:
            raise ValueError("stages.verification needs stages.feedback to be enabled.")
        verification_cfg = stages_cfg["verification"]
        stages["verification"] = VerificationStage(
            model_for("verification"), verification_cfg["template_path"], verification_cfg.get("decoding", {}),
            verification_cfg.get("system_prompt_path"), thinking,
        )
        stages["max_regenerations"] = int(verification_cfg.get("max_regenerations", 0))
    return stages


def process_batch(stages: Dict[str, Any], samples: List[Sample]) -> List[Dict[str, Any]]:
    n = len(samples)
    seconds = {agent: [0.0] * n for agent in AGENTS}

    def charge(agent: str, positions: List[int], elapsed: float):
        for position in positions:
            seconds[agent][position] += elapsed / len(positions)

    # 1. Grading agent
    grader, elapsed = stages["grading"].run_batch(samples)
    charge("grader", list(range(n)), elapsed)

    # 2. Optional vote: evaluator, re-grade on disagreement, per-criterion majority
    voting = [None] * n
    final = grader
    if "voting" in stages:
        voting, vote_seconds = stages["voting"].run_batch(samples, grader)
        for agent in ("evaluator", "regrade"):
            for p in range(n):
                seconds[agent][p] += vote_seconds[agent][p]
        final = [outcome["final"] for outcome in voting]

    # 3. Feedback on the final marks (skipped when no agent produced a valid grade)
    feedback: List[Optional[Dict[str, Any]]] = [None] * n
    graded = [p for p in range(n) if final[p]["valid"]]
    if "feedback" in stages and graded:
        results, elapsed = stages["feedback"].run_batch(
            [samples[p] for p in graded], [final[p]["scores"] for p in graded]
        )
        charge("feedback", graded, elapsed)
        for p, result in zip(graded, results):
            feedback[p] = result

    # 4. Optional verification; rejected feedback is rewritten with the verifier's notes
    verdicts: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    initial_feedback = [result["feedback"] if result else None for result in feedback]
    revisions = [0] * n
    if "verification" in stages:
        pending = [p for p in graded if feedback[p] is not None]
        for attempt in range(stages["max_regenerations"] + 1):
            if not pending:
                break
            results, elapsed = stages["verification"].run_batch(
                [samples[p] for p in pending],
                [final[p]["scores"] for p in pending],
                [feedback[p]["feedback"] for p in pending],
            )
            charge("verifier", pending, elapsed)
            for p, verdict in zip(pending, results):
                verdicts[p].append(verdict)
            rejected = [p for p, verdict in zip(pending, results) if verdict["is_consistent"] is False]
            if not rejected or attempt == stages["max_regenerations"]:
                break
            results, elapsed = stages["feedback"].run_batch(
                [samples[p] for p in rejected],
                [final[p]["scores"] for p in rejected],
                [revision_notes(verdicts[p][-1]) for p in rejected],
            )
            charge("feedback", rejected, elapsed)
            for p, result in zip(rejected, results):
                feedback[p] = result
                revisions[p] += 1
            pending = rejected

    records = []
    for p, sample in enumerate(samples):
        record = {
            # Fields read by calculate_metrics (the final grade)
            "dataset_index": sample.index,
            "dataset_id": str(sample.id),
            "valid": final[p]["valid"],
            "error": final[p]["error"],
            "ground_truth": sample.criteria_scores,
            "prediction": final[p]["scores"],
            "criteria_max_score": sample.criteria_max_scores,
            "true_final_score": float(sample.ground_truth_score),
            "predicted_final_score": final[p]["final_score"],
            "raw_output": grader[p]["raw_output"],
            "generation_seconds": sum(seconds[agent][p] for agent in AGENTS),
            # Grading agent alone
            "grader_valid": grader[p]["valid"],
            "grader_error": grader[p]["error"],
            "grader_prediction": grader[p]["scores"],
            "grader_final_score": grader[p]["final_score"],
            "agent_seconds": {agent: seconds[agent][p] for agent in AGENTS},
        }
        if voting[p] is not None:
            evaluator, regrade = voting[p]["evaluator"], voting[p]["regrade"]
            record.update({
                "evaluator_valid": evaluator["valid"],
                "evaluator_prediction": evaluator["scores"],
                "evaluator_raw_output": evaluator["raw_output"],
                "regraded": regrade is not None,
                "regrade_valid": regrade["valid"] if regrade else None,
                "regrade_prediction": regrade["scores"] if regrade else None,
                "regrade_raw_output": regrade["raw_output"] if regrade else None,
                "agents_agreed": final[p]["resolution"] == "agree",
                "resolution": final[p]["resolution"],
                "final_changed": final[p]["valid"] and final[p]["scores"] != grader[p]["scores"],
            })
        if "feedback" in stages:
            record.update({
                "feedback": feedback[p]["feedback"] if feedback[p] else None,
                "feedback_valid": feedback[p]["feedback_valid"] if feedback[p] else None,
                "feedback_raw_output": feedback[p]["feedback_raw_output"] if feedback[p] else None,
            })
        if "verification" in stages:
            record.update({
                "feedback_initial": initial_feedback[p],
                "feedback_revisions": revisions[p],
                "verifier_initial": verdicts[p][0] if verdicts[p] else None,
                "verifier": verdicts[p][-1] if verdicts[p] else None,
                "verifier_attempts": verdicts[p],
            })
        records.append(record)
    return records


def run_pipeline(config: Dict[str, Any], output_run_dir: str,
                 llm_factory: Callable[[dict], Any] = LLMWrapper):
    """Runs the configured stages on every seed's test split, then scores each seed.

    Rerunning with the same output directory resumes: rows already in
    records.jsonl are skipped.
    """
    input_path = config["data"]["input_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found at: {input_path}")

    samples = load_jsonl(input_path)
    seeds = resolve_seeds(config)
    test_indices_by_seed = {
        seed: [sample.index for sample in question_splits(samples, seed)["test"]] for seed in seeds
    }
    for seed, indices in test_indices_by_seed.items():
        print(f"Seed {seed}: {len(indices)} test rows", flush=True)
    run_indices = sorted(set().union(*test_indices_by_seed.values()))

    os.makedirs(output_run_dir, exist_ok=True)
    records_path = os.path.join(output_run_dir, "records.jsonl")
    completed = set(load_records(records_path))
    pending = [index for index in run_indices if index not in completed]
    print(f"{len(run_indices) - len(pending)} rows already done, {len(pending)} to run", flush=True)

    if pending:
        stages = build_stages(config, llm_factory)
        batch_size = int(config.get("experiment", {}).get("batch_size", 8))
        position = 0
        with open(records_path, "a", encoding="utf-8") as handle:
            while position < len(pending):
                batch = [samples[index] for index in pending[position:position + batch_size]]
                try:
                    records = process_batch(stages, batch)
                except Exception as error:
                    if not is_out_of_memory(error) or batch_size == 1:
                        raise
                    clear_cuda_cache()
                    batch_size = max(1, batch_size // 2)
                    print(f"CUDA OOM: retrying with batch size {batch_size}", flush=True)
                    continue
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                position += len(batch)
                print(f"Processed {position}/{len(pending)}", flush=True)

    write_reports(load_records(records_path), test_indices_by_seed, samples, output_run_dir)
    print(f"Pipeline finished. Output written to {output_run_dir}", flush=True)
