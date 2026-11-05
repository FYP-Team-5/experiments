"""Compares finished runs on the same test rows. Needs no GPU: run it on a laptop
after copying the run folders back from the cluster.

    python scripts/compare.py                                  # every run in outputs/comparison/
    python scripts/compare.py outputs/comparison/cmp_grading_base outputs/comparison/cmp_grading_qlora

Writes report.md and CSVs to outputs/comparison/report/ (or --out).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gfv.compare import (compare_feedback, compare_grading, feedback_systems, grading_systems,
                             load_runs, summaries, write_report)
from src.gfv.data import load_jsonl, question_splits


BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


def configured_seeds():
    """The split seeds the runs use, from configs/base.yaml."""
    from omegaconf import OmegaConf
    return [int(seed) for seed in OmegaConf.load(BASE_CONFIG).experiment.seeds]


def main():
    parser = argparse.ArgumentParser(description="Paired comparison of finished runs")
    parser.add_argument("runs", nargs="*", help="Run folders containing records.jsonl "
                                                "(default: every folder in outputs/comparison/)")
    parser.add_argument("--data", default="data/processed/data.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Split seeds to score (default: experiment.seeds in configs/base.yaml)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument("--out", default="outputs/comparison/report")
    args = parser.parse_args()

    paths = args.runs or sorted(
        os.path.dirname(path) for path in glob.glob(os.path.join("outputs", "comparison", "*", "records.jsonl"))
    )
    if not paths:
        parser.error("No runs found; pass run folders or copy them to outputs/comparison/.")

    seeds = args.seeds or configured_seeds()
    samples = load_jsonl(args.data)
    test_indices_by_seed = {
        seed: [sample.index for sample in question_splits(samples, seed)["test"]] for seed in seeds
    }
    print("Seeds:", ", ".join(str(seed) for seed in seeds))
    runs = load_runs(paths)
    print("Runs:", ", ".join(runs))

    summary, agent_stats, metrics_by_seed = summaries(runs, test_indices_by_seed, samples)
    grading = compare_grading(grading_systems(runs), test_indices_by_seed, samples, args.bootstrap)
    feedback = compare_feedback(feedback_systems(runs), test_indices_by_seed, samples, args.bootstrap)
    write_report(args.out, summary, agent_stats, grading, feedback, args.bootstrap, metrics_by_seed)
    print(f"Wrote {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
