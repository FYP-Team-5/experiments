"""Scores a records file on each seed's test split.

Works on this pipeline's records.jsonl and on the notebooks' JSONL output, e.g.

    python scripts/evaluate.py \
        --records "notebooks/multi_agent_vote/multi_agent_results/multi_agent_rubrics_nf4_v1_do_sample_true__qwen3_5_9b__non-thinking.jsonl" \
        --out outputs/notebook_vote_rescored
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gfv.data import load_jsonl, question_splits
from src.gfv.metrics import load_records, write_reports


def main():
    parser = argparse.ArgumentParser(description="Score a records JSONL file per seed")
    parser.add_argument("--records", required=True, help="JSONL with one record per dataset_index")
    parser.add_argument("--data", default="data/processed/data.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--out", default=None, help="Directory for the CSVs (default: next to --records)")
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    test_indices_by_seed = {
        seed: [sample.index for sample in question_splits(samples, seed)["test"]] for seed in args.seeds
    }
    output_dir = args.out or os.path.dirname(os.path.abspath(args.records))
    os.makedirs(output_dir, exist_ok=True)
    write_reports(load_records(args.records), test_indices_by_seed, samples, output_dir)
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()
