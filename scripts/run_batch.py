# scripts/run_batch.py
import os
import sys
import argparse
from datetime import datetime
from omegaconf import OmegaConf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_stage import load_and_merge_config
from src.gfv.pipeline import run_pipeline


def parse_experiments_file(file_path: str) -> list[str]:
    """Reads experiment config paths from a line-separated text file, ignoring empty lines and comments."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Experiment list file not found: {file_path}")

    config_paths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            # Ignore empty lines and comment lines starting with #
            if clean_line and not clean_line.startswith("#"):
                config_paths.append(clean_line)
    return config_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple GFV experiment stages sequentially read from a batch file."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="slurm/experiments.txt",
        help="Path to file containing list of experiment config YAML paths (default: slurm/experiments.txt)",
    )
    args = parser.parse_args()

    config_paths = parse_experiments_file(args.file)
    print(f"Loaded {len(config_paths)} experiment configurations from: {args.file}")

    for idx, config_path in enumerate(config_paths, 1):
        if not os.path.exists(config_path):
            print(f"\n[Warning] Config file not found, skipping ({idx}/{len(config_paths)}): {config_path}")
            continue

        print(f"\n==========================================================")
        print(f"[{idx}/{len(config_paths)}] Executing Experiment: {config_path}")
        print(f"==========================================================")

        # 1. Load & recursively resolve config (merges defaults via OmegaConf)
        cfg_obj = load_and_merge_config(config_path)
        config = OmegaConf.to_container(cfg_obj, resolve=True)
        config["_resolved_config_path"] = config_path

        # 2. Setup timestamped run output directory: outputs/<exp_name>/<timestamp>/
        exp_name = config.get("experiment_name") or os.path.splitext(os.path.basename(config_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_run_dir = os.path.join("outputs", exp_name, timestamp)
        os.makedirs(output_run_dir, exist_ok=True)

        # 3. Save exact resolved config for experiment auditing
        with open(os.path.join(output_run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg_obj, f=f)

        # 4. Execute GFV stage pipeline
        run_pipeline(config, output_run_dir)

    print("\nBatch execution completed for all listed experiments!")


if __name__ == "__main__":
    main()