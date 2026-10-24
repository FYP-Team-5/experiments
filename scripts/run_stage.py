import os
import sys
import argparse
from datetime import datetime
from omegaconf import OmegaConf, DictConfig, ListConfig

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gfv.pipeline import run_pipeline


def load_and_merge_config(config_path: str) -> DictConfig:
    """Recursively loads YAML files, merges 'defaults' references, and resolves nested model configs."""
    raw_cfg = OmegaConf.load(config_path)
    merged_cfg = OmegaConf.create()

    # 1. Handle Hydra-style 'defaults' references
    if OmegaConf.is_config(raw_cfg) and "defaults" in raw_cfg:
        defaults_list = raw_cfg.get("defaults")
        
        if isinstance(defaults_list, (list, ListConfig)):
            for item in defaults_list:
                rel_path = item if isinstance(item, str) else list(item.values())[0]

                if not os.path.exists(rel_path):
                    rel_path = os.path.join(os.path.dirname(config_path), rel_path)

                if os.path.exists(rel_path):
                    sub_cfg = load_and_merge_config(rel_path)
                    merged_cfg = OmegaConf.merge(merged_cfg, sub_cfg)

    # 2. Merge top-level config values over defaults
    merged_cfg = OmegaConf.merge(merged_cfg, raw_cfg)

    # Clean up 'defaults' key
    if "defaults" in merged_cfg:
        del merged_cfg["defaults"]

    # 3. Dynamically resolve stage model paths (e.g., models.grading: "configs/models/qwen2.5.yaml")
    if "models" in merged_cfg and isinstance(merged_cfg.models, (dict, DictConfig)):
        for agent_role, model_target in list(merged_cfg.models.items()):
            if isinstance(model_target, str) and model_target.endswith((".yaml", ".yml")):
                model_path = model_target
                if not os.path.exists(model_path):
                    model_path = os.path.join(os.path.dirname(config_path), model_path)
                
                if os.path.exists(model_path):
                    merged_cfg.models[agent_role] = load_and_merge_config(model_path)

    return merged_cfg


def main():
    parser = argparse.ArgumentParser(description="Run GFV pipeline stage")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for output directory")
    args = parser.parse_args()

    # Load, recursively merge defaults, and resolve model references
    cfg_obj = load_and_merge_config(args.config)
    config = OmegaConf.to_container(cfg_obj, resolve=True)
    config["_resolved_config_path"] = args.config

    # Output directory matching project layout
    exp_name = config.get("experiment_name") or os.path.splitext(os.path.basename(args.config))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_run_dir = args.output_dir or os.path.join("outputs", exp_name, timestamp)
    os.makedirs(output_run_dir, exist_ok=True)

    # Save exact resolved config into run directory for auditability
    with open(os.path.join(output_run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg_obj, f=f)

    # Run execution pipeline
    run_pipeline(config, output_run_dir)


if __name__ == "__main__":
    main()