# Grading, feedback and verification experiments

A config-driven pipeline that grades short answers against rubric criteria, writes feedback, and optionally checks it. It runs both multi-agent methods the team has built, on the same data and with the same metrics, so they can be compared. [MULTI_AGENT_COMPARISON.md](MULTI_AGENT_COMPARISON.md) explains the comparison.

## Pipeline

For each row, these agents run in order. All of them except the grader can be switched off in the config.

1. **Grading** (`stages.grading`): marks each criterion 0 or 1. Its prompt is the one the grading adapter `SmuFypTeam5/GradingQlora` was trained on.
2. **Voting** (`stages.voting`): the grade-voting method from [notebooks/multi_agent_vote/](notebooks/multi_agent_vote/). An evaluation agent grades again with sampling. If its marks differ from the grader's, a third grade runs, and each criterion takes the majority mark.
3. **Feedback** (`stages.feedback`): writes feedback on the final marks. The default prompt is the one the feedback adapter `SmuFypTeam5/feedback` was trained on.
4. **Verification** (`stages.verification`): an LLM judge checks that the score and the feedback are consistent with each other, the student's answer and the rubric. Rejected feedback is rewritten with the judge's notes, up to `max_regenerations` times.

Each stage's model is set under `models:` in the experiment config. Stages that use the same model config share one copy of the weights.

## Data and splits

`data/processed/data.jsonl` is made from `data/training_data_moh_eng_rubrics.csv` by `scripts/convert_csv_to_jsonl.py`. The pipeline uses the question-level 80/10/10 split from the qlora notebooks. `experiment.seeds` in `configs/base.yaml` lists the split seeds; the pipeline runs every row in any seed's test split once, then scores each seed on its own test rows. It is set to seed 42 only (389 test rows), because both adapters were trained on the seed-42 split: the test rows of seeds 123 and 2026 include questions they were trained on. `scripts/compare.py` scores the same seeds unless `--seeds` is given.

## Running

[RUNNING_ON_ORIGAMI.md](RUNNING_ON_ORIGAMI.md) has step-by-step instructions for running the comparison on the Origami cluster. The simplest way is to open [run_comparison.ipynb](run_comparison.ipynb) in a `mynotebook` GPU session and press **Run All**: it runs the four comparison configs, resumes after an interruption, and shows the report.

```bash
pip install -r requirements.txt

# One experiment
PYTHONPATH=. python scripts/run_stage.py --config configs/experiments/cmp_grading_qlora.yaml

# Resume an interrupted run: pass its output directory
PYTHONPATH=. python scripts/run_stage.py --config configs/experiments/cmp_grading_qlora.yaml \
    --output_dir outputs/cmp_grading_qlora/20261103_120000

# The comparison on the cluster: submit each config as its own job. The
# CS480G3 account allows 2 submitted jobs, so submit two, then the next one
# after one finishes. Results go to outputs/comparison/<config>/;
# resubmitting a config resumes it.
bash slurm/submit.sh cmp_grading_base
bash slurm/submit.sh cmp_grading_qlora
bash slurm/submit.sh cmp_feedback_base        # after one of the above finishes
bash slurm/submit.sh cmp_feedback_adapter

# Every experiment in a list, in a single job
sbatch slurm/run_stage.sbatch                                                  # slurm/experiments.txt
sbatch --export=ALL,EXPERIMENT_LIST_FILE=slurm/comparison.txt slurm/run_stage.sbatch

# Compare finished runs (no GPU needed): paired differences per seed with
# bootstrap intervals by question, McNemar tests, and cost per agent
python scripts/compare.py                                # every run in outputs/comparison/

# Score an existing records file, including the notebooks' JSONL output
python scripts/evaluate.py --records path/to/records.jsonl --out path/to/report

# Tests (fake model, no GPU needed)
python -m pytest tests
```

## Experiments

| Config | What it runs |
|---|---|
| `exp_01_prompt_v1` | QLoRA grader → base feedback from the rubric only → verifier (audit only) |
| `exp_02_prompt_with_grade_v1` | QLoRA grader → base feedback from the rubric and score → verifier (audit only) |
| `cmp_grading_base` | Base grader, single vs 3-agent vote |
| `cmp_grading_qlora` | QLoRA grader, single vs 3-agent vote |
| `cmp_feedback_base` | QLoRA grader → base feedback → verifier, up to 2 rewrites |
| `cmp_feedback_adapter` | QLoRA grader → feedback adapter → verifier, up to 2 rewrites |
| `exp_local_test` | Every stage with Qwen3.5-2B on 5 rows, on CPU |

## Outputs

Each run writes to `outputs/<experiment_name>/<timestamp>/`:

| File | Contents |
|---|---|
| `config_resolved.yaml` | The merged config |
| `records.jsonl` | One record per row: the final grade (`prediction`, `valid`, `predicted_final_score`), the grading agent's own grade (`grader_*`), every agent's raw output, the feedback before and after verification, the verifier's verdicts, and seconds per agent |
| `metrics_by_seed.csv` | The grading metrics of the qlora Rubrics notebooks, in their order, per seed. A voting run has two systems: `single` (grading agent alone) and `vote` |
| `metrics_seed_summary.csv` | Mean and std of each metric across seeds |
| `agent_stats_by_seed.csv` | Agreement, re-grade and change rates, verifier pass rates before and after rewrites (an unreadable verdict counts as a fail), how often a verdict was unreadable, and minutes per agent |

`scripts/compare.py` writes `report.md`, `grading_summary.csv`, `grading_by_seed.csv`, `grading_paired.csv`, `feedback_paired.csv` and `agent_stats.csv` to `outputs/comparison/report/`.

`calculate_metrics` in `src/gfv/metrics.py` follows the qlora Rubrics notebooks (`evaluate_current_model` and `summarize_question_metrics`): the same metrics in the same order, including `recovered_output_rate` and the question-macro metrics. Its 12 metrics shared with the multi-agent notebooks are computed as they compute them, so rescoring their JSONL output with `scripts/evaluate.py` reproduces their `metrics_by_seed.csv`.

## Layout

| Path | Contents |
|---|---|
| `configs/base.yaml` | Defaults for every experiment: seeds, decoding, prompts, which stages run |
| `configs/experiments/` | One file per experiment; overrides `base.yaml` |
| `configs/models/` | Base model, grading adapter, feedback adapter, local test model |
| `prompts/` | System prompts and templates. Placeholders are `{name}`; other braces are left as they are |
| `src/gfv/` | Data, model wrapper, parsers, metrics, pipeline and one module per stage |
| `notebooks/multi_agent_vote/` | The original grade-voting notebooks and their results, from the qlora repo's `feat/multi-agent` branch |
| `scripts/` | Run one experiment, run a list, convert the CSV, score a records file, compare runs |
| `slurm/` | Cluster jobs (`run_one.sbatch` per config, `run_stage.sbatch` per list), `submit.sh` to submit one config, experiment lists |
| `tests/` | Pipeline tests with a fake model, and prompt checks against the adapters' training prompts |
