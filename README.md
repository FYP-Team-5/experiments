# Grading, feedback and verification experiments

A config-driven pipeline that grades short answers against rubric criteria and writes feedback on them. It was built to answer one question: **do multi-agent methods (grade voting, feedback verification) improve on a single grader and a single feedback writer?**

It runs both multi-agent methods the team built on the same data, with the same prompts and metrics, so they can be compared.

## Findings

Measured on 389 test rows from questions that no model was trained on.

| Question | Result |
|---|---|
| Does fine-tuning the grader help? | **Yes.** Exact match rises from 0.733 to 0.789, and F1 from 0.879 to 0.914. |
| Does a 3-agent grade vote help? | **No.** It changes exact match by +0.003 (base grader) and −0.010 (fine-tuned grader), neither significant. |
| Does the fine-tuned feedback adapter help? | **Yes.** 90.4% of its feedback passes an independent Qwen3.5-27B judge, against 81.9% for the base model. |
| Does a verify-and-rewrite loop help? | **No.** It changes the judge's pass rate by less than 1 point. |

**Conclusion:** the best and cheapest pipeline is the grading adapter followed by the feedback adapter, with no voting and no verification loop.

## Where to find things

| If you want | Read |
|---|---|
| Every grading metric, with confidence intervals and significance tests | [results/comparison/report/report.md](results/comparison/report/report.md) |
| The 27B judge's results on the feedback | [results/comparison/judge_27b/report.md](results/comparison/judge_27b/report.md) |
| Actual feedback from both models on ten test rows | [results/comparison/side_by_side_examples.md](results/comparison/side_by_side_examples.md) |

```
experiments/
├── configs/                 base.yaml, one file per experiment, one per model
├── prompts/                 Prompt text for grading, feedback and verification
├── src/gfv/                 The pipeline: data, model wrapper, parsers, metrics, one module per stage
├── scripts/                 Run an experiment, convert data, score records, compare runs
├── slurm/                   Cluster job scripts
├── tests/                   Pipeline tests with a fake model (no GPU needed)
├── data/                    The dataset, as CSV and as the JSONL the pipeline reads
├── results/comparison/      Committed reports of the final comparison run
└── notebooks/
    ├── run_comparison.ipynb     Runs the four comparison configs and shows the report
    ├── judge_feedback_27b.ipynb Scores the feedback with the Qwen3.5-27B judge
    └── multi_agent_vote/        The original grade-voting notebooks and their results
```

## Pipeline

For each row, these agents run in order. All except the grader can be switched off in the config.

| Stage | What it does |
|---|---|
| 1. Grading | Marks each rubric criterion 0 or 1. Its prompt is the one the grading adapter `SmuFypTeam5/GradingQlora` was trained on. |
| 2. Voting | A second agent grades again with sampling. If the two disagree, a third grade runs and each criterion takes the majority mark. |
| 3. Feedback | Writes feedback on the final marks. The default prompt is the one the feedback adapter `SmuFypTeam5/feedback` was trained on. |
| 4. Verification | A judge agent checks that the score and the feedback agree with each other, the student's answer and the rubric. Rejected feedback is rewritten with the judge's notes, up to `max_regenerations` times. |

All stages use Qwen3.5-9B, with or without an adapter. Each stage's model is set under `models:` in the experiment config, and stages with the same model config share one copy of the weights.

## Data

`data/training_data_moh_eng_rubrics.csv` has 3,778 graded answers to 95 questions. `scripts/convert_csv_to_jsonl.py` turns it into `data/processed/data.jsonl`, which the pipeline reads.

The pipeline uses the same question-level 80/10/10 split as the adapters' training notebooks. Results are scored on the seed-42 test split only (389 rows), because both adapters were trained on the seed-42 training split. The seeds are set by `experiment.seeds` in `configs/base.yaml`.

## Experiments

| Config | What it runs |
|---|---|
| `cmp_grading_base` | Base grader, single vs 3-agent vote |
| `cmp_grading_qlora` | QLoRA grader, single vs 3-agent vote |
| `cmp_feedback_base` | QLoRA grader → base feedback → verifier, up to 2 rewrites |
| `cmp_feedback_adapter` | QLoRA grader → feedback adapter → verifier, up to 2 rewrites |
| `exp_01_prompt_v1` | QLoRA grader → base feedback from the rubric only → verifier (audit only) |
| `exp_02_prompt_with_grade_v1` | QLoRA grader → base feedback from the rubric and score → verifier (audit only) |
| `exp_local_test` | Every stage with Qwen3.5-2B on 5 rows, on CPU |

The four `cmp_*` configs are the comparison reported above.

## Running

The simplest way is to open [notebooks/run_comparison.ipynb](notebooks/run_comparison.ipynb) in a GPU session and press **Run All**. It runs the four comparison configs, resumes after an interruption, and shows the report.

From the command line:

```bash
pip install -r requirements.txt

# Tests (fake model, no GPU needed)
python -m pytest tests

# One experiment
PYTHONPATH=. python scripts/run_stage.py --config configs/experiments/cmp_grading_qlora.yaml

# Resume an interrupted run: pass its output directory
PYTHONPATH=. python scripts/run_stage.py --config configs/experiments/cmp_grading_qlora.yaml \
    --output_dir outputs/cmp_grading_qlora/20261103_120000

# On the cluster: one Slurm job per config (resubmitting resumes it)
bash slurm/submit.sh cmp_grading_base

# Compare finished runs (no GPU needed)
python scripts/compare.py

# Score an existing records file, including the original notebooks' JSONL output
python scripts/evaluate.py --records path/to/records.jsonl --out path/to/report
```

[notebooks/judge_feedback_27b.ipynb](notebooks/judge_feedback_27b.ipynb) scores the feedback runs with the Qwen3.5-27B judge. It needs vLLM and a 48 GB GPU.

## Outputs

Each run writes to `outputs/<experiment_name>/<timestamp>/` (the comparison runs to `outputs/comparison/<config>/`):

| File | Contents |
|---|---|
| `config_resolved.yaml` | The merged config |
| `records.jsonl` | One record per row: the final grade, the grading agent's own grade, every agent's raw output, the feedback before and after verification, the verifier's verdicts, and seconds per agent |
| `metrics_by_seed.csv` | The grading metrics per seed. A voting run has two systems: `single` (grading agent alone) and `vote` |
| `metrics_seed_summary.csv` | Mean and std of each metric across seeds |
| `agent_stats_by_seed.csv` | Agreement, re-grade and change rates, verifier pass rates before and after rewrites, and minutes per agent |

`scripts/compare.py` writes its report to `outputs/comparison/report/`, and the judge notebook to `outputs/comparison/judge_27b/`.

`outputs/` is not committed. The reports of the final comparison run are copied to [results/comparison/](results/comparison/), without the per-row `records.jsonl` and `judgements.jsonl`.

## Related repositories

| Repo | Relation |
|---|---|
| `qlora` | Trains the grading adapter. The grading metrics and the split here follow its Rubrics notebooks, and the grade-voting notebooks come from its `feat/multi-agent` branch. |
| `feedback` | Trains the feedback adapter. The 27B judge's prompts and checks in `src/gfv/judge.py` are copied from its notebooks, and `tests/test_judge.py` checks that they still match. |
