# Comparing the two multi-agent pipelines fairly

There are two multi-agent implementations, and they couldn't be compared. This file lists what had to change in each before the results mean anything side by side.

## Status on branch `feat/multi-agent-compare`

Both methods now run in this repo's pipeline: grade voting is `stages.voting`, feedback verification is `stages.verification`. The notebooks and their results were moved here from the qlora repo's `feat/multi-agent` branch, under `notebooks/multi_agent_vote/`, with their author kept.

- **Done:** A1–A13, B1–B3 (in this pipeline; the notebooks themselves are unchanged), C1–C3, in C4 the grading metrics, the verifier's pass rates and the time per agent, and C5 (`scripts/compare.py`). Rescoring the notebooks' JSONL output with `scripts/evaluate.py` reproduces their `metrics_by_seed.csv` exactly (all 3 seeds, single and vote). `tests/` runs the pipeline end to end with a fake model.
- **Run on the cluster (2026-11-04):** all four `cmp_*` configs. The verifier (`consistency_v1.txt`, 512 tokens) had two problems in the feedback runs:
  - About 25% of its final verdicts could not be read. It reasoned inside its JSON and was cut off.
  - It saw only the score, without the maximum or the criterion marks. It passed 69–75% of feedback on full-mark answers, but only 6–10% on zero-score answers and 1–3% on partial credit, mostly flagging `SCORE_FEEDBACK_MISMATCH` when the feedback named a missing point.

  The verifier now uses `consistency_v2.txt` with 1024 tokens. It asks for the JSON only, and shows the score out of the maximum, each criterion's mark and the reference answer. An unreadable verdict now counts as a fail instead of being dropped.

  The QLoRA runs also used the earlier `SmuFypTeam5/GradingQlora` (seed-2026 split), so seeds 42 and 123 included questions it was trained on. The adapter was replaced on 2026-11-05 (commit `6ad11d55`) by one trained on the seed-42 split, like the feedback adapter. The comparison is now scored on seed 42 only, the one split that is clean for both adapters.
- **Second run on the cluster (2026-11-05), seed 42:**
  - **The verifier fix worked.** 0.0–0.8% of verdicts are unreadable. Pass rates before → after rewrites are 0.589 → 0.763 for base feedback and 0.689 → 0.815 for the feedback adapter. Feedback on partial-credit answers now passes 27–59%.
  - **The new grading adapter was not applied.** Its weights are named after Qwen3.5's multimodal layout (`model.language_model.layers.N`), which matches nothing in the `AutoModelForCausalLM` the pipeline loads. PEFT only warned, and `cmp_grading_qlora` graded like the base model (0.735 exact match vs the adapter's own 0.784). `load_adapter` in `src/gfv/llm.py` now renames the weights and checks that every weight loaded.
- **Third run on the cluster (2026-11-06), seed 42, with the adapter loading fixed:** QLoRA single grader 0.789 exact match (the adapter's own evaluation: 0.784), against 0.733 for the base model (McNemar p = 0.026). Voting changes nothing for either. The feedback adapter passes the 9B verifier more often than base feedback, before (0.724 vs 0.574) and after rewrites (0.827 vs 0.757).
- **Feedback judged by Qwen3.5-27B (2026-11-07):** `judge_feedback_27b.ipynb` with `src/gfv/judge.py` (B4, C4) judged the third run's feedback with the feedback repo's judge and checks. It judged each feedback against the marks it was written for (**given**, the QLoRA grader's) and against the dataset's marks (**gold**). Of 1,270 distinct judgements, 36 (2.8%) are unreadable: the judge used up its 8,192 tokens while thinking. They count as fails, as in the feedback repo. Leaving them out changes no conclusion.

  | Pass rate on all four checks (387 rows) | Base feedback | Feedback adapter |
  |---|---|---|
  | given, before rewrites | 0.819 | **0.904** |
  | given, after the 9B verifier's rewrites | 0.827 | 0.897 |
  | gold, before rewrites | 0.726 | 0.770 |
  | gold, after rewrites | 0.731 | 0.762 |
  | Format checks clean | 0.468 | 0.997 |

  - **The judge reproduces the feedback repo's result.** That result (base 79.7%, adapter 89.7%, format 44.2% and 100%) had feedback written for and judged against the same marks, as with **given** here.
  - **The feedback adapter writes better feedback.** Under **given** it beats base by about 8 points before rewrites (95% CI [+0.012, +0.136], McNemar 15 vs 48, p < 0.001) and after them. Most of the gain is on partial-credit answers (0.887 vs 0.745), and in factual accuracy (0.959 vs 0.897) and faithfulness to the student's answer (0.933 vs 0.894).
  - **The verify-and-rewrite loop adds nothing.** Rewrites move the pass rate by +0.008 for base and −0.008 for the adapter, neither significant. On the rewritten rows alone: base 0.697 → 0.715 (165 rows), adapter 0.832 → 0.804 (107 rows). The 9B verifier agrees with the 27B judge on only 65–80% of rows and fails much feedback the judge passes (30% of base's first drafts). Its pass-rate gains above are the verifier accepting its own rewrites, not better feedback.
  - **End to end (gold), grading errors dominate.** On the 307 rows where the QLoRA grader's marks match the dataset's, given and gold results are the same (adapter 0.941, base 0.870). On the 80 rows where they differ, only 11–19% pass: the feedback explains the wrong grade. The adapter's lead under gold is about 3–4 points, and most intervals include 0.
  - **Conclusion:** grading adapter → feedback adapter, without the verification loop, is the best and cheapest pipeline. Further gains have to come from grading.

  The full report, with pass rates per check, by outcome, paired tests, agreement with the 9B verifier and failures per check, is [results/comparison/judge_27b/report.md](results/comparison/judge_27b/report.md); the grading and 9B-verifier comparison is [results/comparison/report/report.md](results/comparison/report/report.md). [side_by_side_examples.md](side_by_side_examples.md) shows the feedback on the rows of the feedback repo's examples, with each rewrite and the 9B verifier's verdicts.

On the notebook's own sampled run, `scripts/compare.py` already finds that voting changes no grading metric significantly on any seed: every 95% interval includes 0, and McNemar's p is between 0.55 and 1.0.

The sections below describe the problems as they were before this branch.

## The two pipelines

| | qlora `feat/multi-agent` | experiments (`src/gfv/`) |
|---|---|---|
| Where | `qlora/Multi Agent/Multi_Agent_Rubrics_Qwen35_9B*.ipynb` (now also `notebooks/multi_agent_vote/`) | `src/gfv/pipeline.py`, `src/gfv/stages/` |
| Agents | Grader → Feedback → Evaluator (re-grades; on disagreement a third grade and a per-criterion majority vote) → feedback regenerated if the grade changed | Grading → Feedback → Verification (LLM judge checks that score and feedback are consistent) |
| Models | Base Qwen3.5-9B NF4 for every role, no adapter | Grading uses the QLoRA adapter `SmuFypTeam5/GradingQlora`; feedback and verification use base Qwen3.5-9B |
| What the second check looks at | The grade | The feedback |
| Status | Run on 3 seeds, results in `multi_agent_results/` | Only the feedback stage runs; no outputs exist |

## Why the current results can't be compared

1. The experiments pipeline has never produced a grade, a verification verdict or a metric. It only writes `feedback_generations.jsonl`.
2. The two pipelines use different graders (base vs QLoRA), so any gap would mix "multi-agent vs single" with "fine-tuned vs base".
3. Decoding, prompts, seeds and the inputs given to the feedback agent all differ (see the table under C).
4. Neither pipeline measures feedback quality, which is the main thing the experiments design is meant to improve.

## A. Fixes to the experiments pipeline

These issues stop the full pipeline from running, or make it measure something different from the qlora notebook. All are fixed on this branch. A2 is fixed in `src/gfv/prompts.py`, which now fills only `{name}` placeholders, so the prompt files need no escaping.

| # | File | Problem | Fix |
|---|---|---|---|
| A1 | `src/gfv/pipeline.py` | Only `FeedbackStage` is built and run. `GradingStage` and `VerificationStage` are never called. | Run Grade → Feedback → Verify for each sample, and pass the grader's criterion marks and score into the next two stages. |
| A2 | `prompts/verification/consistency_v1.txt` | The JSON example contains literal `{ }`, so `str.format()` raises `KeyError: '\n  "is_consistent"'`. | Escape them as `{{ }}`, or fill the template with `str.replace` for the named fields only. |
| A3 | `prompts/feedback/answer_rubric_grade_v1.txt` + `src/gfv/stages/feedback.py:15` | The template needs `{score}`, but `FeedbackStage.run` never passes it (`KeyError: 'score'`). | Add a `score` argument (and the per-criterion marks) to `FeedbackStage.run`. |
| A4 | `configs/experiments/exp_02_prompt_with_grade_v1.yaml` | No `models:` block, so the pipeline raises `KeyError: Feedback model config not found`. The defaults path is misspelled (`qwen3.5-9b-qloroa.yaml`). | Copy the `models:` block from `exp_01` and remove the bad defaults entry. |
| A5 | `src/gfv/llm.py:52` | `generate()` hardcodes the feedback system prompt for every stage. The grading adapter then sees a different system prompt from the one it was trained on. | Let each stage pass its own system prompt; the grading stage must use `prompts/grading/grading_v1.txt` as the system message. |
| A6 | `src/gfv/stages/grading.py:52` + `src/gfv/llm.py:67` | `GradingStage` already applies the chat template, and `generate()` applies it again around that text. | Pass messages (not a rendered string) to `generate()` and apply the template once. |
| A7 | `src/gfv/llm.py:67` | `enable_thinking` is not set, so behaviour depends on the chat template's default. The qlora runs use non-thinking mode. | Pass `enable_thinking=False` to `apply_chat_template`. |
| A8 | `src/gfv/pipeline.py:15` | It reads `config["seed"]`, but the seed is stored under `experiment.seed`. Every run uses 42 whatever the config says. | Read `config["experiment"]["seed"]` and accept a list of seeds (42, 123, 2026). |
| A9 | `src/gfv/stages/grading.py:98` | The parser returns `[]` on failure and never checks the array length, so a short array counts as a valid grade. | Reuse `parse_prediction` from the qlora notebook: it requires exactly one binary value per criterion and flags the row invalid otherwise. |
| A10 | `src/gfv/stages/grading.py:74` | `total_score = sum(scores)` ignores criterion weights. All max scores in the current dataset are 1, so it changes nothing today, but it breaks as soon as weights differ. | Use `sum(s * m for s, m in zip(scores, criteria_max_scores))`. |
| A11 | `src/gfv/stages/verification.py:25` | It only parses JSON inside a ```` ```json ```` fence and treats anything else as a failure. | Also try parsing the raw text and the outermost `{...}`, like `parse_feedback` in the qlora notebook. |
| A12 | `configs/models/*.yaml` | `device_map: "cpu"` with 4-bit loading. On the cluster this is very slow or fails. | Use `device_map: "auto"`. |
| A13 | `src/gfv/pipeline.py` | The output record has no `ground_truth`, `prediction`, `valid` or `predicted_final_score`, so no grading metric can be computed. | Write the record fields listed under C3. |

## B. Fixes to the qlora multi-agent notebook

| # | Problem | Fix |
|---|---|---|
| B1 | The grader is base Qwen3.5-9B, so the run can't show whether voting helps the fine-tuned grader we would actually deploy. | Add a run with the QLoRA grading adapter as the grader and evaluator. |
| B2 | The greedy run (`multi_agent_rubrics_nf4_v1`) agrees 100% of the time and is identical to the single grader. It tests nothing. | Report only the sampled run (`..._do_sample_true`), or drop the greedy one. |
| B3 | Its grading system prompt says "Do not include explanations **in the final answer**" and "Do not include any text **after the final JSON array**". The QLoRA notebooks and `grading_v1.txt` say "Do not include explanations." and "…before or after the JSON array." The single-grader score on seed 42 differs: exact match 0.725 here vs 0.740 in `Rubrics/Qlora Comparison/qlora_evaluation_comparison.csv`. | Use the text of `prompts/grading/grading_v1.txt` in both pipelines. |
| B4 | Feedback quality is never scored; only the JSON parse rate (99%) is reported. | Score the feedback with the judge described in C4. |

On this branch, B1 is `configs/experiments/cmp_grading_qlora.yaml`. For B2, the voting stage samples by default (`stages.voting.decoding`). For B3, every config uses `prompts/grading/grading_v1.txt` for both the grader and the evaluator. B4 is `judge_feedback_27b.ipynb`, which scores the feedback runs in this pipeline.

## C. A shared evaluation protocol

### C1. Pin these settings in both pipelines

| Setting | Use |
|---|---|
| Dataset | `training_data_moh_eng_rubrics.csv` (the copies in both repos are identical; MD5 `089d01d4…`) |
| Split | Question-level 80/10/10, test split only. Seed 42 (389 rows), the split both adapters were trained on; seeds 123 and 2026 (278 / 396 rows) overlap the adapters' training questions. Both repos already compute the same split. |
| Grader prompt | `prompts/grading/grading_v1.txt` as the system message, numbered criteria in the user message |
| Grader decoding | Greedy, `max_new_tokens=128`, `enable_thinking=False` |
| Second-opinion decoding | Sampling at temperature 0.7 (the qlora voting needs some randomness to disagree) |
| Feedback input | Question, desired answer, student answer, each criterion marked met / not met, final score |
| Quantization | NF4, double quant, bf16 compute |

### C2. Runs that separate the variables

Grading (scored with `calculate_metrics`):

| Grader | Single grader | 3-agent vote | Config (one run gives both columns) |
|---|---|---|---|
| Base Qwen3.5-9B | In `notebooks/multi_agent_vote/multi_agent_results/`, but with the B3 prompt | Same, `_do_sample_true` | `cmp_grading_base.yaml` |
| QLoRA grading adapter | Needed | Needed | `cmp_grading_qlora.yaml` |

Feedback (scored with the judge in C4), all on the same grades from the QLoRA grader:

| Feedback agent | No verifier | Verifier + regenerate on failure | Config (one run gives both columns) |
|---|---|---|---|
| Base Qwen3.5-9B | Needed | Needed | `cmp_feedback_base.yaml` |
| Feedback adapter (`SmuFypTeam5/feedback`) | Needed | Needed | `cmp_feedback_adapter.yaml` |

In the feedback runs, `feedback_initial` is the "no verifier" column and `feedback` is the "verifier + regenerate" column. `slurm/comparison.txt` lists all four configs.

### C3. One output record format

Each row from either pipeline should contain at least:

```json
{
  "split_seed": 42,
  "dataset_index": 3050,
  "ground_truth": [1, 0, 0],
  "prediction": [1, 0, 1],
  "valid": true,
  "criteria_max_score": [1, 1, 1],
  "true_final_score": 1.0,
  "predicted_final_score": 2.0,
  "feedback": "...",
  "verifier": {"is_consistent": true, "detected_issues": []},
  "agent_seconds": {"grader": 0.15, "feedback": 0.6, "verifier": 0.4}
}
```

With this format the qlora notebook's `calculate_metrics` can score the experiments output unchanged.

On this branch, `records.jsonl` holds one record per row, without `split_seed`: a row that is in several seeds' test splits is processed once, as in the notebooks, and `metrics.py` picks each seed's rows by `dataset_index`.

### C4. Metrics

- **Grading:** the metrics of the qlora Rubrics notebooks, in their order: examples, valid examples, recovered-output rate, invalid rate, exact match, per-criterion accuracy, precision, recall, F1, final-score exact / within one / MAE / RMSE / Pearson / QWK, then question count and the question-macro exact match, per-criterion accuracy and criterion F1. Per seed and as mean ± std. `calculate_metrics` in `src/gfv/metrics.py` implements them.
- **Feedback:** the Qwen3.5-27B judge from `feedback/feedback_teacher_qwen3.5-27B.ipynb`: pass rate on all four judge checks, faithfulness, factual accuracy and the format checks. It already scored base 9B (79.7%) and the feedback adapter (89.7%) on the seed-42 test rows, so those numbers serve as the reference.
- **Verifier:** how often it flags feedback the 27B judge also fails, and how often it flags feedback the judge passes. Without this check, its `is_consistent` verdict can't be trusted.
- **Cost:** generation seconds per row for each agent.

### C5. Report differences properly

The gains seen so far are tiny. With the evaluator sampling, the 3-agent vote changed 39 of 950 rows: 16 became exact matches and 16 stopped being exact matches. Every metric moved by less than the spread between seeds (about ±0.04). So:

- Compare the two systems row by row on the same rows.
- Use McNemar's test for exact match and judge pass rate.
- Use bootstrap 95% confidence intervals (resampled by question) for F1, QWK and MAE.
- Don't call a difference real if its interval includes 0.

## Checklist

- [x] A1–A13 fixed in experiments (tested with a fake model, then run on the cluster)
- [x] B3: both methods use the `grading_v1.txt` prompt
- [x] Both methods write the C3 record format (scored on seed 42)
- [x] QLoRA-grader configs added (single and 3-agent)
- [x] The four `cmp_*` configs run on the cluster (third run, 2026-11-06)
- [x] Feedback from every run scored with the 27B judge (2026-11-07)
- [x] Verifier checked against the judge
- [x] Paired tests, confidence intervals and time per row implemented (`scripts/compare.py`)
- [x] Comparison report produced from the cluster runs
