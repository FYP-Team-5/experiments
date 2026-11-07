# Feedback judged by Qwen3.5-27B

The feedback repo's judge (`Qwen/Qwen3.5-27B-FP8`, thinking on) with its four checks, on the feedback from `cmp_feedback_base` and `cmp_feedback_adapter`. `initial` is the feedback before any rewrite, `final` after the 9B verifier's rewrites. `passed` needs all four checks; an unreadable judgement fails. `format_clean` is the feedback notebook's format check.

- **given**: judged against the marks the feedback was written for (the QLoRA grader's). This measures the feedback.
- **gold**: judged against the dataset's marks. Feedback that follows a wrong grade fails, so this is the end-to-end result, grading errors included.

The feedback repo's own result on these 389 rows, with feedback written for and judged against the dataset's marks: base Qwen3.5-9B passes all four checks on 79.7% and the format checks on 44.2%; the feedback adapter 89.7% and 100%.

## Pass rates

| marks | system | rows | covers_missing_points | respects_met_points | faithful_to_student | factually_correct | passed | parsed | format_clean |
|---|---|---|---|---|---|---|---|---|---|
| given | base:initial | 387 | 0.9590 | 0.9510 | 0.8940 | 0.8970 | 0.8190 | 0.9660 | 0.4680 |
| gold | base:initial | 387 | 0.8910 | 0.9480 | 0.8500 | 0.9020 | 0.7260 | 0.9790 | 0.4680 |
| given | base:final | 387 | 0.9480 | 0.9530 | 0.8890 | 0.9120 | 0.8270 | 0.9720 | 0.4830 |
| gold | base:final | 387 | 0.8860 | 0.9480 | 0.8500 | 0.9120 | 0.7310 | 0.9840 | 0.4830 |
| given | adapter:initial | 387 | 0.9840 | 0.9720 | 0.9330 | 0.9590 | 0.9040 | 0.9840 | 0.9970 |
| gold | adapter:initial | 387 | 0.8860 | 0.9480 | 0.8710 | 0.9460 | 0.7700 | 0.9900 | 0.9970 |
| given | adapter:final | 387 | 0.9770 | 0.9660 | 0.9280 | 0.9510 | 0.8970 | 0.9770 | 0.9970 |
| gold | adapter:final | 387 | 0.8790 | 0.9460 | 0.8660 | 0.9430 | 0.7620 | 0.9820 | 0.9970 |

## Pass rate by outcome (of the marks shown to the judge)

| marks | outcome | base:initial | base:final | adapter:initial | adapter:final |
|---|---|---|---|---|---|
| given | all met | 0.8550 | 0.8550 | 0.9090 | 0.8970 |
| given | none met | 0.8770 | 0.9140 | 0.9260 | 0.9630 |
| given | partial | 0.7450 | 0.7450 | 0.8870 | 0.8580 |
| gold | all met | 0.8620 | 0.8700 | 0.8840 | 0.8770 |
| gold | none met | 0.7640 | 0.7870 | 0.7980 | 0.8430 |
| gold | partial | 0.5880 | 0.5810 | 0.6560 | 0.6190 |

## Paired differences (B − A, 95% interval from 2000 resamples of whole questions; McNemar)

**given**

| a | b | pass rate | diff [95% CI] | McNemar |
|---|---|---|---|---|
| base:initial | base:final | 0.819 → 0.827 | +0.008 [-0.009, +0.028] | 22 vs 25, p=0.771 |
| base:initial | adapter:initial | 0.819 → 0.904 | +0.085 [+0.012, +0.136]* | 15 vs 48, p=0.000 |
| base:initial | adapter:final | 0.819 → 0.897 | +0.078 [+0.019, +0.125]* | 17 vs 47, p=0.000 |
| base:final | adapter:initial | 0.827 → 0.904 | +0.078 [+0.012, +0.116]* | 15 vs 45, p=0.000 |
| base:final | adapter:final | 0.827 → 0.897 | +0.070 [+0.019, +0.103]* | 16 vs 43, p=0.001 |
| adapter:initial | adapter:final | 0.904 → 0.897 | -0.008 [-0.024, +0.015] | 9 vs 6, p=0.607 |

**gold**

| a | b | pass rate | diff [95% CI] | McNemar |
|---|---|---|---|---|
| base:initial | base:final | 0.726 → 0.731 | +0.005 [-0.025, +0.032] | 24 vs 26, p=0.888 |
| base:initial | adapter:initial | 0.726 → 0.770 | +0.044 [-0.026, +0.083] | 19 vs 36, p=0.030 |
| base:initial | adapter:final | 0.726 → 0.762 | +0.036 [-0.015, +0.064] | 21 vs 35, p=0.081 |
| base:final | adapter:initial | 0.731 → 0.770 | +0.039 [-0.006, +0.072] | 21 vs 36, p=0.063 |
| base:final | adapter:final | 0.731 → 0.762 | +0.031 [+0.004, +0.053]* | 22 vs 34, p=0.141 |
| adapter:initial | adapter:final | 0.770 → 0.762 | -0.008 [-0.022, +0.015] | 11 vs 8, p=0.648 |

## Agreement with the pipeline's 9B verifier (given marks)

Share of rows by the two verdicts. Low agreement means the 9B verifier's pass rates in the comparison report should not be read as feedback quality.

| system | rows | both_pass | only_27b_passes | only_9b_passes | both_fail | agreement |
|---|---|---|---|---|---|---|
| base:initial | 387 | 0.5220 | 0.2970 | 0.0520 | 0.1290 | 0.6510 |
| base:final | 387 | 0.6770 | 0.1500 | 0.0800 | 0.0930 | 0.7700 |
| adapter:initial | 387 | 0.6740 | 0.2300 | 0.0490 | 0.0470 | 0.7210 |
| adapter:final | 387 | 0.7620 | 0.1340 | 0.0650 | 0.0390 | 0.8010 |

## Why feedback fails

Rows that failed: unreadable judgements (the judge ran out of tokens while thinking), then the number of readable judgements failing each check. A row can fail several checks; `one_check_only` counts rows that failed exactly one. The judge's own notes on each row are in `judge_results.csv` (`issues`).

| marks | system | rows | failed | unreadable | covers_missing_points | respects_met_points | faithful_to_student | factually_correct | one_check_only |
|---|---|---|---|---|---|---|---|---|---|
| given | base:initial | 387 | 70 | 13 | 3 | 6 | 28 | 27 | 50 |
| gold | base:initial | 387 | 106 | 8 | 34 | 12 | 50 | 30 | 73 |
| given | base:final | 387 | 67 | 11 | 9 | 7 | 32 | 23 | 42 |
| gold | base:final | 387 | 104 | 6 | 38 | 14 | 52 | 28 | 69 |
| given | adapter:initial | 387 | 37 | 6 | 0 | 5 | 20 | 10 | 27 |
| gold | adapter:initial | 387 | 89 | 4 | 40 | 16 | 46 | 17 | 58 |
| given | adapter:final | 387 | 40 | 9 | 0 | 4 | 19 | 10 | 29 |
| gold | adapter:final | 387 | 92 | 7 | 40 | 14 | 45 | 15 | 62 |
