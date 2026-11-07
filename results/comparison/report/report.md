# Multi-agent comparison

Each comparison is paired: both systems are scored on the same test rows of one seed. `diff` is B − A; higher is better except for MAE. The interval is a 95% bootstrap (2000 resamples of whole questions); `*` marks an interval that excludes 0.

The grading metrics are those of the qlora Rubrics notebooks, in their order. `recovered_output_rate` is the share of valid grades found inside extra text; for a vote, any recovered input grade counts. The `question_macro_*` metrics give every question equal weight, and an invalid output fails all of its criteria.

## Grading: seed 42

| metric | cmp_grading_base:single | cmp_grading_base:vote | cmp_grading_qlora:single | cmp_grading_qlora:vote |
|---|---|---|---|---|
| examples | 389 | 389 | 389 | 389 |
| valid_examples | 389 | 389 | 387 | 387 |
| recovered_output_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| invalid_output_rate | 0.0000 | 0.0000 | 0.0051 | 0.0051 |
| exact_match_accuracy | 0.7326 | 0.7352 | 0.7892 | 0.7789 |
| per_criterion_accuracy | 0.8675 | 0.8675 | 0.8992 | 0.8959 |
| precision | 0.8833 | 0.8772 | 0.8654 | 0.8607 |
| recall | 0.8745 | 0.8825 | 0.9679 | 0.9679 |
| f1 | 0.8789 | 0.8798 | 0.9137 | 0.9112 |
| final_score_exact_accuracy | 0.7326 | 0.7404 | 0.7943 | 0.7841 |
| final_score_within_one_accuracy | 0.9717 | 0.9743 | 0.9794 | 0.9820 |
| final_score_mae | 0.3111 | 0.3008 | 0.2248 | 0.2326 |
| final_score_rmse | 0.6783 | 0.6669 | 0.5498 | 0.5522 |
| final_score_pearson | 0.8123 | 0.8180 | 0.8766 | 0.8771 |
| final_score_qwk | 0.8051 | 0.8111 | 0.8642 | 0.8635 |
| question_count | 10 | 10 | 10 | 10 |
| question_macro_exact_match_accuracy | 0.7484 | 0.7491 | 0.8095 | 0.8073 |
| question_macro_per_criterion_accuracy | 0.8665 | 0.8663 | 0.8900 | 0.8934 |
| question_macro_criterion_f1 | 0.8571 | 0.8571 | 0.8842 | 0.8876 |

## Grading: paired differences

| a | b | metric | seed 42 |
|---|---|---|---|
| cmp_grading_base:single | cmp_grading_base:vote | exact_match_accuracy | +0.0026 [-0.0154, +0.0200] |
| cmp_grading_base:single | cmp_grading_base:vote | f1 | +0.0010 [-0.0086, +0.0090] |
| cmp_grading_base:single | cmp_grading_base:vote | final_score_mae | -0.0103 [-0.0273, +0.0072] |
| cmp_grading_base:single | cmp_grading_base:vote | final_score_qwk | +0.0060 [-0.0047, +0.0187] |
| cmp_grading_base:single | cmp_grading_qlora:single | exact_match_accuracy | +0.0566 [-0.0060, +0.1371] |
| cmp_grading_base:single | cmp_grading_qlora:single | f1 | +0.0349 [+0.0085, +0.0772]* |
| cmp_grading_base:single | cmp_grading_qlora:single | final_score_mae | -0.0862 [-0.1822, -0.0124]* |
| cmp_grading_base:single | cmp_grading_qlora:single | final_score_qwk | +0.0591 [+0.0148, +0.1343]* |
| cmp_grading_base:single | cmp_grading_qlora:vote | exact_match_accuracy | +0.0463 [-0.0053, +0.1295] |
| cmp_grading_base:single | cmp_grading_qlora:vote | f1 | +0.0323 [+0.0095, +0.0793]* |
| cmp_grading_base:single | cmp_grading_qlora:vote | final_score_mae | -0.0785 [-0.1791, -0.0123]* |
| cmp_grading_base:single | cmp_grading_qlora:vote | final_score_qwk | +0.0585 [+0.0170, +0.1404]* |
| cmp_grading_base:vote | cmp_grading_qlora:single | exact_match_accuracy | +0.0540 [-0.0030, +0.1385] |
| cmp_grading_base:vote | cmp_grading_qlora:single | f1 | +0.0339 [+0.0083, +0.0801]* |
| cmp_grading_base:vote | cmp_grading_qlora:single | final_score_mae | -0.0760 [-0.1792, -0.0067]* |
| cmp_grading_base:vote | cmp_grading_qlora:single | final_score_qwk | +0.0531 [+0.0078, +0.1349]* |
| cmp_grading_base:vote | cmp_grading_qlora:vote | exact_match_accuracy | +0.0437 [-0.0073, +0.1323] |
| cmp_grading_base:vote | cmp_grading_qlora:vote | f1 | +0.0313 [+0.0067, +0.0820]* |
| cmp_grading_base:vote | cmp_grading_qlora:vote | final_score_mae | -0.0682 [-0.1816, -0.0052]* |
| cmp_grading_base:vote | cmp_grading_qlora:vote | final_score_qwk | +0.0525 [+0.0081, +0.1439]* |
| cmp_grading_qlora:single | cmp_grading_qlora:vote | exact_match_accuracy | -0.0103 [-0.0312, +0.0089] |
| cmp_grading_qlora:single | cmp_grading_qlora:vote | f1 | -0.0026 [-0.0089, +0.0071] |
| cmp_grading_qlora:single | cmp_grading_qlora:vote | final_score_mae | +0.0078 [-0.0144, +0.0304] |
| cmp_grading_qlora:single | cmp_grading_qlora:vote | final_score_qwk | -0.0006 [-0.0128, +0.0214] |

## Grading: McNemar on exact match (rows only A got right vs only B)

| a | b | seed 42 |
|---|---|---|
| cmp_grading_base:single | cmp_grading_base:vote | 5 vs 6, p=1.000 |
| cmp_grading_base:single | cmp_grading_qlora:single | 34 vs 56, p=0.026 |
| cmp_grading_base:single | cmp_grading_qlora:vote | 36 vs 54, p=0.073 |
| cmp_grading_base:vote | cmp_grading_qlora:single | 31 vs 52, p=0.028 |
| cmp_grading_base:vote | cmp_grading_qlora:vote | 33 vs 50, p=0.078 |
| cmp_grading_qlora:single | cmp_grading_qlora:vote | 6 vs 2, p=0.289 |

## Feedback: verifier pass rate, A → B

Judged by the pipeline's own verifier (base Qwen3.5-9B), so this is not yet an independent measure of feedback quality; score `feedback_initial` and `feedback` with the 27B judge for that. A verdict that could not be read counts as a fail; the table below gives how often that happened.

| a | b | seed 42 |
|---|---|---|
| cmp_feedback_adapter:initial | cmp_feedback_adapter:final | 0.724 → 0.827 (+0.103 [+0.062, +0.137]*, McNemar p=0.000) |
| cmp_feedback_base:final | cmp_feedback_adapter:final | 0.757 → 0.827 (+0.070 [+0.015, +0.128]*, McNemar p=0.002) |
| cmp_feedback_base:final | cmp_feedback_adapter:initial | 0.757 → 0.724 (-0.034 [-0.096, +0.029], McNemar p=0.203) |
| cmp_feedback_base:initial | cmp_feedback_adapter:final | 0.574 → 0.827 (+0.253 [+0.144, +0.344]*, McNemar p=0.000) |
| cmp_feedback_base:initial | cmp_feedback_adapter:initial | 0.574 → 0.724 (+0.150 [+0.043, +0.249]*, McNemar p=0.000) |
| cmp_feedback_base:initial | cmp_feedback_base:final | 0.574 → 0.757 (+0.183 [+0.103, +0.241]*, McNemar p=0.000) |

## Agents and cost (mean over seeds; minutes per seed's test split)

| run | rows_processed | seconds_per_row | generation_minutes_total | generation_minutes_evaluator | generation_minutes_feedback | generation_minutes_grader | generation_minutes_regrade | generation_minutes_verifier | verifier_pass_rate_initial | verifier_pass_rate_final | verifier_unreadable_rate_initial | verifier_unreadable_rate_final | feedback_revised_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cmp_grading_base | 950 | 0.1502 | 1.0172 | 0.4201 | 0.0000 | 0.4578 | 0.1393 | 0.0000 |  |  |  |  |  |
| cmp_grading_qlora | 389 | 0.2069 | 1.3411 | 0.5647 | 0.0000 | 0.5764 | 0.2001 | 0.0000 |  |  |  |  |  |
| cmp_feedback_base | 389 | 3.6740 | 23.8197 | 0.0000 | 7.7348 | 0.5645 | 0.0000 | 15.5205 | 0.5736 | 0.7571 | 0.0000 | 0.0026 | 0.4242 |
| cmp_feedback_adapter | 389 | 3.0491 | 19.7682 | 0.0000 | 6.1793 | 0.5654 | 0.0000 | 13.0234 | 0.7235 | 0.8269 | 0.0026 | 0.0026 | 0.2725 |
