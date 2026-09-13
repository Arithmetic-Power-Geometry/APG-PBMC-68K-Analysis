# Executed Results

This repository contains the executed APG PBMC analyses used in the current manuscript. All values below come from committed workflow outputs.

## Fresh 68k programme ablation

On the balanced 6,400-cell Fresh 68k benchmark, programme classical summaries achieved macro-F1 **0.2838 ± 0.0078**, programme APG achieved **0.3806 ± 0.0135**, and programme classical + APG achieved **0.4127 ± 0.0108**. Across gene- and programme-level summaries, all classical features achieved **0.4258 ± 0.0100**, while all classical + all APG achieved **0.5333 ± 0.0035**.

## Feature-number-matched nonlinear controls

A separate control test asked whether the APG gain could be explained merely by adding more nonlinear coordinates. For each seven-feature classical block, twelve deterministic label-independent nonlinear control coordinates were added (seven squared terms and five log1p terms), exactly matching the twelve APG coordinates. The same balanced cells, five stratified folds, preprocessing and classifier were used.

| Representation | Features | Macro-F1 |
|---|---:|---:|
| Gene classical | 7 | 0.3274 ± 0.0059 |
| Gene classical + matched nonlinear control | 19 | 0.3355 ± 0.0051 |
| Gene classical + APG | 19 | **0.3876 ± 0.0041** |
| Programme classical | 7 | 0.2838 ± 0.0078 |
| Programme classical + matched nonlinear control | 19 | 0.3143 ± 0.0059 |
| Programme classical + APG | 19 | **0.4127 ± 0.0108** |
| All classical | 14 | 0.4258 ± 0.0100 |
| All classical + matched nonlinear control | 38 | 0.4445 ± 0.0056 |
| All classical + all APG | 38 | **0.5333 ± 0.0035** |

APG exceeded the feature-matched nonlinear control in all five folds for all three matched comparisons. Mean paired macro-F1 advantages were **0.0521** at gene level (95% fold-level t interval 0.0480 to 0.0562), **0.0984** at programme level (0.0904 to 0.1064), and **0.0887** for the combined 38-feature representation (0.0858 to 0.0917). With only five folds, the exact two-sided Wilcoxon value is 0.0625 and is treated descriptively rather than as confirmatory inference.

## Additional completed evidence

- 56,903 cells received confident marker-supported provisional annotations; 11,676 remained uncertain.
- Global descriptor differences were quantified using Kruskal-Wallis tests and epsilon-squared effect sizes.
- Count-depth robustness was tested at 80%, 60%, 40% and 20% retained counts.
- Independent Azimuth/Hao validation uses frozen broad labels and a same-modality ADT PCA comparison.
- Grouped technical sensitivity was evaluated across 13 barcode-prefix groups; these groups are not interpreted as donors.

## Scientific boundary

The Fresh 68k analysis is a one-donor methodological benchmark. The compact Azimuth object lacks the underlying RNA expression matrix, so the external same-modality APG test is performed on ADT rather than RNA. The results therefore do not establish multi-donor transcriptomic or clinical generalization.
