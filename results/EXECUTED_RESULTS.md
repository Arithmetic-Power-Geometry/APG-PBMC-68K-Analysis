# Executed Results

The final package contains direct comparisons with Shannon entropy, Rényi-2 entropy, Simpson diversity, Gini inequality, Herfindahl concentration, participation ratio and detected-gene count. APG was evaluated at gene and biological-programme levels.

## Classification comparison
The classical diversity model achieved macro-F1 **0.3027 ± 0.0058**. Adding gene-level APG descriptors achieved **0.3792 ± 0.0104**. The best evaluated compact representation was **M5_Classical_plus_Gene_and_Program_APG**, with macro-F1 **0.5299 ± 0.0037** and balanced accuracy **0.5400 ± 0.0020**.

## Completed evidence
- 56,903 cells received confident marker-supported provisional annotations; 11,676 remained uncertain.
- Global descriptor differences were quantified using Kruskal–Wallis tests and epsilon-squared effect sizes.
- Descriptor-space silhouette scores were computed.
- Count-depth robustness was tested at 80%, 60%, 40% and 20% retained counts.

## Scientific boundary
This is a one-donor methodological benchmark. The internal cross-validation results do not establish multi-donor or clinical generalization.
