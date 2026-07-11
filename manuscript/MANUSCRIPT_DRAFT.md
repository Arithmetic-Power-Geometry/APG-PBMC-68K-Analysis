# Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes

## Abstract
Arithmetic Power Geometry (APG) was evaluated as an interpretable exponent-deformation representation of single-cell transcriptomes. The official 10x Genomics Fresh 68k PBMCs (Donor A) dataset containing 68,579 cells was analysed. APG descriptors were compared directly with Shannon entropy, Rényi-2 entropy, Simpson diversity, Gini inequality, Herfindahl concentration, participation ratio and detected-gene count. Marker-supported provisional immune-cell annotations and biological-programme scores enabled cell-type and programme-level analyses. Three-fold stratified models showed a macro-F1 of 0.3027 for classical diversity descriptors and 0.3792 after adding gene-level APG descriptors. The best evaluated representation, M5_Classical_plus_Gene_and_Program_APG, achieved macro-F1 0.5299. Count-thinning experiments assessed robustness at four sequencing-depth levels. The results establish a reproducible one-donor methodological benchmark and clarify the incremental contribution and limitations of APG in single-cell analysis.

## 1. Introduction
Single-cell profiles are commonly summarized by embeddings or individual diversity measures. APG contributes a bounded family of exponent-dependent defects and an integrated deformation measure. This study tests whether those quantities complement existing descriptors.

## 2. Mathematical framework
For normalized nonnegative expression z_i, w_i=z_i^2/sum_j z_j^2 and D(p)=1-sum_i w_i^(p/2). APG entropy, concentration, effective dimension and integrated deformation are calculated from this distribution.

## 3. Methods
The complete executed procedure is reported in `METHODS.md`.

## 4. Results
Complete descriptor effect sizes are provided in `tables/cell_type_global_tests.csv`. Classification comparisons are provided in `tables/classification_comparison_3fold.csv`. Robustness results are provided in `tables/downsampling_robustness.csv`.

## 5. Discussion
The analysis distinguishes APG-specific deformation information from classical entropy and concentration measures. Results are interpreted as internal methodological evidence because the dataset contains one donor and annotations are computationally derived.

## 6. Conclusion
APG is computationally feasible at single-cell scale and can be evaluated transparently as a complementary structural representation. Independent multi-donor validation is the necessary next stage.
