# Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes

This fully executed package analyses the official 10x Genomics Fresh 68k PBMCs (Donor A) dataset containing 68,579 cells. It compares APG directly with established diversity and concentration measures and includes marker-supported cell annotation, programme-level APG, classification, effect-size analysis, silhouette comparison and count-depth robustness.

## Principal result
Classical diversity macro-F1: **0.3027**. Classical plus gene-level APG macro-F1: **0.3792**. Best tested representation: **M5_Classical_plus_Gene_and_Program_APG**, macro-F1 **0.5299**.

## Contents
- `results/cell_level_descriptors_and_annotations.csv.gz`
- `tables/classification_comparison_3fold.csv`
- `tables/cell_type_global_tests.csv`
- `tables/downsampling_robustness.csv`
- `figures/`
- `manuscript/MANUSCRIPT_DRAFT.md`
- `code/run_complete_analysis.py`

The study is presented as a one-donor methodological and exploratory benchmark.
