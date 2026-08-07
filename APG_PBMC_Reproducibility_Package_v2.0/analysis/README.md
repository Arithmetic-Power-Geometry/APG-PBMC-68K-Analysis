# Executed analysis workflow

This folder contains the scripts used for the strengthened manuscript analyses.

1. `01_extract_azimuth_reference.py` parses the supplied Azimuth `ref.Rds` object and exports the frozen broad cell labels, 228-dimensional ADT profile, 50-dimensional RNA reference PCA, and technical barcode-prefix groups.
2. `02_fresh68k_ablation_and_entropy_case.py` reruns the decisive Fresh 68k programme-score ablation, gene-level ablation, entropy/deformation redundancy analysis, and entropy-matched biological example.
3. `03_azimuth_strict_5fold_pca_apg.py` performs strict five-fold evaluation. ADT PCA is fit inside each training fold before comparison with PCA+APG and PCA+classical+APG. The supplied RNA `refDR` coordinates are treated as frozen reference features.
4. `04_azimuth_group_sensitivity.py` refits ADT PCA while leaving one barcode-prefix technical group out at a time. These groups are explicitly not interpreted as donors.
5. `05_make_figures.py` recreates manuscript figures from the executed result tables.

All random seeds are fixed. The external reference contains 36,433 cells in the compact object supplied for this study. The object does not contain the full nonnegative RNA expression matrix; therefore external APG is computed on its ADT profile, while the RNA PCA is used as an orthogonal latent representation.
