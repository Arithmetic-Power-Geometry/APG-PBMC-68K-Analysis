# APG PBMC Reproducibility Package v2.0

This package contains the manuscript PDF, analysis code, result files, and source/derived data used for the strengthened Arithmetic Power Geometry (APG) PBMC study.

## Package Contents

- `manuscript/main.pdf` – final manuscript PDF.
- `analysis/` – analysis and validation scripts and associated execution files.
- `results/` – numerical results and result tables.
- `source_data/` – source and derived data included for reproducibility.

## Analyses Included

The package supports the principal analyses reported in the manuscript, including gene- and programme-level APG descriptors, classical diversity/concentration descriptors, matched programme-level ablations, PCA versus PCA+APG comparisons, PCA+classical versus PCA+classical+APG comparisons, independent Azimuth/Hao reference validation, fold-specific PCA fitting, technical barcode-prefix sensitivity analysis, exponent-saturation analysis, entropy/deformation redundancy analysis, count-thinning robustness, and silhouette-based clustering assessment.

## External Reference Scope

The compact Azimuth/Hao reference contains frozen broad labels, 228 nonnegative ADT features, and a 50-dimensional RNA reference PCA. It does not contain the full RNA expression matrix. Accordingly, the principal external same-modality comparison is ADT PCA versus ADT PCA+APG. Analyses combining RNA PCA with ADT-derived APG descriptors are interpreted as cross-modal complementarity analyses.

The 13 barcode-prefix groups are used only for technical sensitivity analysis and are not interpreted as donor identifiers.

## Reproducibility Scope

Feature standardization and PCA fitting are performed within training folds where applicable to avoid test-fold leakage. Limiting findings, including high-exponent saturation, negative silhouette coefficients, and count-thinning behaviour, are retained.

APG is evaluated as a compact interpretable structural layer that can complement conventional representations; it is not presented as a replacement for PCA, UMAP, scVI, or general cell-state manifold representations.

## Manuscript Source

Only the compiled manuscript PDF is distributed in this package. 