# Arithmetic Power Geometry (APG) Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This repository presents the complete reproducible research package accompanying the study:

**Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes**

The project introduces **Arithmetic Power Geometry (APG)** as an interpretable mathematical framework for describing the structural organization of single-cell RNA sequencing (scRNA-seq) transcriptomes.

Unlike conventional diversity measures that summarize transcriptomic distributions using entropy alone, APG constructs deterministic structural descriptors based on normalized quadratic weights, allowing complementary characterization of diversity, concentration, effective dimensionality, and program-level organization.

The repository contains fully reproducible code, processed data, statistical analyses, figures, tables, and manuscript materials.

---

# Dataset

The study uses the official public dataset

**10x Genomics Fresh 68k PBMCs (Donor A)**

containing

- **68,579 cells**
- Human peripheral blood mononuclear cells (PBMCs)
- UMI-based scRNA-seq

The dataset is used exclusively for research and methodological evaluation.

---

# Objectives

The objectives of this study are to

- introduce APG descriptors for single-cell transcriptomic analysis;
- evaluate APG against established diversity measures;
- investigate structural differences among immune cell populations;
- study transcriptomic concentration and effective dimensionality;
- evaluate machine-learning performance using APG-derived descriptors;
- assess robustness under sequencing-depth downsampling;
- provide a completely reproducible benchmark for future APG research.

---

# Arithmetic Power Geometry (APG)

For each cell, normalized quadratic weights are computed as

$$
w_i=\frac{x_i^2}{\sum_j x_j^2}
$$

from which APG descriptors are derived, including

- Shannon structural entropy
- concentration
- effective dimension
- diversity
- gene-program descriptors
- structural balance measures

These descriptors provide an interpretable representation of transcriptomic organization while remaining computationally straightforward.

---

# Main Results

The principal classification results are

| Representation | Macro-F1 |
|---------------|---------:|
| Classical diversity descriptors | **0.3027** |
| Classical + Gene-level APG | **0.3792** |
| Classical + Gene + Program APG (M5) | **0.5299** |

The APG representation substantially improves discrimination among immune cell populations compared with conventional diversity descriptors alone.

---

# Repository Structure

```
APG-Single-Cell-PBMC/
│
├── code/
│   ├── run_complete_analysis.py
│   ├── preprocessing/
│   ├── apg/
│   └── utilities/
│
├── data/
│
├── results/
│   └── cell_level_descriptors_and_annotations.csv.gz
│
├── tables/
│   ├── classification_comparison_3fold.csv
│   ├── cell_type_global_tests.csv
│   ├── downsampling_robustness.csv
│   └── ...
│
├── figures/
│
├── manuscript/
│   └── MANUSCRIPT_DRAFT.md
│
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Statistical Analyses

The package includes

- descriptive statistics
- effect-size analysis
- non-parametric hypothesis testing
- silhouette analysis
- cell-type comparisons
- classification benchmarks
- robustness under sequencing-depth reduction
- gene-program structural analysis

---

# Reproducibility

The repository has been designed for complete reproducibility.

Included are

- preprocessing scripts
- feature extraction
- APG descriptor generation
- statistical analysis
- manuscript tables
- publication figures

Running the main pipeline reproduces the analyses reported in the manuscript.

---

# Requirements

Typical Python packages include

- Python 3.11+
- Scanpy
- AnnData
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib

Install dependencies using

```bash
pip install -r requirements.txt
```

---

# Running the Analysis

```bash
python code/run_complete_analysis.py
```

Generated outputs will be written to the corresponding

- results/
- tables/
- figures/

directories.

---

# Citation

If you use this repository in your research, please cite the accompanying manuscript.

BibTeX and DOI information will be added upon publication.

---

# Author

**Dr. Mohammad Amir Khusru Akhtar**

Usha Martin University  
Ranchi, Jharkhand, India

Email: **akakhtar.2024@gmail.com**

---

# Correspondence

Dr. Mohammad Amir Khusru Akhtar

Email: akakhtar.2024@gmail.com

---

# License

This project is released under the **Apache License 2.0**.

See the LICENSE file for details.

---

# Acknowledgements

The authors acknowledge

- 10x Genomics for providing the public PBMC dataset.
- The open-source scientific Python ecosystem.
- Researchers contributing to reproducible single-cell bioinformatics.

---

# Disclaimer

This repository is intended for research and educational purposes. The analyses represent a methodological evaluation of APG on a publicly available single-donor benchmark dataset and should not be interpreted as clinical recommendations.
