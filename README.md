# Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21306503.svg)](https://doi.org/10.5281/zenodo.21306503)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This repository contains the complete reproducible research package accompanying the paper:

> **Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes**

The study introduces **Arithmetic Power Geometry (APG)** as an interpretable mathematical framework for compact structural characterization of single-cell RNA sequencing (scRNA-seq) transcriptomes. APG complements conventional diversity and concentration measures by representing each transcriptome using deterministic structural descriptors derived from normalized quadratic weights.

The repository includes:

- Complete analysis pipeline
- Source code
- Statistical analysis
- Figures
- Tables
- Supplementary material
- Reproducible manuscript

---

## Paper

**Author**

**Dr. Mohammad Amir Khusru Akhtar**

Usha Martin University  
Ranchi – 834001, Jharkhand, India

📧 akakhtar.2024@gmail.com

**DOI**

https://doi.org/10.5281/zenodo.21306503

---

## Abstract

Single-cell RNA sequencing produces high-dimensional molecular profiles, whereas compact cell-level summaries often reduce each transcriptome to one diversity or concentration statistic.

Arithmetic Power Geometry (APG) introduces a deterministic structural representation based on normalized quadratic weights. From these weights APG derives entropy, concentration, effective dimension, bounded exponent-deformation profiles, and integrated deformation, providing complementary structural information beyond classical diversity measures.

The framework is evaluated on the official **10x Genomics Fresh 68k PBMC (Donor A)** dataset containing **68,579 cells**, where APG is compared directly with Shannon entropy, Rényi entropy, Simpson diversity, Gini inequality, Herfindahl concentration, participation ratio, and detected-gene count.

Results demonstrate that APG descriptors substantially improve compact transcriptomic representation while remaining fully interpretable and computationally efficient.

---

# Key Contributions

- Introduces Arithmetic Power Geometry (APG) for single-cell transcriptomics.
- Defines deterministic structural descriptors using normalized quadratic weights.
- Compares APG directly with established diversity and concentration measures.
- Evaluates both gene-level and biological programme-level APG.
- Benchmarks APG using the official 68k PBMC dataset.
- Provides complete reproducible code and analysis.
- Releases all figures, tables and supplementary material.

---

# Mathematical Framework

For each cell, APG computes normalized quadratic weights

```
             xᵢ²
wᵢ = ------------------
      Σⱼ (xⱼ²)
```

where

- xᵢ denotes normalized gene expression
- wᵢ denotes the APG structural weight

From these weights APG computes

- Structural entropy
- Normalized entropy
- Maximum concentration
- Quadratic concentration
- Effective dimension
- Exponent deformation D(p)
- Integrated deformation A(2,6)

---

# Dataset

Official public dataset

**10x Genomics Fresh 68k PBMCs (Donor A)**

Dataset summary

| Item | Value |
|------|------:|
| Cells | 68,579 |
| Genes | 32,738 |
| Median genes/cell | 525 |
| Median UMI counts | 1,292 |
| Mean reads/cell | 20,491 |
| Reference genome | hg19 |
| Cell Ranger | 1.1.0 |

The original dataset is publicly available from

https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0

---

# Main Results

## Classification Performance

| Representation | Macro-F1 |
|---------------|---------:|
| Shannon only | 0.1710 |
| Classical descriptors | 0.3027 |
| Classical + Gene APG | 0.3792 |
| Programme APG | 0.3887 |
| Classical + Gene + Programme APG | **0.5299** |

---

## Major Findings

- APG improves transcriptomic discrimination beyond classical diversity descriptors.
- Programme-level APG produced the strongest between-cell-type effects.
- APG descriptors remained robust under sequencing-depth reduction.
- APG provides interpretable structural summaries complementary to existing single-cell workflows.

---

# Repository Structure

```
APG-Single-Cell-PBMC
│
├── code/
│
├── data/
│
├── figures/
│
├── tables/
│
├── results/
│
├── manuscript/
│
├── supplementary/
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Arithmetic-Power-Geometry/APG-Single-Cell-PBMC.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Analysis

Execute

```bash
python code/run_complete_analysis.py
```

Generated outputs will automatically appear in

- figures/
- tables/
- results/

---

# Reproducibility

This repository contains

- Complete source code
- Derived APG descriptors
- Statistical analyses
- Publication-quality figures
- Tables
- Supplementary information
- Fully reproducible workflow

All reported results can be reproduced directly from the provided scripts.

---

# Citation

If you use this repository, please cite

Akhtar, M. A. K. (2026).

**Arithmetic Power Geometry Descriptors for Interpretable Structural Analysis of Single-Cell PBMC Transcriptomes.**

Zenodo.

https://doi.org/10.5281/zenodo.21306503

---

# Author

**Dr. Mohammad Amir Khusru Akhtar**

Usha Martin University

Ranchi, Jharkhand, India

Email

akakhtar.2024@gmail.com

---

# License

Copyright © 2026

**Dr. Mohammad Amir Khusru Akhtar**

Licensed under the **Apache License 2.0**.

See the LICENSE file for details.

---

# Acknowledgements

The author gratefully acknowledges

- 10x Genomics for providing the publicly available PBMC benchmark dataset.
- The Scanpy and scientific Python communities.
- Open-source contributors whose software made this research possible.

---

# Disclaimer

This repository is intended solely for research and educational purposes.

The analyses represent a computational methodological benchmark on a publicly available single-donor dataset. APG is proposed as an interpretable structural representation for exploratory single-cell analysis and should not be interpreted as a validated clinical diagnostic or decision-support system.
