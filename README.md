Overview

This repository contains the code developed for a final-year project investigating whether OCT-derived retinal features can predict systemic manifestations of Alport syndrome.

The pipeline extracts ETDRS-based features from retinal thickness maps, performs preprocessing and statistical analysis, and applies machine learning models to evaluate predictive performance and feature importance.

Repository Structure
etdrs_feature_extraction.py

Extracts ETDRS-grid features from OCT-derived retinal thickness maps.

Key steps:

Loads layer-wise thickness maps (NumPy arrays)

Estimates foveal centre from total thickness

Applies ETDRS sector masks (central, inner, outer)

Computes mean thickness values across 9 sectors

Outputs 90 features per scan (9 layers × 10 regions)

preprocessing.py

Prepares the extracted features for downstream analysis.

Key steps:

Loads feature dataset

Cleans missing values

Standardises formatting and structure

Ensures consistency across volumes and patients

statistical_analysis.py

Performs statistical comparison of OCT features across clinical groups.

Key steps:

Converts clinical labels to binary variables (e.g. SNHL, eGFR threshold)

Computes point-biserial correlations (effect sizes)

Performs Mann–Whitney U tests

Aggregates results at both volume-level and patient-level

Produces layer-level summary statistics

models.py

Main modelling pipeline for predictive analysis.

Key features:

Supports multiple analysis modes:

OCT-only

Clinical-only

Combined OCT + clinical features

Supports multiple targets:

SNHL (hearing loss)

eGFR classification

Methods implemented:

Random Forest classifier

Multi-layer Perceptron (MLP)

Pipeline components:

Grouped cross-validation (patient-level separation)

Feature importance:

Permutation importance

Model-specific importance (MDI / connection weights)

SHAP values

Feature selection:

Importance-based feature pool

Random subset search (OCT/combined)

Exhaustive subset search (clinical-only)

Robustness analysis across multiple random seeds

Outputs:

Feature importance tables

Ranked feature summaries

Optimal feature subsets

Model performance metrics

How to Use

Set the base directory in each script:

OUT_DIR = "path/to/your/data"

Ensure the input dataset is placed in this directory.

Run the scripts in order:

1. etdrs_feature_extraction.py
2. preprocessing.py
3. statistical_analysis.py
4. models.py

Results will be saved automatically to structured output folders.

Key Assumptions

Input OCT data has already been segmented into retinal layers.

Thickness maps are stored as .npy arrays.

Clinical dataset includes:

Patient identifiers (patient_id)

SNHL status (Yes/No)

eGFR values

Additional clinical covariates

Patient-level grouping is required to avoid data leakage (enforced via grouped cross-validation).

Left eyes are horizontally flipped to standardise anatomical orientation.

Notes

No patient-identifiable information is included in this repository.

File paths must be updated locally before running the scripts.

The modelling pipeline is computationally intensive, particularly when running SHAP and subset selection stages.
