# alport-oct-dissertation-code

## Overview
This repository contains the code developed for a final-year project investigating whether OCT-derived retinal features can predict systemic manifestations of Alport syndrome.

The pipeline extracts ETDRS-based features from retinal thickness maps, performs preprocessing and statistical analysis, and applies machine learning models to evaluate predictive performance and feature importance.

---

## Repository Structure

### `etdrs_feature_extraction.py`
Extracts ETDRS-grid features from OCT-derived retinal thickness maps.

**Key steps:**
- Loads layer-wise thickness maps (`.npy` files)
- Estimates foveal centre from total thickness map
- Applies ETDRS sector masks (central, inner, outer)
- Computes mean thickness values across 9 sectors
- Outputs **90 features per scan** (9 layers × 10 regions)

---

### `preprocessing.py`
Prepares the extracted features for analysis.

**Key steps:**
- Loads feature dataset
- Cleans and standardises data
- Handles missing values
- Ensures consistent formatting across patients and volumes

---

### `statistical_analysis.py`
Performs statistical comparisons of OCT features across clinical groups.

**Key steps:**
- Converts clinical labels to binary variables (e.g. SNHL, eGFR threshold)
- Computes **point-biserial correlations** (effect size)
- Performs **Mann–Whitney U tests**
- Aggregates results at:
  - Volume level
  - Patient level
- Produces layer-level summary statistics

---

### `models.py`
Main modelling pipeline for predictive analysis.

**Supported configurations:**
- **Targets:**
  - SNHL (hearing loss)
  - eGFR classification
- **Feature modes:**
  - OCT-only
  - Clinical-only
  - Combined OCT + clinical

**Models implemented:**
- Random Forest
- Multi-layer Perceptron (MLP)

**Pipeline components:**
- Grouped cross-validation (**patient-level separation**)
- Feature importance methods:
  - Permutation importance
  - Model-based importance (MDI / connection weights)
  - SHAP values
- Feature selection:
  - Importance-based feature pool
  - Random subset search (OCT/combined)
  - Exhaustive subset search (clinical-only)
- Robustness analysis across multiple random seeds

**Outputs:**
- Feature importance tables
- Ranked feature summaries
- Optimal feature subsets
- Model performance metrics

---

## How to Use

1. Set your data directory in each script:
```python
OUT_DIR = "path/to/your/data"
```

2. Place your dataset in this directory.

3. Run the pipeline in order:

```
etdrs_feature_extraction.py
preprocessing.py
statistical_analysis.py
models.py
```

4. Outputs will be saved automatically to structured result folders.

---

## Key Assumptions

- OCT data has already been segmented into retinal layers
- Thickness maps are stored as `.npy` arrays
- Clinical dataset includes:
  - `patient_id`
  - SNHL status (Yes/No)
  - eGFR values
  - Additional clinical covariates
- **Patient-level grouping is enforced** to prevent data leakage
- Left eyes are flipped horizontally to standardise anatomical orientation

---

## Notes

- No patient-identifiable information is included in this repository
- File paths must be updated locally before running
- The modelling pipeline is computationally intensive (especially SHAP and subset search)
