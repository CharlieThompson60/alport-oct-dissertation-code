# Performs volume-level and patient-level statistical analysis of OCT features, including binary group comparisons, directional summaries, layer-level summaries, and optional exclusion of an outlier patient. #

import os
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Configuration
# ============================================================

# Path to the folder containing the extracted feature spreadsheet
OUT_DIR = "path/to/file"

# Input file from the feature extraction step
INPUT_FILE = os.path.join(OUT_DIR, "volume_level_features_90cols.xlsx")

# Output summary file
OUTPUT_FILE = os.path.join(OUT_DIR, "statistical_analysis_summary.xlsx")

# Sheet name in the input workbook
SHEET_NAME = "formatted_90cols"

# Threshold used to binarise eGFR
EGFR_THRESHOLD = 45

# Minimum number of samples required in each group
MIN_PER_GROUP = 6

# Optional outlier patient to exclude from patient-level analysis
OUTLIER_PATIENT = 10


# ============================================================
# Helper functions
# ============================================================

def yes_no_to_binary(value):
    """
    Converts yes/no style values to binary form.
    Returns NaN for missing or unrecognised values.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()
    if value == "yes":
        return 1
    if value == "no":
        return 0
    return np.nan


def identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identifies numeric OCT feature columns by excluding metadata and label fields.
    """
    excluded_cols = {
        "scan_id",
        "patient_id",
        "SNHL",
        "SNHL_bin",
        "EGFR",
        "EGFR_bin45",
        "volume",
        "eye",
        "flipped",
    }

    return [
        col for col in df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def feature_stats_binary_groups(
    df_in: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    min_per_group: int = 6,
) -> pd.DataFrame:
    """
    Computes feature-wise statistics for a binary label.

    For each feature:
    - group means
    - mean difference (label 1 minus label 0)
    - point-biserial correlation
    - Mann–Whitney U p-value
    """
    results = []

    valid_mask = df_in[label_col].isin([0, 1])

    for feature in feature_cols:
        group_0 = df_in.loc[df_in[label_col] == 0, feature].dropna()
        group_1 = df_in.loc[df_in[label_col] == 1, feature].dropna()

        if len(group_0) < min_per_group or len(group_1) < min_per_group:
            continue

        x = pd.to_numeric(df_in.loc[valid_mask, feature], errors="coerce")
        y = df_in.loc[valid_mask, label_col]

        matched = x.notna() & y.notna()
        if matched.sum() < (min_per_group * 2):
            continue

        try:
            r_pb, p_pb = stats.pointbiserialr(y[matched], x[matched])
        except Exception:
            r_pb, p_pb = np.nan, np.nan

        try:
            _, p_u = stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
        except Exception:
            p_u = np.nan

        results.append({
            "feature": feature,
            "n0": len(group_0),
            "n1": len(group_1),
            "mean_label0": float(group_0.mean()),
            "mean_label1": float(group_1.mean()),
            "mean_diff_1_minus_0": float(group_1.mean() - group_0.mean()),
            "r_pointbiserial": r_pb,
            "p_pointbiserial": p_pb,
            "p_mannwhitney": p_u,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    return out.sort_values(by="r_pointbiserial", key=np.abs, ascending=False).reset_index(drop=True)


def directional_summary(results_df: pd.DataFrame, label_name: str, total_features: int) -> pd.DataFrame:
    """
    Summarises whether features tend to be lower or higher in label = 1.
    """
    if results_df.empty:
        return pd.DataFrame([{
            "label_name": label_name,
            "n_lower_in_label1": 0,
            "n_higher_in_label1": 0,
            "n_equal": 0,
            "total_features": total_features,
            "overall_pattern": "no valid features",
        }])

    lower_count = int((results_df["mean_diff_1_minus_0"] < 0).sum())
    higher_count = int((results_df["mean_diff_1_minus_0"] > 0).sum())
    equal_count = int((results_df["mean_diff_1_minus_0"] == 0).sum())

    if lower_count > higher_count:
        pattern = "thinning predominates"
    elif higher_count > lower_count:
        pattern = "thickening predominates"
    else:
        pattern = "no consistent direction"

    return pd.DataFrame([{
        "label_name": label_name,
        "n_lower_in_label1": lower_count,
        "n_higher_in_label1": higher_count,
        "n_equal": equal_count,
        "total_features": total_features,
        "overall_pattern": pattern,
    }])


def extract_layer_name(feature_name: str) -> str:
    """
    Extracts the retinal layer name from a feature name.
    Example: Layer_GCIPL_I_UR_mean_px -> GCIPL
    """
    if feature_name.startswith("Layer_"):
        parts = feature_name.split("_")
        return parts[1] if len(parts) > 1 else "Other"

    if feature_name.startswith("Thickness_total"):
        return "Thickness_total"

    return "Other"


def layer_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarises feature-wise results at the retinal layer level.
    """
    if results_df.empty:
        return pd.DataFrame()

    df_layer = results_df.copy()
    df_layer["layer"] = df_layer["feature"].apply(extract_layer_name)

    layer_stats = []

    for layer, group in df_layer.groupby("layer"):
        total = len(group)
        lower = int((group["mean_diff_1_minus_0"] < 0).sum())
        higher = int((group["mean_diff_1_minus_0"] > 0).sum())
        mean_abs_r = float(group["r_pointbiserial"].abs().mean())

        layer_stats.append({
            "layer": layer,
            "n_features": total,
            "n_lower_in_label1": lower,
            "n_higher_in_label1": higher,
            "proportion_lower": lower / total if total > 0 else np.nan,
            "mean_abs_effect_size": mean_abs_r,
        })

    return (
        pd.DataFrame(layer_stats)
        .sort_values(by=["proportion_lower", "mean_abs_effect_size"], ascending=False)
        .reset_index(drop=True)
    )


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary outcome columns used in the statistical analysis.
    """
    df = df.copy()

    if "SNHL" in df.columns:
        df["SNHL_bin"] = df["SNHL"].apply(yes_no_to_binary)

    if "EGFR" in df.columns:
        df["EGFR"] = pd.to_numeric(df["EGFR"], errors="coerce")
        df["EGFR_bin45"] = np.where(
            df["EGFR"].notna(),
            (df["EGFR"] >= EGFR_THRESHOLD).astype(int),
            np.nan,
        )

    return df


def aggregate_to_patient_eye_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates numeric values to patient-eye level by mean.
    A clean binary eGFR label is then recreated from the aggregated eGFR.
    """
    required_cols = {"patient_id", "eye"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Patient-level aggregation requires 'patient_id' and 'eye' columns.")

    df_patient = df.groupby(["patient_id", "eye"], as_index=False).mean(numeric_only=True)

    if "EGFR" in df_patient.columns:
        df_patient["EGFR_bin45"] = np.where(
            df_patient["EGFR"].notna(),
            (df_patient["EGFR"] >= EGFR_THRESHOLD).astype(int),
            np.nan,
        )

    return df_patient


# ============================================================
# Main analysis
# ============================================================

def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    df = prepare_labels(df)

    print("Data loaded successfully")
    print("Shape:", df.shape)

    feature_cols = identify_feature_columns(df)
    total_features = len(feature_cols)

    print("Number of numeric OCT features:", total_features)

    if "SNHL_bin" in df.columns:
        print("\nSNHL counts (volume rows):")
        print(df["SNHL_bin"].value_counts(dropna=False))

    if "EGFR_bin45" in df.columns:
        print("\neGFR_bin45 counts (volume rows):")
        print(df["EGFR_bin45"].value_counts(dropna=False))

    # Volume-level analysis
    snhl_vol = feature_stats_binary_groups(df, "SNHL_bin", feature_cols, min_per_group=MIN_PER_GROUP)
    egfr_vol = feature_stats_binary_groups(df, "EGFR_bin45", feature_cols, min_per_group=MIN_PER_GROUP)

    # Patient-level aggregation
    df_patient = aggregate_to_patient_eye_level(df)
    df_patient_no_outlier = df_patient[df_patient["patient_id"] != OUTLIER_PATIENT].copy()

    print("\nPatient-level aggregated shape:", df_patient.shape)
    print(f"Patient-level rows without patient {OUTLIER_PATIENT}:", len(df_patient_no_outlier))

    snhl_pat = feature_stats_binary_groups(
        df_patient, "SNHL_bin", feature_cols, min_per_group=MIN_PER_GROUP
    )
    egfr_pat = feature_stats_binary_groups(
        df_patient, "EGFR_bin45", feature_cols, min_per_group=MIN_PER_GROUP
    )

    snhl_pat_no_outlier = feature_stats_binary_groups(
        df_patient_no_outlier, "SNHL_bin", feature_cols, min_per_group=MIN_PER_GROUP
    )
    egfr_pat_no_outlier = feature_stats_binary_groups(
        df_patient_no_outlier, "EGFR_bin45", feature_cols, min_per_group=MIN_PER_GROUP
    )

    # Directional summaries
    snhl_direction = directional_summary(
        snhl_pat, "SNHL (1 = hearing loss)", total_features
    )
    egfr_direction = directional_summary(
        egfr_pat, f"eGFR < {EGFR_THRESHOLD} vs ≥ {EGFR_THRESHOLD}", total_features
    )

    snhl_direction_no_outlier = directional_summary(
        snhl_pat_no_outlier, f"SNHL (excluding patient {OUTLIER_PATIENT})", total_features
    )
    egfr_direction_no_outlier = directional_summary(
        egfr_pat_no_outlier,
        f"eGFR < {EGFR_THRESHOLD} vs ≥ {EGFR_THRESHOLD} (excluding patient {OUTLIER_PATIENT})",
        total_features,
    )

    # Layer summaries
    snhl_layer_summary = layer_summary(snhl_pat)
    egfr_layer_summary = layer_summary(egfr_pat)
    snhl_layer_summary_no_outlier = layer_summary(snhl_pat_no_outlier)
    egfr_layer_summary_no_outlier = layer_summary(egfr_pat_no_outlier)

    print("\nTop SNHL-associated features (volume-level):")
    print(snhl_vol.head(20))

    print(f"\nTop eGFR-associated features (volume-level) [threshold = {EGFR_THRESHOLD}]:")
    print(egfr_vol.head(20))

    print("\nSNHL directional summary (patient-level):")
    print(snhl_direction)

    print("\neGFR directional summary (patient-level):")
    print(egfr_direction)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="input_data", index=False)
        df_patient.to_excel(writer, sheet_name="patient_level", index=False)
        df_patient_no_outlier.to_excel(writer, sheet_name="patient_level_no_outlier", index=False)

        snhl_vol.to_excel(writer, sheet_name="snhl_volume_level", index=False)
        egfr_vol.to_excel(writer, sheet_name="egfr_volume_level", index=False)

        snhl_pat.to_excel(writer, sheet_name="snhl_patient_level", index=False)
        egfr_pat.to_excel(writer, sheet_name="egfr_patient_level", index=False)

        snhl_pat_no_outlier.to_excel(writer, sheet_name="snhl_patient_no_outlier", index=False)
        egfr_pat_no_outlier.to_excel(writer, sheet_name="egfr_patient_no_outlier", index=False)

        snhl_direction.to_excel(writer, sheet_name="snhl_direction", index=False)
        egfr_direction.to_excel(writer, sheet_name="egfr_direction", index=False)

        snhl_direction_no_outlier.to_excel(writer, sheet_name="snhl_direction_no_outlier", index=False)
        egfr_direction_no_outlier.to_excel(writer, sheet_name="egfr_direction_no_outlier", index=False)

        snhl_layer_summary.to_excel(writer, sheet_name="snhl_layer_summary", index=False)
        egfr_layer_summary.to_excel(writer, sheet_name="egfr_layer_summary", index=False)

        snhl_layer_summary_no_outlier.to_excel(writer, sheet_name="snhl_layer_no_outlier", index=False)
        egfr_layer_summary_no_outlier.to_excel(writer, sheet_name="egfr_layer_no_outlier", index=False)

    print("\nSaved statistical summary to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
