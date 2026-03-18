import os
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# User settings
# ============================================================

# Path to the folder containing the modelling dataset
OUT_DIR = "path/to/file"

# Input dataset
DATA_PATH = os.path.join(OUT_DIR, "CLINICAL_AND_EDTRS_OCT_FINAL_DATASET.xlsx")
SHEET_NAME = None

# Analysis toggles
TARGET = "SNHL"          # "SNHL" or "eGFR"
FEATURE_MODE = "both"    # "oct", "clinical", or "both"

# Output directory for this run
OUTPUT_DIR = os.path.join(OUT_DIR, f"results_{TARGET.lower()}_{FEATURE_MODE}")

# Threshold used for eGFR classification
EGFR_THRESHOLD = 45

# Cross-validation settings
RANDOM_SEEDS = list(range(42, 62))
ROBUSTNESS_SEEDS = list(range(100, 125))
N_SPLITS = 5

# Clinical covariates used in clinical-only and combined analyses
CLINICAL_COVARIATES = [
    "Sex",
    "Gene",
    "Age",
    "base_unit",
    "allele_dose",
    "combined_pathogenicity",
    "NPHS2_modifier",
]

# Random Forest settings
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 2

# MLP settings
MLP_HIDDEN_LAYERS = (64, 32)
MLP_ALPHA = 1e-4
MLP_LR_INIT = 1e-3
MLP_MAX_ITER = 2000

# Importance settings
N_REPEATS_PERM = 20
TOP_K = 10
POOL_TOP_K = 20
SHAP_BACKGROUND = 100

# Wrapper settings for OCT or combined modes
N_SUBSET_TRIALS = 5000
SUBSET_SIZE = 10
TOP_N_SUBSETS = 100
TOP_FREQ_TABLE = 50

# Exhaustive search settings for clinical-only mode
CLINICAL_SUBSET_SIZES = [3, 4, 5, 6, 7]
TOP_N_CLINICAL_SUBSETS = 50

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# General helpers
# ============================================================

def print_section(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def yes_no_to_binary(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value == "yes":
        return 1
    if value == "no":
        return 0
    return np.nan


def load_dataset(path: str, sheet_name=None) -> pd.DataFrame:
    if sheet_name is None:
        return pd.read_excel(path)
    return pd.read_excel(path, sheet_name=sheet_name)


def encode_clinical_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_clin = [c for c in CLINICAL_COVARIATES if c not in df.columns]
    if missing_clin:
        raise ValueError(f"Missing clinical covariate columns: {missing_clin}")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})

    if "Gene" in df.columns:
        unique_genes = sorted(df["Gene"].dropna().unique())
        gene_map = {g: i for i, g in enumerate(unique_genes)}
        df["Gene"] = df["Gene"].map(gene_map)

    if "base_unit" in df.columns:
        unique_units = sorted(df["base_unit"].dropna().unique())
        unit_map = {u: i for i, u in enumerate(unique_units)}
        df["base_unit"] = df["base_unit"].map(unit_map)

    if "combined_pathogenicity" in df.columns:
        if df["combined_pathogenicity"].dtype == object:
            unique_vals = sorted(df["combined_pathogenicity"].dropna().unique())
            path_map = {v: i for i, v in enumerate(unique_vals)}
            df["combined_pathogenicity"] = df["combined_pathogenicity"].map(path_map)
        else:
            df["combined_pathogenicity"] = pd.to_numeric(
                df["combined_pathogenicity"], errors="coerce"
            )

    for col in ["Age", "allele_dose", "NPHS2_modifier"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CLINICAL_COVARIATES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Clinical covariate '{col}' is non-numeric after encoding."
            )

    return df


def prepare_labels(df: pd.DataFrame, target: str, egfr_threshold: int) -> tuple[pd.DataFrame, str]:
    df = df.copy()

    if "patient_id" not in df.columns:
        raise ValueError("Input dataset must contain 'patient_id'.")

    if target == "SNHL":
        if "SNHL" not in df.columns:
            raise ValueError("SNHL analysis requires an 'SNHL' column.")
        df["target_bin"] = df["SNHL"].apply(yes_no_to_binary)
        label_name = "SNHL"
    elif target == "eGFR":
        if "EGFR" not in df.columns:
            raise ValueError("eGFR analysis requires an 'EGFR' column.")
        df["EGFR_numeric"] = pd.to_numeric(df["EGFR"], errors="coerce")
        df["target_bin"] = np.where(
            df["EGFR_numeric"].isna(),
            np.nan,
            np.where(df["EGFR_numeric"] >= egfr_threshold, 1, 0),
        )
        label_name = f"eGFR_ge_{egfr_threshold}"
    else:
        raise ValueError("TARGET must be 'SNHL' or 'eGFR'.")

    return df, label_name


def identify_oct_feature_columns(
    df: pd.DataFrame,
    clinical_covariates: list[str],
) -> list[str]:
    excluded = {
        "patient_id",
        "id_family",
        "SNHL",
        "EGFR",
        "EGFR_numeric",
        "target_bin",
        "volume",
        "eye",
        "flipped",
        "scan_id",
    }

    oct_cols = []
    for col in df.columns:
        if col in excluded or col in clinical_covariates:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            oct_cols.append(col)

    return oct_cols


def select_feature_columns(
    oct_cols: list[str],
    clinical_cols: list[str],
    feature_mode: str,
) -> list[str]:
    if feature_mode == "oct":
        return oct_cols
    if feature_mode == "clinical":
        return clinical_cols
    if feature_mode == "both":
        return oct_cols + clinical_cols
    raise ValueError("FEATURE_MODE must be 'oct', 'clinical', or 'both'.")


def feature_type_map(feature_cols: list[str]) -> list[str]:
    return ["clinical" if f in CLINICAL_COVARIATES else "OCT" for f in feature_cols]


def assert_no_leakage(groups, train_idx, test_idx, label: str = "") -> None:
    train_ids = set(groups[train_idx])
    test_ids = set(groups[test_idx])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Patient leakage {label}: {list(overlap)[:5]}"


def impute_fold(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    medians = X_train_df.median(numeric_only=True)
    return X_train_df.fillna(medians), X_test_df.fillna(medians)


def shap_positive_class_matrix(shap_values, positive_class: int = 1) -> np.ndarray:
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[positive_class])
        if sv.ndim != 2:
            raise ValueError(f"Unexpected SHAP list element shape: {sv.shape}")
        return sv

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        return sv[:, :, positive_class]
    if sv.ndim == 2:
        return sv
    raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")


def connection_weight_importance(mlp: MLPClassifier) -> np.ndarray:
    weights = mlp.coefs_[0]
    for w in mlp.coefs_[1:]:
        weights = weights @ w
    return np.abs(weights).sum(axis=1)


def make_model(model_type: str, seed: int):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=MLP_ALPHA,
            learning_rate_init=MLP_LR_INIT,
            max_iter=MLP_MAX_ITER,
            random_state=seed,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def compute_cv_f1(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_idx: list[int],
    model_type: str,
    seed: int,
    n_splits: int = N_SPLITS,
) -> float:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_f1s = []

    for train_idx, test_idx in cv.split(X, y, groups):
        assert_no_leakage(groups, train_idx, test_idx, f"{model_type} seed={seed}")

        X_train_df = X.iloc[train_idx, feature_idx].copy()
        X_test_df = X.iloc[test_idx, feature_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_imp, X_test_imp = impute_fold(X_train_df, X_test_df)

        if model_type == "rf":
            model = make_model("rf", seed)
            model.fit(X_train_imp.values, y_train)
            y_pred = model.predict(X_test_imp.values)
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_imp.values)
            X_test_sc = scaler.transform(X_test_imp.values)

            model = make_model("mlp", seed)
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return float(np.mean(fold_f1s))


# ============================================================
# Importance analysis
# ============================================================

def run_importance_analysis(
    model_type: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    feature_cols: list[str],
    random_seeds: list[int],
    n_splits: int,
    target: str,
    verbose: bool = True,
):
    n_features = len(feature_cols)
    total_folds = len(random_seeds) * n_splits

    all_perm = []
    all_method2 = []
    all_shap_abs = []
    all_shap_signed = []

    topk_perm = np.zeros(n_features, dtype=int)
    topk_method2 = np.zeros(n_features, dtype=int)
    topk_shap = np.zeros(n_features, dtype=int)

    fold_f1s = []
    fold_counter = 0
    t_start = time.time()

    for seed in random_seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all, groups_all), 1):
            fold_counter += 1

            assert_no_leakage(
                groups_all, train_idx, test_idx, f"{model_type} seed={seed} fold={fold_idx}"
            )

            X_train_df = X_all.iloc[train_idx].copy()
            X_test_df = X_all.iloc[test_idx].copy()
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            X_train_imp, X_test_imp = impute_fold(X_train_df, X_test_df)

            if model_type == "rf":
                model = make_model("rf", seed)
                model.fit(X_train_imp.values, y_train)
                X_train_eval = X_train_imp.values
                X_test_eval = X_test_imp.values
            else:
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train_imp.values)
                X_test_sc = scaler.transform(X_test_imp.values)

                model = make_model("mlp", seed)
                model.fit(X_train_sc, y_train)
                X_train_eval = X_train_sc
                X_test_eval = X_test_sc

            y_pred = model.predict(X_test_eval)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fold_f1s.append(f1)

            perm = permutation_importance(
                model,
                X_test_eval,
                y_test,
                scoring="f1",
                n_repeats=N_REPEATS_PERM,
                random_state=seed,
                n_jobs=-1,
            )
            perm_imp = perm.importances_mean
            all_perm.append(perm_imp)
            topk_perm[np.argsort(perm_imp)[-TOP_K:]] += 1

            if model_type == "rf":
                method2_imp = model.feature_importances_
            else:
                method2_imp = connection_weight_importance(model)

            all_method2.append(method2_imp)
            topk_method2[np.argsort(method2_imp)[-TOP_K:]] += 1

            rng = np.random.RandomState(seed * 10_000 + fold_idx)
            bg_size = min(SHAP_BACKGROUND, X_train_eval.shape[0])
            bg_idx = rng.choice(X_train_eval.shape[0], size=bg_size, replace=False)
            background = X_train_eval[bg_idx]

            if model_type == "rf":
                explainer = shap.TreeExplainer(model, background)
                shap_raw = explainer.shap_values(X_test_eval)
                if isinstance(shap_raw, list):
                    sv_pos = np.asarray(shap_raw[1])
                else:
                    sv_pos = shap_positive_class_matrix(shap_raw, positive_class=1)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_raw = explainer.shap_values(X_test_eval, nsamples="auto", silent=True)
                sv_pos = shap_positive_class_matrix(shap_raw, positive_class=1)

            mean_abs_shap = np.abs(sv_pos).mean(axis=0)
            mean_signed_shap = sv_pos.mean(axis=0)

            all_shap_abs.append(mean_abs_shap)
            all_shap_signed.append(mean_signed_shap)
            topk_shap[np.argsort(mean_abs_shap)[-TOP_K:]] += 1

            if verbose and (fold_counter % 10 == 0 or fold_counter == total_folds):
                elapsed = time.time() - t_start
                rate = elapsed / fold_counter
                remaining = rate * (total_folds - fold_counter)
                print(
                    f"  [{fold_counter:3d}/{total_folds}] "
                    f"{model_type.upper()} seed={seed} fold={fold_idx} | "
                    f"F1={f1:.3f} | Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min"
                )

    method2_label = "MDI" if model_type == "rf" else "CW"

    perm_arr = np.vstack(all_perm)
    method2_arr = np.vstack(all_method2)
    shap_abs_arr = np.vstack(all_shap_abs)
    shap_signed_arr = np.vstack(all_shap_signed)

    ftypes = feature_type_map(feature_cols)

    if target == "SNHL":
        direction_label_pos = "increases SNHL probability"
        direction_label_neg = "decreases SNHL probability"
    else:
        direction_label_pos = f"increases eGFR>={EGFR_THRESHOLD} probability"
        direction_label_neg = f"decreases eGFR>={EGFR_THRESHOLD} probability"

    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "feature_type": ftypes,
        "mean_perm_imp": perm_arr.mean(axis=0),
        "std_perm_imp": perm_arr.std(axis=0),
        "topk_freq": topk_perm,
        "topk_prop": topk_perm / total_folds,
    }).sort_values("mean_perm_imp", ascending=False).reset_index(drop=True)

    method2_df = pd.DataFrame({
        "feature": feature_cols,
        "feature_type": ftypes,
        f"mean_{method2_label.lower()}_imp": method2_arr.mean(axis=0),
        f"std_{method2_label.lower()}_imp": method2_arr.std(axis=0),
        "topk_freq": topk_method2,
        "topk_prop": topk_method2 / total_folds,
    }).sort_values(f"mean_{method2_label.lower()}_imp", ascending=False).reset_index(drop=True)

    shap_mean_signed = shap_signed_arr.mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "feature_type": ftypes,
        "mean_abs_shap": shap_abs_arr.mean(axis=0),
        "std_abs_shap": shap_abs_arr.std(axis=0),
        "mean_signed_shap": shap_mean_signed,
        "direction": np.where(shap_mean_signed > 0, direction_label_pos, direction_label_neg),
        "topk_freq": topk_shap,
        "topk_prop": topk_shap / total_folds,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return {
        "perm_df": perm_df,
        "m2_df": method2_df,
        "m2_label": method2_label,
        "shap_df": shap_df,
        "fold_f1s": fold_f1s,
    }


def build_summary_table(results: dict) -> pd.DataFrame:
    perm_ranked = results["perm_df"][["feature", "feature_type"]].copy()
    perm_ranked["perm_rank"] = range(1, len(perm_ranked) + 1)

    method2_label = results["m2_label"]
    method2_ranked = results["m2_df"][["feature"]].copy()
    method2_ranked[f"{method2_label}_rank"] = range(1, len(method2_ranked) + 1)

    shap_ranked = results["shap_df"][["feature"]].copy()
    shap_ranked["SHAP_rank"] = range(1, len(shap_ranked) + 1)

    combined = perm_ranked.merge(method2_ranked, on="feature").merge(shap_ranked, on="feature")
    rank_cols = ["perm_rank", f"{method2_label}_rank", "SHAP_rank"]
    combined["mean_rank"] = combined[rank_cols].mean(axis=1)

    return combined.sort_values("mean_rank").reset_index(drop=True)


def build_feature_pool(results: dict, top_k: int = POOL_TOP_K) -> list[str]:
    top_perm = set(results["perm_df"].head(top_k)["feature"])
    top_method2 = set(results["m2_df"].head(top_k)["feature"])
    top_shap = set(results["shap_df"].head(top_k)["feature"])
    return sorted(top_perm | top_method2 | top_shap)


def run_subset_trials(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    pool_features: list[str],
    feature_cols: list[str],
    model_type: str,
    n_trials: int,
    subset_size: int,
    seed_base: int = 42,
) -> pd.DataFrame:
    pool_idx = [feature_cols.index(f) for f in pool_features]
    rng = np.random.RandomState(seed_base)
    results = []

    t_start = time.time()
    for trial in range(n_trials):
        chosen = rng.choice(pool_idx, size=subset_size, replace=False)
        chosen_sorted = tuple(sorted(chosen))

        mean_f1 = compute_cv_f1(
            X_all,
            y_all,
            groups_all,
            feature_idx=list(chosen_sorted),
            model_type=model_type,
            seed=seed_base,
            n_splits=N_SPLITS,
        )

        results.append({
            "subset_id": trial,
            "features": chosen_sorted,
            "mean_f1": mean_f1,
        })

        if (trial + 1) % 500 == 0 or trial == n_trials - 1:
            elapsed = time.time() - t_start
            rate = elapsed / (trial + 1)
            remaining = rate * (n_trials - trial - 1)
            print(
                f"  [{trial+1:5d}/{n_trials}] "
                f"{model_type.upper()} subset trials | Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min"
            )

    return pd.DataFrame(results).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def run_robustness(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    top_subsets_df: pd.DataFrame,
    model_type: str,
    robustness_seeds: list[int],
) -> pd.DataFrame:
    records = []
    n_subsets = len(top_subsets_df)
    t_start = time.time()

    for i, row in top_subsets_df.iterrows():
        feature_idx = list(row["features"])
        seed_f1s = []

        for seed in robustness_seeds:
            mean_f1 = compute_cv_f1(
                X_all,
                y_all,
                groups_all,
                feature_idx=feature_idx,
                model_type=model_type,
                seed=seed,
                n_splits=N_SPLITS,
            )
            seed_f1s.append(mean_f1)

        records.append({
            "subset_rank": i,
            "features": row["features"],
            "stage1_f1": row["mean_f1"],
            "robust_mean_f1": np.mean(seed_f1s),
            "robust_std_f1": np.std(seed_f1s),
            "robust_min_f1": np.min(seed_f1s),
            "robust_max_f1": np.max(seed_f1s),
        })

        if (i + 1) % 10 == 0 or i == n_subsets - 1:
            elapsed = time.time() - t_start
            rate = elapsed / (i + 1)
            remaining = rate * (n_subsets - i - 1)
            print(
                f"  [{i+1:3d}/{n_subsets}] "
                f"{model_type.upper()} robustness | Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min"
            )

    return pd.DataFrame(records).sort_values("robust_mean_f1", ascending=False).reset_index(drop=True)


def feature_frequency_table(
    robust_df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = TOP_FREQ_TABLE,
) -> pd.DataFrame:
    top = robust_df.head(top_n)
    counts = np.zeros(len(feature_cols), dtype=int)

    for _, row in top.iterrows():
        for idx in row["features"]:
            counts[idx] += 1

    ftypes = feature_type_map(feature_cols)

    out = pd.DataFrame({
        "feature": feature_cols,
        "feature_type": ftypes,
        f"count_in_top{top_n}": counts,
        f"proportion_in_top{top_n}": counts / top_n,
    })

    out = out[out[f"count_in_top{top_n}"] > 0]
    return out.sort_values(f"count_in_top{top_n}", ascending=False).reset_index(drop=True)


# ============================================================
# Exhaustive subset search for clinical-only mode
# ============================================================

def run_exhaustive_subset_search(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    feature_cols: list[str],
    model_type: str,
    subset_sizes: list[int],
    stage1_seed: int = 42,
) -> pd.DataFrame:
    all_results = []
    subset_id = 0

    combos = []
    for k in subset_sizes:
        combos.extend(list(combinations(range(len(feature_cols)), k)))

    total = len(combos)
    t_start = time.time()

    for i, combo in enumerate(combos, 1):
        mean_f1 = compute_cv_f1(
            X_all,
            y_all,
            groups_all,
            feature_idx=list(combo),
            model_type=model_type,
            seed=stage1_seed,
            n_splits=N_SPLITS,
        )

        all_results.append({
            "subset_id": subset_id,
            "subset_size": len(combo),
            "features": tuple(combo),
            "mean_f1": mean_f1,
        })
        subset_id += 1

        if i % 20 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = elapsed / i
            remaining = rate * (total - i)
            print(
                f"  [{i:3d}/{total}] {model_type.upper()} exhaustive subsets | "
                f"Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min"
            )

    return pd.DataFrame(all_results).sort_values("mean_f1", ascending=False).reset_index(drop=True)


# ============================================================
# Output helpers
# ============================================================

def print_top_features(df: pd.DataFrame, col: str, label: str, n: int = 10) -> None:
    print(f"\n  Top {n} by {label}:")
    top = df.head(n)
    for i, row in top.iterrows():
        tag = " [CLINICAL]" if row.get("feature_type", "") == "clinical" else ""
        print(f"    {i+1:2d}. {row['feature']:<45s} {row[col]:.6f}{tag}")


def save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def format_subset_output(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["feature_names"] = out["features"].apply(
        lambda tup: "; ".join(feature_cols[i] for i in tup)
    )
    return out.drop(columns=["features"])


# ============================================================
# Core pipeline
# ============================================================

def run_model_pipeline(
    model_type: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    feature_cols: list[str],
    target: str,
    feature_mode: str,
):
    print_section(f"{model_type.upper()} analysis")

    importance_results = None
    summary_df = None
    pool_features = None
    stage1_df = None
    robust_df = None
    best_features = None
    freq_df = None

    if feature_mode == "clinical":
        print_section(f"{model_type.upper()} exhaustive subset search", "-")
        stage1_df = run_exhaustive_subset_search(
            X_all=X_all,
            y_all=y_all,
            groups_all=groups_all,
            feature_cols=feature_cols,
            model_type=model_type,
            subset_sizes=CLINICAL_SUBSET_SIZES,
            stage1_seed=42,
        )

        top_stage1 = stage1_df.head(TOP_N_CLINICAL_SUBSETS)
        print_section(f"{model_type.upper()} robustness evaluation", "-")
        robust_df = run_robustness(
            X_all=X_all,
            y_all=y_all,
            groups_all=groups_all,
            top_subsets_df=top_stage1,
            model_type=model_type,
            robustness_seeds=ROBUSTNESS_SEEDS,
        )

        best_features = [feature_cols[i] for i in robust_df.iloc[0]["features"]]
        freq_df = feature_frequency_table(robust_df, feature_cols, top_n=min(TOP_FREQ_TABLE, len(robust_df)))

    else:
        print_section(f"{model_type.upper()} importance analysis", "-")
        importance_results = run_importance_analysis(
            model_type=model_type,
            X_all=X_all,
            y_all=y_all,
            groups_all=groups_all,
            feature_cols=feature_cols,
            random_seeds=RANDOM_SEEDS,
            n_splits=N_SPLITS,
            target=target,
        )

        method2_col = "mean_mdi_imp" if model_type == "rf" else "mean_cw_imp"
        print_top_features(importance_results["perm_df"], "mean_perm_imp", "Permutation importance")
        print_top_features(importance_results["m2_df"], method2_col, importance_results["m2_label"])
        print_top_features(importance_results["shap_df"], "mean_abs_shap", "SHAP importance")

        summary_df = build_summary_table(importance_results)

        print_section(f"{model_type.upper()} feature pool", "-")
        pool_features = build_feature_pool(importance_results, top_k=POOL_TOP_K)
        clinical_in_pool = [f for f in pool_features if f in CLINICAL_COVARIATES]
        print(f"  Pool size: {len(pool_features)}")
        print(f"  Clinical covariates in pool: {len(clinical_in_pool)} → {clinical_in_pool}")

        print_section(f"{model_type.upper()} random subset evaluation", "-")
        stage1_df = run_subset_trials(
            X_all=X_all,
            y_all=y_all,
            groups_all=groups_all,
            pool_features=pool_features,
            feature_cols=feature_cols,
            model_type=model_type,
            n_trials=N_SUBSET_TRIALS,
            subset_size=SUBSET_SIZE,
            seed_base=42,
        )

        top_stage1 = stage1_df.head(TOP_N_SUBSETS)
        print_section(f"{model_type.upper()} robustness evaluation", "-")
        robust_df = run_robustness(
            X_all=X_all,
            y_all=y_all,
            groups_all=groups_all,
            top_subsets_df=top_stage1,
            model_type=model_type,
            robustness_seeds=ROBUSTNESS_SEEDS,
        )

        best_features = [feature_cols[i] for i in robust_df.iloc[0]["features"]]
        freq_df = feature_frequency_table(robust_df, feature_cols, top_n=min(TOP_FREQ_TABLE, len(robust_df)))

    return {
        "importance_results": importance_results,
        "summary_df": summary_df,
        "pool_features": pool_features,
        "stage1_df": stage1_df,
        "robust_df": robust_df,
        "best_features": best_features,
        "freq_df": freq_df,
    }


def build_overlap_table(
    rf_results: dict,
    mlp_results: dict,
    feature_mode: str,
) -> pd.DataFrame:
    rf_best_set = set(rf_results["best_features"])
    mlp_best_set = set(mlp_results["best_features"])

    if feature_mode == "clinical":
        all_features = sorted(rf_best_set | mlp_best_set)
        overlap_df = pd.DataFrame({
            "feature": all_features,
            "feature_type": ["clinical" if f in CLINICAL_COVARIATES else "OCT" for f in all_features],
            "RF_best_subset": [1 if f in rf_best_set else 0 for f in all_features],
            "MLP_best_subset": [1 if f in mlp_best_set else 0 for f in all_features],
        })
        overlap_df["total"] = overlap_df[["RF_best_subset", "MLP_best_subset"]].sum(axis=1)
        return overlap_df.sort_values("total", ascending=False).reset_index(drop=True)

    rf_pool_set = set(rf_results["pool_features"])
    mlp_pool_set = set(mlp_results["pool_features"])
    all_features = sorted(rf_pool_set | mlp_pool_set | rf_best_set | mlp_best_set)

    overlap_df = pd.DataFrame({
        "feature": all_features,
        "feature_type": ["clinical" if f in CLINICAL_COVARIATES else "OCT" for f in all_features],
        "RF_top20_pool": [1 if f in rf_pool_set else 0 for f in all_features],
        "MLP_top20_pool": [1 if f in mlp_pool_set else 0 for f in all_features],
        "RF_best_subset": [1 if f in rf_best_set else 0 for f in all_features],
        "MLP_best_subset": [1 if f in mlp_best_set else 0 for f in all_features],
    })
    overlap_df["total"] = overlap_df[
        ["RF_top20_pool", "MLP_top20_pool", "RF_best_subset", "MLP_best_subset"]
    ].sum(axis=1)

    return overlap_df.sort_values("total", ascending=False).reset_index(drop=True)


# ============================================================
# Main script
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_section("MODELLING PIPELINE")
    print(f"Target:        {TARGET}")
    print(f"Feature mode:  {FEATURE_MODE}")
    print(f"Output dir:    {OUTPUT_DIR}")

    df = load_dataset(DATA_PATH, sheet_name=SHEET_NAME)
    df = encode_clinical_covariates(df)
    df, _ = prepare_labels(df, TARGET, EGFR_THRESHOLD)

    oct_feature_cols = identify_oct_feature_columns(df, CLINICAL_COVARIATES)

    feature_cols = select_feature_columns(
        oct_cols=oct_feature_cols,
        clinical_cols=CLINICAL_COVARIATES,
        feature_mode=FEATURE_MODE,
    )

    df_model = df[df["target_bin"].isin([0, 1])].copy()

    print_section("DATA SUMMARY", "-")
    print(f"Rows:             {len(df_model)}")
    print(f"Unique patients:  {df_model['patient_id'].nunique()}")
    print(f"Total features:   {len(feature_cols)}")
    print(f"Class counts:\n{df_model['target_bin'].value_counts(dropna=False)}")

    X_all = df_model[feature_cols].copy()
    y_all = df_model["target_bin"].astype(int).values
    groups_all = df_model["patient_id"].values

    rf_results = run_model_pipeline(
        model_type="rf",
        X_all=X_all,
        y_all=y_all,
        groups_all=groups_all,
        feature_cols=feature_cols,
        target=TARGET,
        feature_mode=FEATURE_MODE,
    )

    mlp_results = run_model_pipeline(
        model_type="mlp",
        X_all=X_all,
        y_all=y_all,
        groups_all=groups_all,
        feature_cols=feature_cols,
        target=TARGET,
        feature_mode=FEATURE_MODE,
    )

    overlap_df = build_overlap_table(rf_results, mlp_results, FEATURE_MODE)

    rf_best = rf_results["robust_df"].iloc[0]
    mlp_best = mlp_results["robust_df"].iloc[0]

    print_section("FINAL SUMMARY")
    print(f"{'Model':<8s} {'Robust Mean F1':>16s} {'± Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 55)
    print(
        f"{'RF':<8s} {rf_best['robust_mean_f1']:>16.4f} "
        f"{rf_best['robust_std_f1']:>8.4f} "
        f"{rf_best['robust_min_f1']:>8.4f} "
        f"{rf_best['robust_max_f1']:>8.4f}"
    )
    print(
        f"{'MLP':<8s} {mlp_best['robust_mean_f1']:>16.4f} "
        f"{mlp_best['robust_std_f1']:>8.4f} "
        f"{mlp_best['robust_min_f1']:>8.4f} "
        f"{mlp_best['robust_max_f1']:>8.4f}"
    )

    print("\nBest RF subset:")
    for feat in rf_results["best_features"]:
        tag = " [CLINICAL]" if feat in CLINICAL_COVARIATES else ""
        print(f"  • {feat}{tag}")

    print("\nBest MLP subset:")
    for feat in mlp_results["best_features"]:
        tag = " [CLINICAL]" if feat in CLINICAL_COVARIATES else ""
        print(f"  • {feat}{tag}")

    print_section("SAVING RESULTS")

    prefix = f"{TARGET.lower()}_{FEATURE_MODE}"

    # RF outputs
    if rf_results["importance_results"] is not None:
        save_csv(rf_results["importance_results"]["perm_df"], OUTPUT_DIR, f"{prefix}_rf_permutation_importance.csv")
        save_csv(rf_results["importance_results"]["m2_df"], OUTPUT_DIR, f"{prefix}_rf_method2_importance.csv")
        save_csv(rf_results["importance_results"]["shap_df"], OUTPUT_DIR, f"{prefix}_rf_shap_importance.csv")
        save_csv(rf_results["summary_df"], OUTPUT_DIR, f"{prefix}_rf_combined_ranks.csv")

    save_csv(format_subset_output(rf_results["stage1_df"], feature_cols), OUTPUT_DIR, f"{prefix}_rf_stage1_subsets.csv")
    save_csv(format_subset_output(rf_results["robust_df"], feature_cols), OUTPUT_DIR, f"{prefix}_rf_robust_subsets.csv")
    save_csv(rf_results["freq_df"], OUTPUT_DIR, f"{prefix}_rf_feature_frequency.csv")

    # MLP outputs
    if mlp_results["importance_results"] is not None:
        save_csv(mlp_results["importance_results"]["perm_df"], OUTPUT_DIR, f"{prefix}_mlp_permutation_importance.csv")
        save_csv(mlp_results["importance_results"]["m2_df"], OUTPUT_DIR, f"{prefix}_mlp_method2_importance.csv")
        save_csv(mlp_results["importance_results"]["shap_df"], OUTPUT_DIR, f"{prefix}_mlp_shap_importance.csv")
        save_csv(mlp_results["summary_df"], OUTPUT_DIR, f"{prefix}_mlp_combined_ranks.csv")

    save_csv(format_subset_output(mlp_results["stage1_df"], feature_cols), OUTPUT_DIR, f"{prefix}_mlp_stage1_subsets.csv")
    save_csv(format_subset_output(mlp_results["robust_df"], feature_cols), OUTPUT_DIR, f"{prefix}_mlp_robust_subsets.csv")
    save_csv(mlp_results["freq_df"], OUTPUT_DIR, f"{prefix}_mlp_feature_frequency.csv")

    save_csv(overlap_df, OUTPUT_DIR, f"{prefix}_feature_overlap_table.csv")

    if FEATURE_MODE != "clinical":
        pool_features = sorted(set((rf_results["pool_features"] or []) + (mlp_results["pool_features"] or [])))
        pool_df = pd.DataFrame({
            "feature": pool_features,
            "feature_type": ["clinical" if f in CLINICAL_COVARIATES else "OCT" for f in pool_features],
            "in_RF_pool": [1 if f in set(rf_results["pool_features"] or []) else 0 for f in pool_features],
            "in_MLP_pool": [1 if f in set(mlp_results["pool_features"] or []) else 0 for f in pool_features],
        })
        save_csv(pool_df, OUTPUT_DIR, f"{prefix}_feature_pools.csv")

    print(f"\nAll done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
