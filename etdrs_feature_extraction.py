import os
import glob
import re
import hashlib
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

# Path to folder containing processed OCT volumes (update this to your local path)
OUT_DIR = "path/to/all_output"

# Output Excel file
OUTPUT_FILE = os.path.join(OUT_DIR, "volume_level_features_90cols.xlsx")

# Optional private CSV for laterality overrides.
# Expected columns: patient_id, volume, eye
LATERALITY_OVERRIDE_CSV = None

SECTORS = ["C", "I_UR", "I_UL", "I_LR", "I_LL", "O_UR", "O_UL", "O_LR", "O_LL"]
SECTOR_COLUMNS = [f"{s}_mean_px" for s in SECTORS]

LAYER_ORDER = ["GCIPL", "INL", "mRNFL", "MZ", "OL", "ONL", "OPL", "RPE", "Thickness_total"]

HEIGHT_MIN = 50
CENTRE_FRAC = 0.35

TARGET_SIZE = 224
FOV_MM = 6.0
PX_PER_MM = TARGET_SIZE / FOV_MM

R_C_MM = 0.5
R_IN_MM = 1.5
R_OUT_MM = 3.0


# ============================================================
# Helper functions for identifiers and metadata
# ============================================================

def extract_patient_id(volume_label: str) -> str:
    """
    Extracts a patient identifier from a folder label.
    This is only used internally for metadata handling and is
    not written to the public output table.
    """
    s = (volume_label or "").strip()
    if not s:
        return "UNKNOWN"

    m = re.match(r"^(\d+)", s)
    if m:
        return m.group(1)

    m = re.match(r"^(ALPORT_\d+)", s, flags=re.IGNORECASE)
    if m:
        return s[:len(m.group(1))]

    m = re.match(r"^(Alport\d+)", s, flags=re.IGNORECASE)
    if m:
        return s[:len(m.group(1))]

    first = s.split()[0].strip()
    return first if first else s


def extract_volume_index(volume_label: str):
    """
    Extracts the volume index from a label such as '.e2e_vol0'.
    Returns None if no volume index is found.
    """
    m = re.search(r"\.e2e_vol(\d+)", (volume_label or ""), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def canonical_patient_id(patient_id: str) -> str:
    """
    Normalises patient IDs so that optional private override files
    can match reliably even if folder labels are inconsistent.
    """
    s = (patient_id or "").strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1)
    return s.upper() if s else "UNKNOWN"


def coerce_duplicate_numeric_id(patient_id: str) -> str:
    """
    Reduces repeated numeric IDs such as '3030' to '30' if the
    string is formed by the same numeric token repeated twice.
    """
    s = patient_id or ""
    if s.isdigit() and len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half] == s[half:]:
            return s[:half]
    return s


def load_laterality_overrides(csv_path: str | None) -> dict:
    """
    Loads optional laterality overrides from a private CSV file.

    Expected columns:
        patient_id, volume, eye

    Example row:
        15,0,L

    Returns a dictionary keyed by (canonical_patient_id, volume_index).
    """
    if not csv_path:
        return {}

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Override file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"patient_id", "volume", "eye"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Override CSV is missing columns: {sorted(missing)}")

    overrides = {}
    for _, row in df.iterrows():
        pid = coerce_duplicate_numeric_id(canonical_patient_id(str(row["patient_id"])))
        vol = int(row["volume"])
        eye = str(row["eye"]).strip().upper()

        if eye not in {"L", "R"}:
            continue

        overrides[(pid, vol)] = eye

    return overrides


def read_laterality_from_meta_regex(volume_folder: str):
    """
    Reads laterality from meta.json using a regex.
    Returns (eye, status), where eye is 'L', 'R', or None.
    """
    meta_path = os.path.join(volume_folder, "meta.json")
    if not os.path.isfile(meta_path):
        return None, "meta_missing"

    try:
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None, "meta_unreadable"

    matches = re.findall(r'"laterality"\s*:\s*"([^"]+)"', text)
    if not matches:
        return None, "laterality_not_found"

    last = matches[-1].strip().upper()

    if last in {"L", "LEFT", "OS"}:
        return "L", "ok"
    if last in {"R", "RIGHT", "OD"}:
        return "R", "ok"

    return None, f"laterality_unrecognised:{last}"


def resolve_laterality(volume_folder: str, volume_label: str, overrides: dict):
    """
    Resolves eye laterality using metadata first and optional private
    overrides second.
    """
    patient_id = extract_patient_id(volume_label)
    patient_id = coerce_duplicate_numeric_id(canonical_patient_id(patient_id))
    volume_index = extract_volume_index(volume_label)

    eye, meta_status = read_laterality_from_meta_regex(volume_folder)

    if eye is None and volume_index is not None:
        key = (patient_id, volume_index)
        if key in overrides:
            return overrides[key], f"private_override_from_{meta_status}"

    return eye, meta_status


def make_scan_id(volume_folder: str) -> str:
    """
    Creates a non-identifying scan ID from the folder path.
    This avoids writing patient identifiers into the output table.
    """
    digest = hashlib.sha256(volume_folder.encode("utf-8")).hexdigest()[:12]
    return f"scan_{digest}"


# ============================================================
# Image processing helpers
# ============================================================

def inpaint_invalid_values(arr: np.ndarray, max_iter: int = 200, tol: float = 1e-3) -> np.ndarray:
    """
    Fills NaN or infinite values by iterative neighbour averaging.
    """
    x = arr.astype(float).copy()
    invalid_mask = ~np.isfinite(x)

    if not invalid_mask.any():
        return x

    if np.isfinite(x).any():
        x[invalid_mask] = np.nanmean(x[np.isfinite(x)])
    else:
        x[invalid_mask] = 0.0

    for _ in range(max_iter):
        prev = x.copy()

        up = np.roll(x, -1, axis=0)
        up[-1, :] = x[-1, :]

        down = np.roll(x, 1, axis=0)
        down[0, :] = x[0, :]

        left = np.roll(x, 1, axis=1)
        left[:, 0] = x[:, 0]

        right = np.roll(x, -1, axis=1)
        right[:, -1] = x[:, -1]

        new_values = (up + down + left + right) / 4.0
        x[invalid_mask] = new_values[invalid_mask]

        diff = np.nanmax(np.abs(x - prev))
        if diff < tol:
            break

    return x


def resize_to_square(arr: np.ndarray, target: int = 224) -> np.ndarray:
    """
    Resizes a 2D array to a square shape using 1D interpolation
    along each axis.
    """
    height, width = arr.shape
    arr = arr.astype(float)

    y_old = np.linspace(0, height - 1, height)
    y_new = np.linspace(0, height - 1, target)
    temp = np.empty((target, width), dtype=float)

    for j in range(width):
        temp[:, j] = np.interp(y_new, y_old, arr[:, j])

    x_old = np.linspace(0, width - 1, width)
    x_new = np.linspace(0, width - 1, target)
    out = np.empty((target, target), dtype=float)

    for i in range(target):
        out[i, :] = np.interp(x_new, x_old, temp[i, :])

    return out


def find_foveal_centre(total_map: np.ndarray, centre_frac: float = 0.35):
    """
    Estimates the foveal centre as the minimum value within a central
    search window of the total thickness map.
    """
    height, width = total_map.shape
    cy0, cx0 = height // 2, width // 2

    box_h = max(1, int(height * centre_frac))
    box_w = max(1, int(width * centre_frac))

    y1 = max(0, cy0 - box_h // 2)
    y2 = min(height, cy0 + box_h // 2)
    x1 = max(0, cx0 - box_w // 2)
    x2 = min(width, cx0 + box_w // 2)

    sub = total_map[y1:y2, x1:x2]

    if sub.size == 0 or not np.isfinite(sub).any():
        if not np.isfinite(total_map).any():
            return cy0, cx0
        idx = np.nanargmin(total_map)
        return np.unravel_index(idx, total_map.shape)

    idx = np.nanargmin(sub)
    dy, dx = np.unravel_index(idx, sub.shape)
    return y1 + dy, x1 + dx


def etdrs_masks_mm_calibrated(shape, centre):
    """
    Builds ETDRS-style masks using a 6 mm field of view and the
    estimated foveal centre.
    """
    height, width = shape
    cy, cx = centre

    yy, xx = np.indices((height, width))
    dy = yy - cy
    dx = xx - cx
    r = np.sqrt(dx**2 + dy**2)

    r_c = R_C_MM * PX_PER_MM
    r_in = R_IN_MM * PX_PER_MM
    r_out = R_OUT_MM * PX_PER_MM

    central = r <= r_c
    inner = (r > r_c) & (r <= r_in)
    outer = (r > r_in) & (r <= r_out)

    up = dy < 0
    down = ~up
    left = dx < 0
    right = ~left

    return {
        "C": central,
        "I_UR": inner & up & right,
        "I_UL": inner & up & left,
        "I_LR": inner & down & right,
        "I_LL": inner & down & left,
        "O_UR": outer & up & right,
        "O_UL": outer & up & left,
        "O_LR": outer & down & right,
        "O_LL": outer & down & left,
    }


def sector_means(thickness_map: np.ndarray, masks: dict):
    """
    Computes the mean thickness within each ETDRS sector.
    """
    output = {}
    for sector_name, mask in masks.items():
        values = thickness_map[mask]
        values = values[np.isfinite(values)]
        output[sector_name] = float(values.mean()) if values.size else np.nan
    return output


def normalise_laterality(arr2d: np.ndarray, eye: str):
    """
    Flips left-eye maps horizontally so that all scans share a
    common anatomical orientation.
    """
    return np.fliplr(arr2d) if eye == "L" else arr2d


def layer_prefix(layer: str) -> str:
    """
    Returns the output column prefix for a retinal layer.
    """
    return "Thickness_total" if layer == "Thickness_total" else f"Layer_{layer}"


# ============================================================
# Main pipeline
# ============================================================

def main():
    overrides = load_laterality_overrides(LATERALITY_OVERRIDE_CSV)

    total_files = glob.glob(
        os.path.join(OUT_DIR, "**", "thickness_maps", "Thickness_total.npy"),
        recursive=True,
    )

    print("Found Thickness_total maps:", len(total_files))
    if not total_files:
        raise RuntimeError("No Thickness_total.npy files found. Check OUT_DIR.")

    qc_rows = []
    kept_total_paths = []

    for total_path in total_files:
        volume_folder = os.path.dirname(os.path.dirname(total_path))
        volume_label = os.path.basename(volume_folder)
        scan_id = make_scan_id(volume_folder)

        arr = np.load(total_path, mmap_mode="r")
        height, width = arr.shape

        keep = height >= HEIGHT_MIN
        reason = "" if keep else f"height<{HEIGHT_MIN}"

        volume_index = extract_volume_index(volume_label)
        eye, meta_status = resolve_laterality(volume_folder, volume_label, overrides)

        qc_rows.append({
            "scan_id": scan_id,
            "volume": volume_index,
            "eye": eye,
            "laterality_source": meta_status,
            "orig_shape": f"{height}x{width}",
            "keep_volume": keep,
            "exclude_reason": reason,
        })

        if keep:
            kept_total_paths.append(total_path)

    qc_df = pd.DataFrame(qc_rows)
    print("Kept volumes:", int(qc_df["keep_volume"].sum()), "/", len(qc_df))

    rows = []

    for total_path in kept_total_paths:
        volume_folder = os.path.dirname(os.path.dirname(total_path))
        volume_label = os.path.basename(volume_folder)
        scan_id = make_scan_id(volume_folder)

        volume_index = extract_volume_index(volume_label)
        eye, meta_status = resolve_laterality(volume_folder, volume_label, overrides)

        eye_text = "Left" if eye == "L" else ("Right" if eye == "R" else "")
        flipped = "Yes" if eye == "L" else "No"

        total_raw = np.load(total_path).astype(float)
        total_raw = normalise_laterality(total_raw, eye)
        total_filled = inpaint_invalid_values(total_raw) if not np.isfinite(total_raw).all() else total_raw
        total_map = resize_to_square(total_filled, target=TARGET_SIZE)

        centre = find_foveal_centre(total_map, centre_frac=CENTRE_FRAC)
        masks = etdrs_masks_mm_calibrated(total_map.shape, centre)

        total_sector_values = sector_means(total_map, masks)
        rows.append({
            "scan_id": scan_id,
            "volume": volume_index,
            "eye": eye_text,
            "flipped": flipped,
            "layer": "Thickness_total",
            **{f"{sector}_mean_px": total_sector_values[sector] for sector in SECTORS},
            "global_mean_px": float(np.nanmean(total_map)),
        })

        layer_files = glob.glob(os.path.join(volume_folder, "thickness_maps", "Layer_*.npy"))

        for layer_file in layer_files:
            layer_raw = np.load(layer_file).astype(float)
            layer_raw = normalise_laterality(layer_raw, eye)
            layer_filled = inpaint_invalid_values(layer_raw) if not np.isfinite(layer_raw).all() else layer_raw
            layer_map = resize_to_square(layer_filled, target=TARGET_SIZE)

            layer_name = os.path.splitext(os.path.basename(layer_file))[0].replace("Layer_", "")
            layer_sector_values = sector_means(layer_map, masks)

            rows.append({
                "scan_id": scan_id,
                "volume": volume_index,
                "eye": eye_text,
                "flipped": flipped,
                "layer": layer_name,
                **{f"{sector}_mean_px": layer_sector_values[sector] for sector in SECTORS},
                "global_mean_px": float(np.nanmean(layer_map)),
            })

    df_long = pd.DataFrame(rows)

    index_cols = ["scan_id", "volume", "eye", "flipped"]
    wide = df_long.pivot_table(
        index=index_cols,
        columns="layer",
        values=SECTOR_COLUMNS + ["global_mean_px"],
        aggfunc="mean",
    )

    flattened_columns = []
    for feature_name, layer_name in wide.columns:
        if feature_name == "global_mean_px":
            flattened_columns.append(layer_prefix(layer_name))
        else:
            flattened_columns.append(f"{layer_prefix(layer_name)}_{feature_name}")

    wide.columns = flattened_columns
    wide = wide.reset_index()

    ordered_feature_columns = []
    for layer_name in LAYER_ORDER:
        prefix = layer_prefix(layer_name)
        ordered_feature_columns.append(prefix)
        for sector in SECTORS:
            ordered_feature_columns.append(f"{prefix}_{sector}_mean_px")

    ordered_feature_columns = [col for col in ordered_feature_columns if col in wide.columns]

    formatted = wide[["scan_id", "volume", "eye", "flipped"] + ordered_feature_columns].copy()

    print("Formatted table shape:", formatted.shape)
    print("Feature columns count:", len(ordered_feature_columns))

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        formatted.to_excel(writer, sheet_name="formatted_90cols", index=False)
        wide.to_excel(writer, sheet_name="wide_raw", index=False)
        df_long.to_excel(writer, sheet_name="debug_long", index=False)
        qc_df.to_excel(writer, sheet_name="qc", index=False)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
