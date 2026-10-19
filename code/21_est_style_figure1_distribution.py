# -*- coding: utf-8 -*-
"""
21_est_style_figure1_distribution.py

Final publication-quality EST-like Figure 1 for the fluoride coagulation ML project.

Final design:
    (A) 3 x 3 histograms:
        8 mainly raw/process-relevant input variables + target F removal (%)

    (B) 3 x 3 grouped boxplots:
        The same 8 variables + target F removal (%)

Important final adjustment:
    Fixed 90% grouping is replaced by quantile-based low/high groups:
        Low group  = lower 40% of F removal
        High group = upper 40% of F removal
        Middle 20% is omitted from boxplot comparison

Reason:
    Fixed R >= 90% grouping can produce empty or unstable boxplots in imbalanced data.
    Quantile grouping is more suitable for showing distributional differences between
    low- and high-performance samples.

Outputs:
    outputs/figures/Figure1_EST_like_feature_distribution.png
    outputs/figures/Figure1_EST_like_feature_distribution.pdf
    outputs/tables/21_est_feature_distribution_summary.csv
"""

from pathlib import Path
import re
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import shapiro, mannwhitneyu

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [current, current.parent, current.parent.parent]

    for p in candidates:
        if (p / "outputs").exists():
            return p

    if current.name.lower() == "src":
        return current.parent

    raise FileNotFoundError("Cannot locate project root.")


PROJECT_ROOT = find_project_root()

DATA_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "model_ready_v1.csv",
    PROJECT_ROOT / "data" / "processed" / "model_ready-v1.csv",
    PROJECT_ROOT / "data" / "processed" / "model_ready.csv",
]

DATA_PATH = None
for p in DATA_PATH_CANDIDATES:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError("No model-ready dataset found in data/processed/")

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Global settings
# ============================================================

N_INPUT_FEATURES = 8
LOW_Q = 0.40
HIGH_Q = 0.60
RANDOM_STATE = 42

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8.8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8.0,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "#F2F2F7",
    "figure.facecolor": "white",
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

COLOR_HIST = "#8CC9E3"
COLOR_HIST_EDGE = "#73B8D6"
COLOR_LOW = "#8FC3D9"
COLOR_HIGH = "#E69482"
COLOR_LINE = "#FF0000"
COLOR_TEXT = "#222222"


# ============================================================
# 3. Utility functions
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read file: {path}")


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("²", "2")
    s = s.replace("%", "pct")
    s = s.replace("−", "-")
    s = s.replace("×", "x")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_numeric_series(series: pd.Series, min_valid: int = 20) -> bool:
    vals = safe_numeric(series)
    return vals.notna().sum() >= min_valid


def safe_ascii_label(label: str) -> str:
    label = str(label)
    replacements = {
        "×": "x",
        "−": "-",
        "–": "-",
        "—": "-",
        "≥": ">=",
        "≤": "<=",
        "℃": "C",
        "°C": "C",
        "μ": "u",
        "F⁻": "F",
        "F−": "F",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)

    label = re.sub(r"[^\x00-\x7F]+", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def format_p_value(p):
    if pd.isna(p):
        return "P=NA"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def safe_shapiro(series: pd.Series):
    x = safe_numeric(series).dropna()

    if len(x) < 3:
        return np.nan, np.nan

    if len(x) > 5000:
        x = x.sample(5000, random_state=RANDOM_STATE)

    try:
        w, p = shapiro(x)
        return w, p
    except Exception:
        return np.nan, np.nan


def safe_mannwhitney(x1, x2):
    x1 = safe_numeric(pd.Series(x1)).dropna()
    x2 = safe_numeric(pd.Series(x2)).dropna()

    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    try:
        _, p = mannwhitneyu(x1, x2, alternative="two-sided")
        return p
    except Exception:
        return np.nan


# ============================================================
# 4. Target handling
# ============================================================

def is_likely_derived_removal_column(col: str) -> bool:
    n = normalize_name(col)
    derived_keywords = [
        "反算",
        "出液度反算",
        "去除率_p",
        "removal_pred",
        "predicted_removal",
        "calculated_removal",
        "estimated_removal",
    ]
    return any(k in n for k in derived_keywords)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Prefer the final true removal column, especially 氟去除率_pct.
    Avoid derived removal columns such as 出液度反算去除率_p.
    """
    exact_candidates = [
        "氟去除率_pct",
        "氟去除率",
        "氟去除率 (%)",
        "氟去除率(%)",
        "除氟率",
        "除氟率 (%)",
        "除氟率(%)",
        "去除率",
        "去除率 (%)",
        "去除率(%)",
        "F removal",
        "F removal (%)",
        "Fluoride removal",
        "Fluoride removal (%)",
        "Removal",
        "Removal (%)",
        "Removal rate",
        "Removal rate (%)",
        "target",
        "y",
    ]

    for cand in exact_candidates:
        cand_norm = normalize_name(cand)
        for c in df.columns:
            if is_likely_derived_removal_column(c):
                continue

            if normalize_name(c) == cand_norm:
                vals = safe_numeric(df[c])
                if vals.notna().sum() >= 10:
                    print(f"[OK] Target column detected by exact match: {c}")
                    return c

    for c in df.columns:
        if is_likely_derived_removal_column(c):
            continue

        n = normalize_name(c)
        raw = str(c).strip().lower()

        is_target_like = (
            "氟去除率" in n or
            "除氟率" in n or
            ("removal" in raw and "pred" not in raw and "error" not in raw)
        )

        if is_target_like:
            vals = safe_numeric(df[c])
            if vals.notna().sum() >= 10:
                print(f"[OK] Target column detected by fuzzy match: {c}")
                return c

    possible = []
    for c in df.columns:
        if is_likely_derived_removal_column(c):
            continue

        vals = safe_numeric(df[c]).dropna()
        if len(vals) < 10:
            continue

        in_range_ratio = ((vals >= 0) & (vals <= 100)).mean()
        if in_range_ratio > 0.85:
            possible.append((c, len(vals), vals.mean(), vals.min(), vals.max()))

    if possible:
        selected = possible[-1][0]
        print(f"[WARN] Target column auto-selected as fallback: {selected}")
        return selected

    raise ValueError("Target column not found.")


def convert_target_to_percent(y: pd.Series) -> pd.Series:
    """
    Ensure target is expressed as 0-100 percentage.
    If target appears to be 0-1 fraction, multiply by 100.
    """
    y = safe_numeric(y)
    valid = y.dropna()

    if len(valid) == 0:
        return y

    ymax = valid.quantile(0.99)

    if ymax <= 1.5:
        print("[INFO] Target appears to be fraction scale. Converted to percentage by multiplying 100.")
        return y * 100.0

    return y


def build_group_masks(y_percent: pd.Series):
    """
    Quantile-based grouping:
        low group  = lower 40%
        high group = upper 40%
        middle 20% omitted
    """
    y = safe_numeric(y_percent)

    q_low = y.quantile(LOW_Q)
    q_high = y.quantile(HIGH_Q)

    if not np.isfinite(q_low) or not np.isfinite(q_high):
        raise ValueError("Cannot calculate quantile thresholds for F removal grouping.")

    # In case of many duplicated values, fallback to 33/67 quantiles
    if q_low >= q_high:
        q_low = y.quantile(0.33)
        q_high = y.quantile(0.67)

    # Final fallback: median split
    if q_low >= q_high:
        median_val = y.median()
        low_mask = y < median_val
        high_mask = y >= median_val
        q_low = median_val
        q_high = median_val
        group_desc = "median split"
    else:
        low_mask = y <= q_low
        high_mask = y >= q_high
        group_desc = f"lower {int(LOW_Q * 100)}% vs upper {int((1 - HIGH_Q) * 100)}%"

    print(f"[OK] Grouping method: {group_desc}")
    print(f"[OK] Low group threshold : F removal <= {q_low:.2f}%")
    print(f"[OK] High group threshold: F removal >= {q_high:.2f}%")
    print(f"[OK] Group counts: Low = {int(low_mask.sum())}, High = {int(high_mask.sum())}")

    return low_mask, high_mask, q_low, q_high, group_desc


# ============================================================
# 5. Feature labeling and filtering
# ============================================================

def semantic_label(col: str):
    n = normalize_name(col)

    if "游离氟占比" in n or "free_fluoride_fraction" in n:
        return "Free fluoride fraction"

    if ("influent" in n and "ph" in n) or ("initial" in n and "ph" in n) or "进水ph" in n or "原水ph" in n:
        return "Influent pH"

    if ("effluent" in n and "ph" in n) or ("out" in n and "ph" in n) or "出水ph" in n or "final_ph" in n:
        return "Effluent pH"

    if ("initial" in n and ("f" in n or "fluoride" in n)) or "初始氟" in n or "进水氟" in n:
        return "Initial F (mg/L)"

    if ("dose" in n and "mm" in n) or "剂量" in n or "投加量" in n:
        return "Dose (mM)"

    if "retention" in n or "residence" in n or "停留时间" in n:
        return "Retention time (min)"

    if "temperature" in n or "temp" in n or "水温" in n:
        return "Temperature (C)"

    if "interfer" in n or "干扰离子" in n:
        if "浓度" in n or "mmol" in n or "mg_l" in n:
            return "Interfering ion concentration"
        return "Interfering ion code"

    if "coagulant" in n or "混凝剂" in n:
        return "Coagulant code"

    if "stir" in n or "搅拌时间" in n:
        return "Stirring time (min)"

    if "settling" in n or "sedimentation" in n or "沉淀时间" in n:
        return "Settling time (min)"

    if "flocculant" in n or "絮凝剂" in n:
        return "Flocculant code"

    if "氟去除率" in n or "除氟率" in n or "removal" in n or "去除率" in n:
        return "F removal (%)"

    return None


def is_bad_feature_column(col: str) -> bool:
    n = normalize_name(col)

    bad_keywords = [
        "pred", "prediction", "residual", "error",
        "rank", "score", "r2", "rmse", "mae", "mse",
        "target", "y_true", "y_pred",
        "seed", "split", "fold",
        "index", "row", "id", "no", "number", "serial",

        "source", "paper", "doi", "year",
        "编号", "序号", "文献", "来源", "年份", "行号",
        "样本编号", "数据编号",

        "反算",
        "去除率_p",
        "氟去除率",
        "除氟率",
        "removal_rate",
        "removal_pct",
        "removal_percent",

        "出水氟",
        "出液氟",
        "effluent_f",
        "outlet_f",
        "final_f",
        "residual_f",
        "残余氟",

        "原始",
        "_original",
    ]

    return any(k in n for k in bad_keywords)


def score_match(col_name: str, keywords):
    n = normalize_name(col_name)
    best = 0

    for kw in keywords:
        k = normalize_name(kw)
        if n == k:
            best = max(best, 100 + len(k))
        elif k in n:
            best = max(best, 50 + len(k))

    return best


def choose_final_features(df: pd.DataFrame, target_col: str, n_features: int = 8):
    numeric_cols = [c for c in df.columns if is_numeric_series(df[c], min_valid=20)]

    candidates = []
    for c in numeric_cols:
        if c == target_col:
            continue
        if is_bad_feature_column(c):
            continue
        candidates.append(c)

    semantic_groups = [
        ("dose", ["剂量_mM", "dose_mm", "dose_m_m", "coagulant_dose", "剂量", "投加量"]),
        ("influent_pH", ["进水pH", "influent_ph", "initial_ph", "inlet_ph", "原水ph"]),
        ("initial_f", ["初始氟", "initial_f", "initial_fluoride", "influent_f", "进水氟"]),
        ("retention", ["停留时间", "retention_time", "residence_time"]),
        ("temperature", ["水温", "temperature", "temp"]),
        ("interfering", ["干扰离子", "interfering"]),
        ("effluent_pH", ["出水pH", "effluent_ph", "outlet_ph", "final_ph"]),
        ("free_fluoride_fraction", ["游离氟占比", "free_fluoride_fraction"]),

        ("coagulant", ["混凝剂", "coagulant", "coagulant_code", "coagulant_category"]),
        ("stirring", ["搅拌时间", "stirring_time", "stir_time"]),
        ("settling", ["沉淀时间", "settling_time", "sedimentation_time"]),
        ("flocculant", ["絮凝剂", "flocculant"]),
    ]

    selected = []
    used_cols = set()
    used_labels = set()

    for _, keywords in semantic_groups:
        best_col = None
        best_score = -1

        for c in candidates:
            if c in used_cols:
                continue

            s = score_match(c, keywords)

            if s > best_score:
                best_score = s
                best_col = c

        if best_col is not None and best_score > 0:
            lab = semantic_label(best_col)

            if lab is not None and lab not in used_labels:
                selected.append(best_col)
                used_cols.add(best_col)
                used_labels.add(lab)

        if len(selected) >= n_features:
            break

    if len(selected) < n_features:
        remaining = [c for c in candidates if c not in used_cols]

        if remaining:
            variances = df[remaining].apply(lambda x: safe_numeric(x).var(), axis=0)
            variances = variances.replace([np.inf, -np.inf], np.nan).dropna()
            variances = variances.sort_values(ascending=False)

            for c in variances.index:
                lab = semantic_label(c)

                if lab is None:
                    continue
                if lab in used_labels:
                    continue

                selected.append(c)
                used_cols.add(c)
                used_labels.add(lab)

                if len(selected) >= n_features:
                    break

    return selected[:n_features]


# ============================================================
# 6. Display transformation
# ============================================================

def prepare_display_series(series: pd.Series, allow_log=True, allow_clip=True):
    x = safe_numeric(series).dropna().copy()

    if len(x) == 0:
        return x, "raw"

    method = "raw"

    if allow_log and (x >= 0).all():
        pos = x[x > 0]
        if len(pos) >= 10:
            p50 = np.percentile(pos, 50)
            p99 = np.percentile(pos, 99)

            if p50 > 0 and p99 / p50 >= 30:
                x = np.log1p(x)
                method = "log1p"

    if allow_clip and len(x) >= 20:
        lo = np.percentile(x, 1)
        hi = np.percentile(x, 99)

        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            x = x.clip(lower=lo, upper=hi)

            if method == "raw":
                method = "1-99% clipped"
            elif method == "log1p":
                method = "log1p + 1-99% clipped"

    return x, method


def display_label_suffix(method: str):
    if method == "raw":
        return ""
    if "log1p" in method:
        return "[log1p]"
    if "clipped" in method:
        return "[display]"
    return ""


# ============================================================
# 7. Plot helpers
# ============================================================

def add_group_panel_label(fig, x, y, text):
    fig.text(
        x,
        y,
        text,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="center"
    )


def style_axis(ax):
    ax.tick_params(axis="both", pad=1.0, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def draw_hist(ax, series, label, target_mode=False, q_low=None, q_high=None):
    original = safe_numeric(series).dropna()

    if target_mode:
        display_x, method = prepare_display_series(series, allow_log=False, allow_clip=False)
    else:
        display_x, method = prepare_display_series(series, allow_log=True, allow_clip=True)

    if len(display_x) == 0:
        ax.axis("off")
        return np.nan, np.nan, method

    nunique = display_x.nunique()
    bins = 12 if nunique >= 12 else max(6, min(int(nunique), 12))

    ax.hist(
        display_x,
        bins=bins,
        color=COLOR_HIST,
        edgecolor=COLOR_HIST_EDGE,
        linewidth=0.6
    )

    ax.set_title(f"Histogram of {label}", pad=1.6)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True)
    style_axis(ax)

    w, p = safe_shapiro(original)
    txt = f"W={w:.4f}\n{format_p_value(p)}" if pd.notna(w) else "W=NA\nP=NA"

    ax.text(
        0.05,
        0.74,
        txt,
        transform=ax.transAxes,
        fontsize=6.3,
        ha="left",
        va="top",
        color=COLOR_TEXT
    )

    suffix = display_label_suffix(method)
    if suffix:
        ax.text(
            0.98,
            0.97,
            suffix,
            transform=ax.transAxes,
            fontsize=5.7,
            ha="right",
            va="top",
            color="#666666"
        )

    if target_mode and q_low is not None and q_high is not None:
        ymax = ax.get_ylim()[1]

        ax.axvline(q_low, color=COLOR_LINE, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.axvline(q_high, color=COLOR_LINE, linestyle=(0, (4, 3)), linewidth=1.0)

        xmin, xmax = ax.get_xlim()
        xspan = xmax - xmin

        ax.text(
            q_low - 0.08 * xspan,
            ymax * 0.84,
            "Low",
            fontsize=6.5,
            fontweight="bold",
            ha="right"
        )

        ax.text(
            q_high + 0.02 * xspan,
            ymax * 0.84,
            "High",
            fontsize=6.5,
            fontweight="bold",
            ha="left"
        )

    return w, p, method


def draw_box(ax, series, label, low_mask, high_mask, target_mode=False):
    raw_low = safe_numeric(series[low_mask]).dropna()
    raw_high = safe_numeric(series[high_mask]).dropna()

    if target_mode:
        method = "raw"
        disp_low = raw_low.copy()
        disp_high = raw_high.copy()
    else:
        _, method = prepare_display_series(series, allow_log=True, allow_clip=True)

        if "log1p" in method:
            disp_low = np.log1p(raw_low.clip(lower=0))
            disp_high = np.log1p(raw_high.clip(lower=0))
        else:
            disp_low = raw_low.copy()
            disp_high = raw_high.copy()

        if len(disp_low) >= 20:
            lo, hi = np.percentile(disp_low, 1), np.percentile(disp_low, 99)
            if hi > lo:
                disp_low = disp_low.clip(lo, hi)

        if len(disp_high) >= 20:
            lo, hi = np.percentile(disp_high, 1), np.percentile(disp_high, 99)
            if hi > lo:
                disp_high = disp_high.clip(lo, hi)

    data = [disp_low.dropna(), disp_high.dropna()]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops=dict(color="#555555", linewidth=1.0),
        boxprops=dict(linewidth=0.8, color="#888888"),
        whiskerprops=dict(linewidth=0.8, color="#888888"),
        capprops=dict(linewidth=0.8, color="#888888"),
        flierprops=dict(
            marker="o",
            markersize=2.8,
            markerfacecolor="none",
            markeredgecolor="#777777",
            alpha=0.9
        )
    )

    bp["boxes"][0].set_facecolor(COLOR_LOW)
    bp["boxes"][1].set_facecolor(COLOR_HIGH)

    ax.set_title(f"Boxplot of {label}", pad=1.6)
    ax.set_ylabel("Value")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Low", "High"])
    ax.grid(True, axis="y")
    style_axis(ax)

    p_mw = safe_mannwhitney(raw_low, raw_high)
    if pd.notna(p_mw):
        ax.text(
            0.04,
            0.95,
            format_p_value(p_mw),
            transform=ax.transAxes,
            fontsize=5.8,
            ha="left",
            va="top",
            color="#555555"
        )

    suffix = display_label_suffix(method)
    if suffix and not target_mode:
        ax.text(
            0.98,
            0.97,
            suffix,
            transform=ax.transAxes,
            fontsize=5.6,
            ha="right",
            va="top",
            color="#666666"
        )

    return p_mw, len(raw_low), len(raw_high), method


# ============================================================
# 8. Main workflow
# ============================================================

def main():
    print("=" * 95)
    print("Generating FINAL EST-like Figure 1 with balanced low/high groups ...")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data path    : {DATA_PATH}")
    print(f"Figure dir   : {FIG_DIR}")
    print(f"Table dir    : {TABLE_DIR}")
    print("=" * 95)

    df = read_csv_smart(DATA_PATH)
    print(f"[OK] Data loaded: {df.shape}")

    target_col = detect_target_column(df)
    df[target_col] = convert_target_to_percent(df[target_col])
    print(f"[OK] Target column detected: {target_col}")

    feature_cols = choose_final_features(df, target_col, n_features=N_INPUT_FEATURES)

    if len(feature_cols) < N_INPUT_FEATURES:
        print(f"[WARN] Only {len(feature_cols)} input features were selected.")

    label_map = {}
    for c in feature_cols:
        lab = semantic_label(c)
        if lab is None:
            raise ValueError(f"Selected feature has no clean label: {c}")
        label_map[c] = safe_ascii_label(lab)

    label_map[target_col] = "F removal (%)"

    print("[OK] Selected input features for Figure 1:")
    for i, c in enumerate(feature_cols, 1):
        print(f"    {i:02d}. {c}  -->  {label_map[c]}")

    y = safe_numeric(df[target_col])
    valid_mask = y.notna()

    df_plot = df.loc[valid_mask].copy()
    y = safe_numeric(df_plot[target_col])

    low_mask, high_mask, q_low, q_high, group_desc = build_group_masks(y)

    hist_cols = feature_cols.copy()
    hist_cols.append(target_col)

    boxplot_cols = feature_cols.copy()
    boxplot_cols.append(target_col)

    hist_cols = hist_cols[:9]
    boxplot_cols = boxplot_cols[:9]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    fig = plt.figure(figsize=(15.8, 8.6))

    outer = GridSpec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.12,
        left=0.055,
        right=0.985,
        top=0.835,
        bottom=0.185
    )

    left_gs = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=outer[0],
        wspace=0.45,
        hspace=0.64
    )

    right_gs = GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=outer[1],
        wspace=0.45,
        hspace=0.64
    )

    hist_summary_rows = []
    box_summary_rows = []

    # Panel A
    for i, col in enumerate(hist_cols):
        ax = fig.add_subplot(left_gs[i // 3, i % 3])

        label = label_map[col]
        target_mode = (col == target_col)

        w, p, disp_method = draw_hist(
            ax=ax,
            series=df_plot[col],
            label=label,
            target_mode=target_mode,
            q_low=q_low if target_mode else None,
            q_high=q_high if target_mode else None
        )

        hist_summary_rows.append({
            "panel": "A",
            "feature": col,
            "plot_label": label,
            "W": w,
            "P": p,
            "display_method_hist": disp_method,
            "grouping": group_desc,
            "q_low": q_low,
            "q_high": q_high
        })

    # Panel B
    for i, col in enumerate(boxplot_cols):
        ax = fig.add_subplot(right_gs[i // 3, i % 3])

        label = label_map[col]
        target_mode = (col == target_col)

        p_mw, n_low, n_high, disp_method = draw_box(
            ax=ax,
            series=df_plot[col],
            label=label,
            low_mask=low_mask,
            high_mask=high_mask,
            target_mode=target_mode
        )

        box_summary_rows.append({
            "panel": "B",
            "feature": col,
            "plot_label": label,
            "mannwhitney_p": p_mw,
            "n_low": n_low,
            "n_high": n_high,
            "display_method_box": disp_method,
            "grouping": group_desc,
            "q_low": q_low,
            "q_high": q_high
        })

    # Panel letters
    add_group_panel_label(fig, 0.038, 0.865, "(A)")
    add_group_panel_label(fig, 0.525, 0.865, "(B)")

    # Title
    title = "Feature distribution and grouped comparison for fluoride removal prediction in coagulation systems"
    fig.text(
        0.5,
        0.925,
        safe_ascii_label(title),
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold"
    )

    # Figure caption
    note = (
        f"Figure 1. (A) Histograms of selected mainly raw input variables and fluoride removal with "
        f"Shapiro-Wilk W and P values. (B) Box plots comparing low- and high-removal samples. "
        f"Low and high groups correspond to the lower {int(LOW_Q * 100)}% and upper {int((1 - HIGH_Q) * 100)}% "
        f"of F removal, respectively; the middle 20% was omitted from grouped boxplot comparison. "
        f"Highly skewed variables were transformed by log1p and/or clipped to the 1st-99th percentiles for display only, "
        f"while all statistical tests were performed using original values. "
        f"Free fluoride fraction was retained as a mechanism-related variable."
    )

    wrapped_note = textwrap.fill(safe_ascii_label(note), width=175)

    fig.text(
        0.055,
        0.075,
        wrapped_note,
        ha="left",
        va="bottom",
        fontsize=8.1
    )

    png_path = FIG_DIR / "Figure1_EST_like_feature_distribution.png"
    pdf_path = FIG_DIR / "Figure1_EST_like_feature_distribution.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Save summary table
    # --------------------------------------------------------
    hist_df = pd.DataFrame(hist_summary_rows)
    box_df = pd.DataFrame(box_summary_rows)

    summary_df = hist_df.merge(
        box_df[[
            "feature",
            "mannwhitney_p",
            "n_low",
            "n_high",
            "display_method_box",
            "grouping",
            "q_low",
            "q_high"
        ]],
        on="feature",
        how="left",
        suffixes=("_hist", "_box")
    )

    summary_path = TABLE_DIR / "21_est_feature_distribution_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 95)
    print("[DONE] FINAL balanced-group EST-like Figure 1 generated successfully.")
    print(f"PNG     : {png_path}")
    print(f"PDF     : {pdf_path}")
    print(f"SUMMARY : {summary_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()