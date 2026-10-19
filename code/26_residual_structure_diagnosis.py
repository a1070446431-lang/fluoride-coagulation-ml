# -*- coding: utf-8 -*-
"""
26_residual_structure_diagnosis.py

Purpose
-------
Diagnose residual structure of the final Selected CatBoost-HGB Weighted 2Model.

This script does NOT build a residual-correction model.
It only checks whether residuals show systematic structure with respect to:
    1. Observed F removal
    2. Predicted F removal
    3. influent pH
    4. coagulant dose
    5. initial fluoride concentration
    6. coagulant type
    7. selected numeric mechanism features

Rationale
---------
Residual learning is only justified if the residuals are not random noise.
If residuals show systematic bias in low-removal, high-dose, abnormal-pH,
or high-initial-F regions, then residual correction can be considered later.

Inputs
------
Real script:
    19_selected_feature_fusion.py

Tables:
    outputs/tables/24_exact_ablation_candidate_results.csv
    outputs/tables/24_exact_ablation_weights.csv

Outputs
-------
Tables:
    outputs/tables/26_residual_predictions_with_features.csv
    outputs/tables/26_residual_by_observed_bin.csv
    outputs/tables/26_residual_by_predicted_bin.csv
    outputs/tables/26_residual_by_ph_bin.csv
    outputs/tables/26_residual_by_dose_bin.csv
    outputs/tables/26_residual_by_initialF_bin.csv
    outputs/tables/26_residual_by_coagulant_top.csv
    outputs/tables/26_residual_numeric_correlation.csv
    outputs/tables/26_residual_learning_decision.txt

Figures:
    outputs/figures/75_residual_structure_final_2model.png/pdf
    outputs/figures/76_residual_bias_feature_bins.png/pdf
"""

from __future__ import annotations

import ast
import math
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# 0. Import real script 19
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_file_upwards(start_path: Path, filename: str) -> Path:
    start_path = start_path.resolve()
    search_roots = [start_path.parent] + list(start_path.parents)

    for root in search_roots:
        candidates = [
            root / filename,
            root / "src" / filename,
        ]
        for p in candidates:
            if p.exists():
                return p

    raise FileNotFoundError(f"Cannot find {filename}")


def load_script19():
    module_path = find_file_upwards(Path(__file__), "19_selected_feature_fusion.py")

    log("=" * 100)
    log("Loading real 19_selected_feature_fusion.py:")
    log(str(module_path))
    log("=" * 100)

    spec = importlib.util.spec_from_file_location("selected_feature_fusion_19", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


sf19 = load_script19()

PROJECT_ROOT = Path(sf19.PROJECT_ROOT)
FIG_DIR = Path(sf19.FIG_DIR)
TABLE_DIR = Path(sf19.TABLE_DIR)

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

COLORS = {
    "teal": "#63B7B2",
    "orange": "#E28E68",
    "purple": "#9B72B0",
    "green": "#63A97A",
    "gray": "#666666",
    "dark": "#333333",
    "red": "#FF0000",
    "grid": "#E6E6E6",
}


# ============================================================
# 2. Utilities
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read CSV: {path}")


def parse_dict_text(x):
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    return ast.literal_eval(s)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_true - y_pred)

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
    }


def normalize_col_name(s: str) -> str:
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("％", "%")
    s = s.lower()
    return s


def find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    norm_map = {c: normalize_col_name(c) for c in cols}

    for cand in candidates:
        cand_norm = normalize_col_name(cand)

        for c, n in norm_map.items():
            if n == cand_norm:
                return c

    for cand in candidates:
        cand_norm = normalize_col_name(cand)

        for c, n in norm_map.items():
            if cand_norm in n:
                return c

    return None


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def save_fig(fig, filename: str):
    png_path = FIG_DIR / filename
    pdf_path = FIG_DIR / filename.replace(".png", ".pdf")

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


# ============================================================
# 3. Reproduce exact final 2Model predictions with test indices
# ============================================================

def get_best_candidate_rows() -> pd.DataFrame:
    cand_path = TABLE_DIR / "24_exact_ablation_candidate_results.csv"

    if not cand_path.exists():
        raise FileNotFoundError(
            f"{cand_path} not found. Run 24_exact_ablation_selected_fusion.py first."
        )

    cand = read_csv_smart(cand_path)

    required = ["Seed", "Kind", "Selected_K", "Selected_Features", "Validation_RMSE", "Weights"]
    for c in required:
        if c not in cand.columns:
            raise KeyError(f"Column {c} not found in {cand_path.name}. Available columns: {list(cand.columns)}")

    sub = cand[cand["Kind"].astype(str).eq("Selected_CatBoost_HGB_Weighted")].copy()

    if sub.empty:
        raise ValueError("Selected_CatBoost_HGB_Weighted not found in candidate results.")

    sub["Validation_RMSE"] = pd.to_numeric(sub["Validation_RMSE"], errors="coerce")
    sub = sub.dropna(subset=["Validation_RMSE"])

    best_rows = (
        sub.sort_values(["Seed", "Validation_RMSE"], ascending=[True, True])
        .groupby("Seed", as_index=False)
        .first()
    )

    return best_rows


def reproduce_final_2model_predictions() -> pd.DataFrame:
    y, X_raw, X_eng, X_mix = sf19.prepare_feature_sets()
    best_rows = get_best_candidate_rows()

    all_rows = []

    log("=" * 100)
    log("Reproducing exact final 2Model predictions with original positional indices")
    log("=" * 100)

    for _, row in best_rows.iterrows():
        seed = int(row["Seed"])
        selected_features = str(row["Selected_Features"]).split(";")
        selected_features = [c for c in selected_features if c in X_mix.columns]
        weights = parse_dict_text(row["Weights"])

        if not selected_features:
            raise ValueError(f"No selected features found for seed {seed}")

        if "CatBoost" not in weights or "HistGradientBoosting" not in weights:
            raise ValueError(f"Invalid CatBoost-HGB weights for seed {seed}: {weights}")

        log(f"Seed {seed}: K={row['Selected_K']}, n_features={len(selected_features)}, weights={weights}")

        idx = np.arange(len(y))

        train_valid_idx, test_idx = train_test_split(
            idx,
            test_size=sf19.TEST_SIZE,
            random_state=seed,
        )

        y_train_full = y.iloc[train_valid_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        X_train_full_k = X_mix.iloc[train_valid_idx][selected_features].reset_index(drop=True)
        X_test_k = X_mix.iloc[test_idx][selected_features].reset_index(drop=True)

        test_pred_dict = {}

        for model_name in ["CatBoost", "HistGradientBoosting"]:
            _, pred_test_m = sf19.fit_predict_model_trainfull(
                model_name,
                X_train_full_k,
                y_train_full,
                X_test_k,
                seed,
            )
            test_pred_dict[model_name] = pred_test_m

        pred_test = sf19.apply_weights(test_pred_dict, weights)

        feature_block = X_mix.iloc[test_idx].reset_index(drop=True).copy()
        feature_block_raw = X_raw.iloc[test_idx].reset_index(drop=True).copy()

        out = pd.DataFrame({
            "Seed": seed,
            "Original_Position_Index": test_idx,
            "Observed": np.asarray(y_test, dtype=float),
            "Predicted": np.asarray(pred_test, dtype=float),
        })

        out["Residual"] = out["Observed"] - out["Predicted"]
        out["AbsResidual"] = out["Residual"].abs()
        out["SquaredResidual"] = out["Residual"] ** 2

        # Attach all mixed features with prefix.
        for c in feature_block.columns:
            out[f"Xmix__{c}"] = feature_block[c].values

        # Attach raw compact features with prefix. These are easier to interpret.
        for c in feature_block_raw.columns:
            out[f"Xraw__{c}"] = feature_block_raw[c].values

        all_rows.append(out)

    pred_df = pd.concat(all_rows, ignore_index=True)

    metrics = regression_metrics(pred_df["Observed"], pred_df["Predicted"])
    log("[OK] Reproduced final 2Model pooled metrics:")
    log(f"    R2   = {metrics['R2']:.4f}")
    log(f"    RMSE = {metrics['RMSE']:.4f}")
    log(f"    MAE  = {metrics['MAE']:.4f}")
    log(f"    Bias = {metrics['Bias']:+.4f}")

    return pred_df


# ============================================================
# 4. Bin summaries and correlations
# ============================================================

def summarize_group(df: pd.DataFrame, group_col: str, min_n: int = 10) -> pd.DataFrame:
    rows = []

    for g, sub in df.groupby(group_col, dropna=False, observed=False):
        n = len(sub)

        if n < min_n:
            continue

        metrics = regression_metrics(sub["Observed"], sub["Predicted"])

        rows.append({
            "Group": str(g),
            "n": n,
            "Observed_mean": sub["Observed"].mean(),
            "Predicted_mean": sub["Predicted"].mean(),
            "Residual_mean": sub["Residual"].mean(),
            "Residual_median": sub["Residual"].median(),
            "AbsResidual_mean": sub["AbsResidual"].mean(),
            "AbsResidual_median": sub["AbsResidual"].median(),
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "Bias": metrics["Bias"],
        })

    return pd.DataFrame(rows)


def add_interpretable_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()

    # Find raw/mixed columns.
    ph_col = find_col(df, [
        "Xraw__进水pH",
        "Xmix__进水pH",
        "Xraw__influent pH",
        "Xmix__influent pH",
    ])

    dose_col = find_col(df, [
        "Xraw__剂量_mM",
        "Xmix__剂量_mM",
        "Xraw__coagulant dose",
        "Xmix__coagulant dose",
        "Xmix__FE_剂量_mM",
    ])

    init_f_col = find_col(df, [
        "Xraw__初始氟_mg_L",
        "Xmix__初始氟_mg_L",
        "Xraw__initial F",
        "Xmix__initial F",
        "Xmix__FE_初始氟_mg_L",
    ])

    coagulant_col = find_col(df, [
        "Xraw__混凝剂",
        "Xmix__混凝剂",
        "Xraw__coagulant",
        "Xmix__coagulant",
    ])

    if ph_col is not None:
        df["Influent_pH_for_diagnosis"] = safe_num(df[ph_col])
    else:
        df["Influent_pH_for_diagnosis"] = np.nan

    if dose_col is not None:
        df["Dose_mM_for_diagnosis"] = safe_num(df[dose_col])
        df["Log1p_Dose_for_diagnosis"] = np.log1p(np.clip(df["Dose_mM_for_diagnosis"], 0, None))
    else:
        df["Dose_mM_for_diagnosis"] = np.nan
        df["Log1p_Dose_for_diagnosis"] = np.nan

    if init_f_col is not None:
        df["Initial_F_for_diagnosis"] = safe_num(df[init_f_col])
        df["Log1p_Initial_F_for_diagnosis"] = np.log1p(np.clip(df["Initial_F_for_diagnosis"], 0, None))
    else:
        df["Initial_F_for_diagnosis"] = np.nan
        df["Log1p_Initial_F_for_diagnosis"] = np.nan

    if coagulant_col is not None:
        df["Coagulant_for_diagnosis"] = df[coagulant_col].astype(str)
    else:
        df["Coagulant_for_diagnosis"] = "Unknown"

    # Removal bins.
    df["Observed_bin"] = pd.cut(
        df["Observed"],
        bins=[-0.01, 20, 40, 60, 80, 100.01],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )

    df["Predicted_bin"] = pd.cut(
        df["Predicted"],
        bins=[-0.01, 20, 40, 60, 80, 100.01],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )

    # pH bins.
    df["pH_bin"] = pd.cut(
        df["Influent_pH_for_diagnosis"],
        bins=[-np.inf, 5.5, 6.5, 7.5, 8.5, np.inf],
        labels=["<5.5", "5.5-6.5", "6.5-7.5", "7.5-8.5", ">8.5"],
    )

    # Quantile bins. Duplicates can happen, so catch errors.
    for col, new_col in [
        ("Log1p_Dose_for_diagnosis", "Dose_log_quantile"),
        ("Log1p_Initial_F_for_diagnosis", "InitialF_log_quantile"),
    ]:
        valid = df[col].dropna()

        if valid.nunique() >= 4:
            try:
                df[new_col] = pd.qcut(
                    df[col],
                    q=4,
                    labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                    duplicates="drop",
                )
            except Exception:
                df[new_col] = "Not enough variation"
        else:
            df[new_col] = "Not enough variation"

    return df


def numeric_residual_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    candidate_cols = []

    # Focus on raw and engineered numeric columns.
    for c in df.columns:
        if c.startswith("Xraw__") or c.startswith("Xmix__"):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 50 and s.nunique(dropna=True) >= 5:
                candidate_cols.append(c)

    for c in candidate_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        sub = pd.DataFrame({
            "x": x,
            "residual": df["Residual"],
            "abs_residual": df["AbsResidual"],
        }).dropna()

        if len(sub) < 50:
            continue

        pearson_r = sub["x"].corr(sub["residual"], method="pearson")
        spearman_r = sub["x"].corr(sub["residual"], method="spearman")
        pearson_abs = sub["x"].corr(sub["abs_residual"], method="pearson")
        spearman_abs = sub["x"].corr(sub["abs_residual"], method="spearman")

        rows.append({
            "Feature": c,
            "n": len(sub),
            "Pearson_with_residual": pearson_r,
            "Spearman_with_residual": spearman_r,
            "Pearson_with_abs_residual": pearson_abs,
            "Spearman_with_abs_residual": spearman_abs,
            "max_abs_corr": np.nanmax([
                abs(pearson_r),
                abs(spearman_r),
                abs(pearson_abs),
                abs(spearman_abs),
            ]),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values("max_abs_corr", ascending=False).reset_index(drop=True)

    return out


def coagulant_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    counts = df["Coagulant_for_diagnosis"].value_counts(dropna=False)

    top_values = list(counts.head(top_n).index)

    df2 = df.copy()
    df2["Coagulant_top_group"] = np.where(
        df2["Coagulant_for_diagnosis"].isin(top_values),
        df2["Coagulant_for_diagnosis"],
        "Other",
    )

    out = summarize_group(df2, "Coagulant_top_group", min_n=10)

    if not out.empty:
        out = out.sort_values("n", ascending=False).reset_index(drop=True)

    return out


# ============================================================
# 5. Decision logic
# ============================================================

def build_decision_report(
    pred_df: pd.DataFrame,
    observed_summary: pd.DataFrame,
    ph_summary: pd.DataFrame,
    dose_summary: pd.DataFrame,
    initf_summary: pd.DataFrame,
    corr_df: pd.DataFrame,
):
    path = TABLE_DIR / "26_residual_learning_decision.txt"

    metrics = regression_metrics(pred_df["Observed"], pred_df["Predicted"])

    lines = []
    lines.append("=" * 100)
    lines.append("Residual-structure diagnosis for final CatBoost-HGB Weighted 2Model")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Pooled performance:")
    lines.append(f"R2   = {metrics['R2']:.4f}")
    lines.append(f"RMSE = {metrics['RMSE']:.4f}")
    lines.append(f"MAE  = {metrics['MAE']:.4f}")
    lines.append(f"Bias = {metrics['Bias']:+.4f}")
    lines.append("")

    # Residual bias range by observed-removal bins.
    def bias_range(summary):
        if summary is None or summary.empty or "Residual_mean" not in summary.columns:
            return np.nan
        return summary["Residual_mean"].max() - summary["Residual_mean"].min()

    observed_bias_range = bias_range(observed_summary)
    ph_bias_range = bias_range(ph_summary)
    dose_bias_range = bias_range(dose_summary)
    initf_bias_range = bias_range(initf_summary)

    max_corr = np.nan

    if corr_df is not None and not corr_df.empty:
        max_corr = corr_df["max_abs_corr"].max()

    lines.append("Structure strength indicators:")
    lines.append(f"Observed-removal bin mean residual range = {observed_bias_range:.3f}")
    lines.append(f"pH-bin mean residual range               = {ph_bias_range:.3f}")
    lines.append(f"Dose-quantile mean residual range        = {dose_bias_range:.3f}")
    lines.append(f"Initial-F-quantile mean residual range   = {initf_bias_range:.3f}")
    lines.append(f"Max absolute residual-feature correlation = {max_corr:.3f}")
    lines.append("")

    lines.append("Interpretation thresholds used here:")
    lines.append("Weak structure:      max bin-bias range < 5 percentage points and max |corr| < 0.15")
    lines.append("Moderate structure:  bin-bias range about 5-10 points or max |corr| about 0.15-0.30")
    lines.append("Strong structure:    bin-bias range > 10 points or max |corr| > 0.30")
    lines.append("")

    max_bin_range = np.nanmax([
        observed_bias_range,
        ph_bias_range,
        dose_bias_range,
        initf_bias_range,
    ])

    if max_bin_range > 10 or (not np.isnan(max_corr) and max_corr > 0.30):
        decision = (
            "Residuals show strong systematic structure. "
            "Residual correction is worth testing, but only with strict validation."
        )
    elif max_bin_range > 5 or (not np.isnan(max_corr) and max_corr > 0.15):
        decision = (
            "Residuals show moderate structure. "
            "Residual correction can be tested as an exploratory/supplementary model, "
            "but it should not replace the current final model unless it improves test performance across seeds."
        )
    else:
        decision = (
            "Residuals show weak structure. "
            "Residual correction is unlikely to be robust and may overfit."
        )

    lines.append("Decision:")
    lines.append(decision)
    lines.append("")

    lines.append("Top residual-feature correlations:")
    if corr_df is not None and not corr_df.empty:
        lines.append(
            corr_df.head(15)[[
                "Feature",
                "n",
                "Pearson_with_residual",
                "Spearman_with_residual",
                "Pearson_with_abs_residual",
                "Spearman_with_abs_residual",
                "max_abs_corr",
            ]].to_string(index=False)
        )
    else:
        lines.append("No valid numeric residual-feature correlation table was generated.")

    lines.append("")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 6. Plotting
# ============================================================

def annotate_bar_values(ax, bars, fmt="{:.2f}", fontsize=7):
    y_top = ax.get_ylim()[1]

    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + y_top * 0.015,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_residual_structure(df: pd.DataFrame, observed_summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8))

    axes = axes.flatten()

    # A: Predicted vs residual
    ax = axes[0]
    sample = df.sample(min(len(df), 900), random_state=42)

    ax.scatter(
        sample["Predicted"],
        sample["Residual"],
        s=12,
        color=COLORS["teal"],
        alpha=0.62,
        edgecolor="none",
    )
    ax.axhline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)

    rmse = math.sqrt(np.mean(df["Residual"] ** 2))
    ax.axhspan(-rmse, rmse, color=COLORS["teal"], alpha=0.10, linewidth=0)

    ax.set_xlabel("Predicted F removal (%)")
    ax.set_ylabel("Residual = observed - predicted (%)")
    ax.set_title("(A) Residual vs predicted value")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    ax.text(
        0.04, 0.08,
        f"RMSE band: +/-{rmse:.2f}%",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75)
    )

    # B: Observed vs residual
    ax = axes[1]
    ax.scatter(
        sample["Observed"],
        sample["Residual"],
        s=12,
        color=COLORS["purple"],
        alpha=0.62,
        edgecolor="none",
    )
    ax.axhline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("(B) Residual vs observed value")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

    # C: Residual by observed bin
    ax = axes[2]
    if not observed_summary.empty:
        x = np.arange(len(observed_summary))
        bars = ax.bar(
            x,
            observed_summary["Residual_mean"],
            color=COLORS["orange"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axhline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(observed_summary["Group"], rotation=0)
        ax.set_ylabel("Mean residual (%)")
        ax.set_xlabel("Observed F removal bin (%)")
        ax.set_title("(C) Mean residual by observed-removal bin")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

        for xi, row in observed_summary.iterrows():
            ax.text(
                xi,
                row["Residual_mean"],
                f"{row['Residual_mean']:+.2f}\nn={int(row['n'])}",
                ha="center",
                va="bottom" if row["Residual_mean"] >= 0 else "top",
                fontsize=7,
            )

    # D: Absolute residual by observed bin
    ax = axes[3]
    if not observed_summary.empty:
        x = np.arange(len(observed_summary))
        bars = ax.bar(
            x,
            observed_summary["AbsResidual_mean"],
            color=COLORS["teal"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(observed_summary["Group"], rotation=0)
        ax.set_ylabel("Mean absolute residual (%)")
        ax.set_xlabel("Observed F removal bin (%)")
        ax.set_title("(D) Error magnitude by observed-removal bin")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

        ax.set_ylim(0, max(1, observed_summary["AbsResidual_mean"].max() * 1.25))
        annotate_bar_values(ax, bars, fmt="{:.2f}", fontsize=7)

    # E: residual histogram
    ax = axes[4]
    ax.hist(
        df["Residual"],
        bins=36,
        color=COLORS["green"],
        alpha=0.80,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax.axvline(df["Residual"].mean(), color=COLORS["dark"], linestyle="-", linewidth=1.0)
    ax.set_xlabel("Residual (%)")
    ax.set_ylabel("Count")
    ax.set_title("(E) Residual distribution")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.text(
        0.04, 0.92,
        f"Mean = {df['Residual'].mean():+.2f}%\nMedian = {df['Residual'].median():+.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75)
    )

    # F: observed-predicted scatter
    ax = axes[5]
    ax.scatter(
        sample["Observed"],
        sample["Predicted"],
        s=12,
        color=COLORS["teal"],
        alpha=0.62,
        edgecolor="none",
    )
    ax.plot([0, 100], [0, 100], color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title("(F) Final 2Model predicted vs observed")

    metrics = regression_metrics(df["Observed"], df["Predicted"])
    ax.text(
        0.06, 0.90,
        f"$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80)
    )

    fig.suptitle(
        "Residual-structure diagnosis of the final CatBoost-HGB weighted 2Model",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    save_fig(fig, "75_residual_structure_final_2model.png")


def plot_feature_bin_bias(
    ph_summary: pd.DataFrame,
    dose_summary: pd.DataFrame,
    initf_summary: pd.DataFrame,
    coagulant_summary_df: pd.DataFrame,
):
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.8))
    axes = axes.flatten()

    plot_specs = [
        (ph_summary, "Influent pH bin", "(A) Mean residual by pH bin"),
        (dose_summary, "Dose quantile", "(B) Mean residual by dose quantile"),
        (initf_summary, "Initial F quantile", "(C) Mean residual by initial-F quantile"),
        (coagulant_summary_df.head(8), "Coagulant group", "(D) Mean residual by top coagulant groups"),
    ]

    for ax, (summary, xlabel, title) in zip(axes, plot_specs):
        if summary is None or summary.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No valid grouped residual data", ha="center", va="center")
            continue

        data = summary.copy()

        x = np.arange(len(data))
        bars = ax.bar(
            x,
            data["Residual_mean"],
            color=COLORS["teal"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.axhline(0, color=COLORS["red"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(data["Group"], rotation=35, ha="right")
        ax.set_ylabel("Mean residual (%)")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)

        for xi, row in data.iterrows():
            val = row["Residual_mean"]

            ax.text(
                xi,
                val,
                f"{val:+.2f}\nn={int(row['n'])}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )

    fig.suptitle(
        "Residual bias across key process and chemical feature groups",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    save_fig(fig, "76_residual_bias_feature_bins.png")


# ============================================================
# 7. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("26 Residual structure diagnosis started")
    log("=" * 100)
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"TABLE_DIR    = {TABLE_DIR}")
    log(f"FIG_DIR      = {FIG_DIR}")
    log("=" * 100)

    pred_df = reproduce_final_2model_predictions()
    pred_df = add_interpretable_columns(pred_df)

    # Group summaries.
    observed_summary = summarize_group(pred_df, "Observed_bin", min_n=10)
    predicted_summary = summarize_group(pred_df, "Predicted_bin", min_n=10)
    ph_summary = summarize_group(pred_df, "pH_bin", min_n=10)
    dose_summary = summarize_group(pred_df, "Dose_log_quantile", min_n=10)
    initf_summary = summarize_group(pred_df, "InitialF_log_quantile", min_n=10)
    coagulant_summary_df = coagulant_summary(pred_df, top_n=10)
    corr_df = numeric_residual_correlation(pred_df)

    # Save tables.
    pred_path = TABLE_DIR / "26_residual_predictions_with_features.csv"
    observed_path = TABLE_DIR / "26_residual_by_observed_bin.csv"
    predicted_path = TABLE_DIR / "26_residual_by_predicted_bin.csv"
    ph_path = TABLE_DIR / "26_residual_by_ph_bin.csv"
    dose_path = TABLE_DIR / "26_residual_by_dose_bin.csv"
    initf_path = TABLE_DIR / "26_residual_by_initialF_bin.csv"
    coagulant_path = TABLE_DIR / "26_residual_by_coagulant_top.csv"
    corr_path = TABLE_DIR / "26_residual_numeric_correlation.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    observed_summary.to_csv(observed_path, index=False, encoding="utf-8-sig")
    predicted_summary.to_csv(predicted_path, index=False, encoding="utf-8-sig")
    ph_summary.to_csv(ph_path, index=False, encoding="utf-8-sig")
    dose_summary.to_csv(dose_path, index=False, encoding="utf-8-sig")
    initf_summary.to_csv(initf_path, index=False, encoding="utf-8-sig")
    coagulant_summary_df.to_csv(coagulant_path, index=False, encoding="utf-8-sig")
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")

    log("\n[OK] Residual tables saved:")
    for p in [
        pred_path,
        observed_path,
        predicted_path,
        ph_path,
        dose_path,
        initf_path,
        coagulant_path,
        corr_path,
    ]:
        log(f"  - {p}")

    # Print key summaries.
    log("\n=== Residual by observed-removal bin ===")
    print(observed_summary.to_string(index=False))

    log("\n=== Residual by pH bin ===")
    print(ph_summary.to_string(index=False))

    log("\n=== Residual by dose quantile ===")
    print(dose_summary.to_string(index=False))

    log("\n=== Top residual-feature correlations ===")
    if corr_df.empty:
        log("No numeric residual-feature correlation generated.")
    else:
        print(corr_df.head(15).to_string(index=False))

    # Decision report.
    build_decision_report(
        pred_df=pred_df,
        observed_summary=observed_summary,
        ph_summary=ph_summary,
        dose_summary=dose_summary,
        initf_summary=initf_summary,
        corr_df=corr_df,
    )

    # Figures.
    plot_residual_structure(pred_df, observed_summary)
    plot_feature_bin_bias(ph_summary, dose_summary, initf_summary, coagulant_summary_df)

    log("\n" + "=" * 100)
    log("[DONE] Residual structure diagnosis completed.")
    log("=" * 100)


if __name__ == "__main__":
    main()