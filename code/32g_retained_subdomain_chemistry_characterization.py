# -*- coding: utf-8 -*-
"""
32g_retained_subdomain_chemistry_characterization.py

Purpose
-------
Step 32G for the fluoride coagulation ML project.

This script characterizes the chemistry/domain meaning of the Step 32F2
reproducible selective applicability-domain (AD) retained subdomain.
It does NOT train any model and does NOT use target values to define the
retained subdomain. It only diagnoses the already selected target-free policy.

Strictness
----------
1. Uses the retained predictions produced by Step 32F2:
   outputs/tables/32f2_best_reproduced_retained_predictions.csv
2. Uses the full prediction-with-AD table for context when available:
   outputs/tables/32b_predictions_with_reliability_zone.csv
3. Merges retained Data_Index with original chemistry rows from one of:
   data/raw/数据集修改最终.xlsx
   data/processed/model_ready_v1_with_source.csv
   data/processed/cleaned_full_with_flags.csv
4. Reports prediction-instance N and unique experimental-row N separately.
5. Writes publication-ready tables and EST-like figures to outputs/.

Expected outputs
----------------
Tables:
  32g_retained_predictions_with_chemistry.csv
  32g_unique_retained_chemical_rows.csv
  32g_numeric_operating_window.csv
  32g_categorical_composition.csv
  32g_ad_component_summary.csv
  32g_representativeness_summary.csv
  32g_retained_subdomain_decision_report.txt
Figures:
  184_32g_retained_chemistry_overview.png/pdf
  185_32g_operating_window_vs_source.png/pdf
  186_32g_ad_profile_and_rule_context.png/pdf
  187_32g_literature_dominance_and_representativeness.png/pdf
"""

from __future__ import annotations

import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
SCRIPT_NAME = "32g_retained_subdomain_chemistry_characterization.py"
FIG_PREFIX = 184

BLUE = "#4E79A7"
GREEN = "#6FA77A"
ORANGE = "#DCA457"
RED = "#C44E52"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#D0D0D0"
DARK = "#222222"
MUTED_PURPLE = "#9C7AAE"

NUMERIC_CANDIDATES = {
    "initial_f": ["初始 F⁻(mg/L)", "初始 F-(mg/L)", "初始F", "Initial_F_mg_L", "initial_fluoride", "Initial fluoride"],
    "dose": ["剂量 (mM)", "剂量", "dose_mM", "Coagulant dose (mM)", "coagulant_dose_mM"],
    "pH_in": ["进水 pH", "进水pH", "influent_pH", "Influent pH", "pH"],
    "retention_time": ["停留时间（min）", "停留时间_min", "停留时间", "retention_time_min", "Retention time (min)"],
    "temperature": ["水温 (℃)", "水温", "temperature_C", "Water temperature (°C)", "Water temperature"],
    "interfering_conc": ["干扰离子浓度 (mmol/L)", "干扰离子浓度", "interfering_ion_conc", "Interfering ion concentration"],
    "effluent_pH": ["出水 pH", "出水pH", "effluent_pH", "Effluent pH"],
    "removal": ["氟去除率 (%)", "氟去除率", "F removal (%)", "Removal", "target", "y"],
    "effluent_f": ["出水 F⁻(mg/L)", "出水 F-(mg/L)", "出水F", "effluent_f_mg_L", "Effluent F"],
    "free_f_fraction": ["游离氟占比", "free_fluoride_fraction", "Free fluoride fraction"],
}

CATEGORICAL_CANDIDATES = {
    "paper": ["文献编号", "Source_Group", "Paper_ID", "paper_id"],
    "coagulant": ["混凝剂", "coagulant", "Coagulant"],
    "interfering_ion": ["干扰离子", "interfering_ions", "Interfering ions"],
    "fluoride_form": ["氟离子分布形态", "氟形态", "fluoride_form", "Fluoride form"],
}

AD_COMPONENTS = [
    "NN_Risk",
    "Mahalanobis_Risk",
    "PCA_Reconstruction_Risk",
    "Leverage_Risk",
    "Distance_AD_Risk",
    "Numeric_Boundary_Risk",
    "Categorical_Unseen_Risk",
    "Boundary_AD_Risk",
    "Corrected_AD_Risk",
    "Max_Component_Risk",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root() -> Path:
    candidates: List[Path] = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / "outputs" / "tables").exists() or (p / "data").exists() or (p / "src").exists():
            return p
    return cwd


def first_existing(paths: Sequence[Path], required: bool = True, name: str = "file") -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    if required:
        msg = "\n".join(str(p) for p in paths)
        raise FileNotFoundError(f"Cannot find required {name}. Tried:\n{msg}")
    return None


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    log(f"[TABLE] Loading {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metric_values(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": int(len(yt))}


def paper_to_std(v: object) -> str:
    if pd.isna(v):
        return "Unknown"
    s = str(v).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return f"Paper_{int(m.group(1)):03d}"
    return s


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lookup:
            return lookup[key]
    # substring fallback
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "")
        for c in df.columns:
            ck = str(c).strip().lower().replace(" ", "")
            if key and (key in ck or ck in key):
                return c
    return None


def clean_label(label: object, max_len: int = 34) -> str:
    """Make plot labels compact while preserving chemistry meaning."""
    if pd.isna(label):
        return "NA"
    s = str(label).strip()
    replacements = {
        "SO₄²⁻": "SO4(2-)",
        "CO₃²⁻": "CO3(2-)",
        "PO₄³⁻": "PO4(3-)",
        "Ca²⁺": "Ca(2+)",
        "Mg²⁺": "Mg(2+)",
        "Cl⁻": "Cl(-)",
        "HCO₃⁻": "HCO3(-)",
        "复合硫酸铝铁": "CSAF",
        "复合硫酸铝镁": "CSAM",
        "硫酸铝": "Al sulfate",
        "聚合氯化铝铁": "PAFC",
        "聚合氯化铝": "PAC",
        "单铝体系": "single-Al",
        "单Al": "single-Al",
        "混凝剂": "coagulant",
        "干扰离子": "ions",
        "（": " (",
        "）": ")",
        "，": ", ",
        "、": "/",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# -----------------------------------------------------------------------------
# Font and style
# -----------------------------------------------------------------------------
def configure_matplotlib() -> None:
    candidates = [
        "Times New Roman",
        "Arial",
        "Microsoft YaHei",
        "SimSun",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = None
    for name in candidates:
        if name in available:
            selected = name
            break
    if selected is None:
        selected = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": selected,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": DARK,
        "axes.linewidth": 1.0,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    log(f"[INFO] Matplotlib font selected: {selected}")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def polish_ax(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.6, alpha=0.8)
    ax.grid(True, axis="x", color="#F0F0F0", linewidth=0.5, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color(DARK)
        spine.set_linewidth(0.9)


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight")
        log(f"[FIG] {p}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Load source chemistry data
# -----------------------------------------------------------------------------
def load_source_chemistry(root: Path) -> Tuple[Optional[pd.DataFrame], str]:
    candidates = [
        root / "data" / "raw" / "数据集修改最终.xlsx",
        root / "数据集修改最终.xlsx",
        root / "data" / "processed" / "model_ready_v1_with_source.csv",
        root / "data" / "processed" / "cleaned_full_with_flags.csv",
        root / "model_ready_v1_with_source.csv",
        root / "cleaned_full_with_flags.csv",
    ]
    p = first_existing(candidates, required=False, name="source chemistry table")
    if p is None:
        return None, "MISSING_SOURCE_CHEMISTRY_TABLE"

    log(f"[TABLE] Loading source chemistry table: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        # The final Excel has the real column names on the second row.
        try:
            df = pd.read_excel(p, header=1)
        except Exception:
            df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, low_memory=False)

    df = df.reset_index(drop=True).copy()
    df["_Source_Row_0Based"] = np.arange(len(df), dtype=int)
    df["_Source_File"] = str(p)

    paper_col = find_col(df, CATEGORICAL_CANDIDATES["paper"])
    if paper_col is not None:
        df["_Paper_Std"] = df[paper_col].map(paper_to_std)
    else:
        df["_Paper_Std"] = "Unknown"
    return df, str(p)


def merge_retained_with_source(retained: pd.DataFrame, source: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = retained.copy()
    out["_Prediction_Instance_Index"] = np.arange(len(out), dtype=int)
    if source is None or "Data_Index" not in out.columns:
        out["_Source_Merge_Status"] = "NO_SOURCE_TABLE_OR_NO_DATA_INDEX"
        return out

    data_idx = pd.to_numeric(out["Data_Index"], errors="coerce")
    out["_Data_Index_Int"] = data_idx.astype("Int64")

    # Direct 0-based merge, which matches the rebuilt final Excel row order in this project.
    src = source.copy()
    merged = out.merge(
        src,
        how="left",
        left_on="_Data_Index_Int",
        right_on="_Source_Row_0Based",
        suffixes=("", "_source"),
    )
    merged["_Source_Merge_Status"] = np.where(
        merged["_Source_Row_0Based"].notna(), "MATCHED_BY_ZERO_BASED_DATA_INDEX", "UNMATCHED"
    )
    return merged


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def get_source_cols(source: Optional[pd.DataFrame]) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    if source is None:
        return ({k: None for k in NUMERIC_CANDIDATES}, {k: None for k in CATEGORICAL_CANDIDATES})
    num_cols = {k: find_col(source, cands) for k, cands in NUMERIC_CANDIDATES.items()}
    cat_cols = {k: find_col(source, cands) for k, cands in CATEGORICAL_CANDIDATES.items()}
    return num_cols, cat_cols


def numeric_stats_for_context(df: pd.DataFrame, label: str, columns: Dict[str, Optional[str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, col in columns.items():
        if col is None or col not in df.columns:
            continue
        vals = to_num(df[col]).dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "Context": label,
            "Variable_Key": key,
            "Variable_Column": col,
            "N": int(vals.shape[0]),
            "Mean": float(vals.mean()),
            "Std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
            "Min": float(vals.min()),
            "Q25": float(vals.quantile(0.25)),
            "Median": float(vals.median()),
            "Q75": float(vals.quantile(0.75)),
            "Max": float(vals.max()),
        })
    return rows


def categorical_counts(df: pd.DataFrame, label: str, columns: Dict[str, Optional[str]], top_n: int = 30) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    n_context = len(df)
    for key, col in columns.items():
        if col is None or col not in df.columns:
            continue
        vc = df[col].fillna("NA").astype(str).str.strip().replace("", "NA").value_counts(dropna=False).head(top_n)
        for category, count in vc.items():
            rows.append({
                "Context": label,
                "Variable_Key": key,
                "Variable_Column": col,
                "Category": category,
                "Count": int(count),
                "Fraction": float(count / n_context) if n_context else np.nan,
                "Plot_Label": clean_label(category),
            })
    return rows


def summarize_ad_components(retained: pd.DataFrame, context_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    frames = {"Retained prediction instances": retained, **context_frames}
    for ctx, df in frames.items():
        for col in AD_COMPONENTS:
            if col not in df.columns:
                continue
            vals = to_num(df[col]).dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "Context": ctx,
                "AD_Component": col,
                "N": int(len(vals)),
                "Mean": float(vals.mean()),
                "Std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "Median": float(vals.median()),
                "Q25": float(vals.quantile(0.25)),
                "Q75": float(vals.quantile(0.75)),
                "Min": float(vals.min()),
                "Max": float(vals.max()),
            })
    return pd.DataFrame(rows)


def subset_full_prediction_context(all_pred: Optional[pd.DataFrame], retained: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if all_pred is None:
        return {}
    strategy = str(retained.get("_32F2_Strategy", retained.get("Strategy", pd.Series([""]))).iloc[0])
    method = str(retained.get("_32F2_Method", retained.get("Method", pd.Series([""]))).iloc[0])
    paper = str(retained.get("文献编号", pd.Series([""])).iloc[0])

    context: Dict[str, pd.DataFrame] = {}
    m = pd.Series(True, index=all_pred.index)
    if "Strategy" in all_pred.columns and strategy:
        m &= all_pred["Strategy"].astype(str).eq(strategy)
    if "Method" in all_pred.columns and method:
        m &= all_pred["Method"].astype(str).eq(method)
    same_strategy_method = all_pred.loc[m].copy()
    if len(same_strategy_method):
        context["Same strategy-method prediction subset"] = same_strategy_method
    if "文献编号" in same_strategy_method.columns and paper:
        same_paper = same_strategy_method.loc[same_strategy_method["文献编号"].astype(str).eq(paper)].copy()
        if len(same_paper):
            context["Same strategy-method and retained paper"] = same_paper
    return context


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_prediction_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    y = to_num(df["Observed"])
    p = to_num(df["Predicted"])
    met = metric_values(y, p)
    ax.scatter(y, p, s=42, color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.45)
    ax.plot([0, 100], [0, 100], linestyle="--", color=RED, linewidth=1.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(title)
    txt = (
        f"$R^2$ = {met['R2']:.3f}\n"
        f"RMSE = {met['RMSE']:.2f}\n"
        f"MAE = {met['MAE']:.2f}\n"
        f"Prediction N = {met['N']}"
    )
    ax.text(0.06, 0.94, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY, alpha=0.90))
    polish_ax(ax)


def plot_operating_window(ax: plt.Axes, retained_unique: pd.DataFrame, num_cols: Dict[str, Optional[str]]) -> None:
    display = [
        ("Initial F", "initial_f", "mg/L"),
        ("Dose", "dose", "mM"),
        ("Influent pH", "pH_in", ""),
        ("Retention", "retention_time", "min"),
        ("Temperature", "temperature", "°C"),
        ("Removal", "removal", "%"),
    ]
    rows = []
    for name, key, unit in display:
        col = num_cols.get(key)
        if col is None or col not in retained_unique.columns:
            continue
        vals = to_num(retained_unique[col]).dropna()
        if len(vals) == 0:
            continue
        rows.append((name, unit, float(vals.min()), float(vals.median()), float(vals.max()), int(len(vals))))
    if not rows:
        ax.text(0.5, 0.5, "No numeric chemistry columns found", ha="center", va="center")
        ax.axis("off")
        return
    y_pos = np.arange(len(rows))[::-1]
    for yi, (name, unit, vmin, vmed, vmax, n) in zip(y_pos, rows):
        ax.plot([vmin, vmax], [yi, yi], color=BLUE, linewidth=4, alpha=0.75, solid_capstyle="round")
        ax.scatter([vmed], [yi], s=75, color=GREEN, edgecolor=DARK, linewidth=0.5, zorder=3)
        if math.isclose(vmin, vmax):
            label = f"{vmed:.2g} {unit}".strip()
        else:
            label = f"{vmin:.2g}–{vmax:.2g}; med={vmed:.2g} {unit}".strip()
        ax.text(vmax + (0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.get_xlim()[1] > ax.get_xlim()[0] else 0.1),
                yi, label, va="center", fontsize=9.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("Retained operating window")
    ax.set_xlabel("Value range (min–max; dot = median)")
    ax.margins(x=0.20)
    polish_ax(ax)


def plot_top_categories(ax: plt.Axes, df: pd.DataFrame, col: Optional[str], title: str, color: str) -> None:
    if col is None or col not in df.columns:
        ax.text(0.5, 0.5, f"Column not found: {title}", ha="center", va="center")
        ax.axis("off")
        return
    vc = df[col].fillna("NA").astype(str).str.strip().replace("", "NA").value_counts().head(10)
    labels = [clean_label(x, max_len=36) for x in vc.index]
    y = np.arange(len(vc))[::-1]
    ax.barh(y, vc.values, color=color, edgecolor=DARK, linewidth=0.7, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Unique retained row count")
    ax.set_title(title)
    for yi, v in zip(y, vc.values):
        ax.text(v + max(vc.values) * 0.02, yi, f"{int(v)}", va="center", fontsize=9.5)
    ax.set_xlim(0, max(vc.values) * 1.25 if len(vc) else 1)
    polish_ax(ax)


def plot_distribution_box(ax: plt.Axes, frames: List[Tuple[str, pd.DataFrame]], col: Optional[str], title: str, ylabel: str) -> None:
    if col is None:
        ax.text(0.5, 0.5, "Column not found", ha="center", va="center")
        ax.axis("off")
        return
    data = []
    labels = []
    for label, df in frames:
        if col in df.columns:
            vals = to_num(df[col]).dropna().to_numpy(dtype=float)
            if len(vals):
                data.append(vals)
                labels.append(label)
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    colors = [LIGHT_GRAY, ORANGE, GREEN, BLUE]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(DARK)
        patch.set_alpha(0.72)
    for median in bp["medians"]:
        median.set_color(DARK)
        median.set_linewidth(1.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
    polish_ax(ax)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def make_figures(
    fig_dir: Path,
    retained: pd.DataFrame,
    retained_chem: pd.DataFrame,
    retained_unique: pd.DataFrame,
    source: Optional[pd.DataFrame],
    full_paper_source: Optional[pd.DataFrame],
    all_pred_context: Dict[str, pd.DataFrame],
    num_cols: Dict[str, Optional[str]],
    cat_cols: Dict[str, Optional[str]],
    rep_summary: pd.DataFrame,
) -> None:
    configure_matplotlib()

    # Figure 184: overview dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    fig.suptitle("32G retained-subdomain chemistry characterization", fontweight="bold", y=0.98)
    plot_prediction_panel(axes[0, 0], retained, "Reproduced retained predictions")
    panel_label(axes[0, 0], "A")
    plot_operating_window(axes[0, 1], retained_unique, num_cols)
    panel_label(axes[0, 1], "B")
    plot_top_categories(axes[1, 0], retained_unique, cat_cols.get("coagulant"), "Coagulant composition", BLUE)
    panel_label(axes[1, 0], "C")
    plot_top_categories(axes[1, 1], retained_unique, cat_cols.get("interfering_ion"), "Interfering-ion composition", ORANGE)
    panel_label(axes[1, 1], "D")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_fig(fig, fig_dir, "184_32g_retained_chemistry_overview")

    # Figure 185: operating window vs source distributions
    source_frames: List[Tuple[str, pd.DataFrame]] = []
    if source is not None:
        source_frames.append(("All data", source))
    if full_paper_source is not None and len(full_paper_source):
        source_frames.append(("Paper_010 all", full_paper_source))
    source_frames.append(("Retained unique", retained_unique))
    variables = [
        ("dose", "Coagulant dose", "mM"),
        ("pH_in", "Influent pH", "pH"),
        ("initial_f", "Initial fluoride", "mg/L"),
        ("removal", "Observed removal", "%"),
        ("retention_time", "Retention time", "min"),
        ("temperature", "Water temperature", "°C"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    fig.suptitle("32G retained subdomain vs source-data distributions", fontweight="bold", y=0.98)
    for ax, (key, title, unit), lab in zip(axes.ravel(), variables, list("ABCDEF")):
        plot_distribution_box(ax, source_frames, num_cols.get(key), title, unit)
        panel_label(ax, lab)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_fig(fig, fig_dir, "185_32g_operating_window_vs_source")

    # Figure 186: AD components and rule context
    ad_cols_present = [c for c in AD_COMPONENTS if c in retained.columns]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle("32G AD-rule context for the retained subdomain", fontweight="bold", y=0.99)
    if ad_cols_present:
        vals = retained[ad_cols_present].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
        y = np.arange(len(vals))[::-1]
        axes[0].barh(y, vals.values, color=GREEN, edgecolor=DARK, alpha=0.88)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([v.replace("_", " ") for v in vals.index])
        axes[0].set_xlabel("Mean component risk in retained instances")
        axes[0].set_title("AD component profile")
        for yi, v in zip(y, vals.values):
            axes[0].text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9.5)
        axes[0].set_xlim(0, min(1.15, max(1.0, float(vals.max()) + 0.15)))
        polish_ax(axes[0])
    else:
        axes[0].text(0.5, 0.5, "No AD component columns found", ha="center", va="center")
        axes[0].axis("off")
    panel_label(axes[0], "A")

    # Reliability zone counts in same strategy-method context and retained subset
    ctx = all_pred_context.get("Same strategy-method prediction subset")
    if ctx is not None and "Reliability_Zone" in ctx.columns:
        order = ["Reliable", "Warning", "Outside"]
        vc_all = ctx["Reliability_Zone"].astype(str).value_counts().reindex(order).fillna(0)
        vc_ret = retained["Reliability_Zone"].astype(str).value_counts().reindex(order).fillna(0) if "Reliability_Zone" in retained.columns else pd.Series(0, index=order)
        x = np.arange(len(order))
        width = 0.36
        axes[1].bar(x - width / 2, vc_all.values, width=width, color=LIGHT_GRAY, edgecolor=DARK, label="Same strategy-method")
        axes[1].bar(x + width / 2, vc_ret.values, width=width, color=BLUE, edgecolor=DARK, label="Retained")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(order)
        axes[1].set_ylabel("Prediction-instance count")
        axes[1].set_title("Reliability-zone context")
        axes[1].legend(frameon=True)
        for xi, v in zip(x + width / 2, vc_ret.values):
            axes[1].text(xi, v + max(1, vc_all.max() * 0.015), f"{int(v)}", ha="center", fontsize=9)
        polish_ax(axes[1])
    else:
        axes[1].text(0.5, 0.5, "Full prediction context not available", ha="center", va="center")
        axes[1].axis("off")
    panel_label(axes[1], "B")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, fig_dir, "186_32g_ad_profile_and_rule_context")

    # Figure 187: representativeness and dominance
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle("32G retained-subdomain representativeness", fontweight="bold", y=0.99)
    if not rep_summary.empty:
        plot_df = rep_summary.copy()
        x = np.arange(len(plot_df))
        bars = axes[0].bar(x, plot_df["Unique_Row_Count"], color=[LIGHT_GRAY, ORANGE, GREEN], edgecolor=DARK, alpha=0.90)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(plot_df["Context"], rotation=18, ha="right")
        axes[0].set_ylabel("Unique experimental-row count")
        axes[0].set_title("Unique-row coverage")
        for bar, frac in zip(bars, plot_df["Fraction_of_All_Data"]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f"{int(bar.get_height())}\n({frac:.1%})", ha="center", va="bottom", fontsize=9)
        polish_ax(axes[0])
    else:
        axes[0].axis("off")
    panel_label(axes[0], "A")

    if "文献编号" in retained.columns:
        vc = retained["文献编号"].astype(str).value_counts().head(10)
        y = np.arange(len(vc))[::-1]
        axes[1].barh(y, vc.values, color=BLUE, edgecolor=DARK, alpha=0.90)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(vc.index)
        axes[1].set_xlabel("Retained prediction-instance count")
        axes[1].set_title("Literature-group dominance")
        for yi, v in zip(y, vc.values):
            axes[1].text(v + max(vc.values) * 0.02, yi, f"{int(v)}", va="center", fontsize=9.5)
        axes[1].set_xlim(0, max(vc.values) * 1.20)
        polish_ax(axes[1])
    else:
        axes[1].axis("off")
    panel_label(axes[1], "B")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, fig_dir, "187_32g_literature_dominance_and_representativeness")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    root = find_project_root()
    table_dir = root / "outputs" / "tables"
    fig_dir = root / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 92)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {root}")
    log("=" * 92)

    retained_path = first_existing([
        table_dir / "32f2_best_reproduced_retained_predictions.csv",
        root / "32f2_best_reproduced_retained_predictions.csv",
    ], required=True, name="32F2 retained prediction table")

    all_pred_path = first_existing([
        table_dir / "32b_predictions_with_reliability_zone.csv",
        root / "32b_predictions_with_reliability_zone.csv",
    ], required=False, name="32B full prediction-with-reliability table")

    retained = safe_read_csv(retained_path)
    all_pred = safe_read_csv(all_pred_path) if all_pred_path is not None else None
    source, source_info = load_source_chemistry(root)
    num_cols, cat_cols = get_source_cols(source)

    # Basic validation
    required_cols = ["Observed", "Predicted", "Data_Index", "Method", "Strategy"]
    missing = [c for c in required_cols if c not in retained.columns]
    if missing:
        raise ValueError(f"Retained prediction table is missing required columns: {missing}")

    # Merge retained predictions with source chemistry.
    retained_chem = merge_retained_with_source(retained, source)
    if "Data_Index" in retained_chem.columns:
        retained_unique = retained_chem.sort_values(["Data_Index", "Split_ID" if "Split_ID" in retained_chem.columns else "Data_Index"]).drop_duplicates("Data_Index")
    else:
        retained_unique = retained_chem.drop_duplicates()

    # Identify retained paper and full paper source context.
    retained_paper = str(retained.get("文献编号", pd.Series(["Unknown"])).iloc[0])
    full_paper_source = None
    if source is not None and "_Paper_Std" in source.columns:
        full_paper_source = source.loc[source["_Paper_Std"].astype(str).eq(retained_paper)].copy()

    # Prediction context from full 32B table.
    all_pred_context = subset_full_prediction_context(all_pred, retained)

    # Tables: retained with chemistry and unique rows.
    safe_to_csv(retained_chem, table_dir / "32g_retained_predictions_with_chemistry.csv")
    safe_to_csv(retained_unique, table_dir / "32g_unique_retained_chemical_rows.csv")

    # Numeric operating window summary.
    numeric_rows: List[Dict[str, object]] = []
    if source is not None:
        numeric_rows.extend(numeric_stats_for_context(source, "All source data", num_cols))
    if full_paper_source is not None and len(full_paper_source):
        numeric_rows.extend(numeric_stats_for_context(full_paper_source, f"{retained_paper} all source rows", num_cols))
    numeric_rows.extend(numeric_stats_for_context(retained_unique, "Retained unique experimental rows", num_cols))
    numeric_summary = pd.DataFrame(numeric_rows)
    safe_to_csv(numeric_summary, table_dir / "32g_numeric_operating_window.csv")

    # Categorical composition summary.
    cat_rows: List[Dict[str, object]] = []
    if source is not None:
        cat_rows.extend(categorical_counts(source, "All source data", cat_cols))
    if full_paper_source is not None and len(full_paper_source):
        cat_rows.extend(categorical_counts(full_paper_source, f"{retained_paper} all source rows", cat_cols))
    cat_rows.extend(categorical_counts(retained_unique, "Retained unique experimental rows", cat_cols))
    cat_summary = pd.DataFrame(cat_rows)
    safe_to_csv(cat_summary, table_dir / "32g_categorical_composition.csv")

    # AD component summary.
    ad_summary = summarize_ad_components(retained, all_pred_context)
    safe_to_csv(ad_summary, table_dir / "32g_ad_component_summary.csv")

    # Representativeness summary.
    rep_rows: List[Dict[str, object]] = []
    all_n = len(source) if source is not None else np.nan
    paper_n = len(full_paper_source) if full_paper_source is not None else np.nan
    retained_unique_n = retained_unique["Data_Index"].nunique() if "Data_Index" in retained_unique.columns else len(retained_unique)
    retained_instance_n = len(retained)
    if source is not None:
        rep_rows.append({
            "Context": "All source data",
            "Unique_Row_Count": int(all_n),
            "Prediction_Instance_Count": np.nan,
            "Fraction_of_All_Data": 1.0,
            "Fraction_of_Retained_Paper": np.nan,
        })
    if full_paper_source is not None:
        rep_rows.append({
            "Context": f"{retained_paper} all source rows",
            "Unique_Row_Count": int(paper_n),
            "Prediction_Instance_Count": np.nan,
            "Fraction_of_All_Data": float(paper_n / all_n) if all_n else np.nan,
            "Fraction_of_Retained_Paper": 1.0,
        })
    rep_rows.append({
        "Context": "Retained unique rows",
        "Unique_Row_Count": int(retained_unique_n),
        "Prediction_Instance_Count": int(retained_instance_n),
        "Fraction_of_All_Data": float(retained_unique_n / all_n) if all_n else np.nan,
        "Fraction_of_Retained_Paper": float(retained_unique_n / paper_n) if paper_n else np.nan,
    })
    rep_summary = pd.DataFrame(rep_rows)
    safe_to_csv(rep_summary, table_dir / "32g_representativeness_summary.csv")

    # Metrics and rule context.
    retained_metrics = metric_values(retained["Observed"], retained["Predicted"])
    policy_id = str(retained.get("_32F2_Policy_ID", pd.Series(["Unknown"])).iloc[0])
    method = str(retained.get("_32F2_Method", retained.get("Method", pd.Series(["Unknown"]))).iloc[0])
    strategy = str(retained.get("_32F2_Strategy", retained.get("Strategy", pd.Series(["Unknown"]))).iloc[0])
    rule_name = str(retained.get("_32F2_Rule_Name", pd.Series(["Unknown"])).iloc[0])
    reliability_zones = "; ".join(sorted(retained["Reliability_Zone"].astype(str).unique())) if "Reliability_Zone" in retained.columns else "NA"

    # Figures.
    make_figures(
        fig_dir=fig_dir,
        retained=retained,
        retained_chem=retained_chem,
        retained_unique=retained_unique,
        source=source,
        full_paper_source=full_paper_source,
        all_pred_context=all_pred_context,
        num_cols=num_cols,
        cat_cols=cat_cols,
        rep_summary=rep_summary,
    )

    # Decision report.
    report_lines = []
    report_lines.append("32G retained-subdomain chemistry/domain characterization report")
    report_lines.append("=" * 92)
    report_lines.append(f"Project root: {root}")
    report_lines.append(f"Retained prediction table: {retained_path}")
    report_lines.append(f"Full prediction table: {all_pred_path if all_pred_path else 'NOT AVAILABLE'}")
    report_lines.append(f"Source chemistry table: {source_info}")
    report_lines.append("")
    report_lines.append("Selected reproducible policy carried from Step 32F2")
    report_lines.append("-" * 92)
    report_lines.append(f"Policy ID: {policy_id}")
    report_lines.append(f"Strategy: {strategy}")
    report_lines.append(f"Method: {method}")
    report_lines.append(f"Rule name: {rule_name}")
    report_lines.append(f"Reliability zone(s) retained: {reliability_zones}")
    report_lines.append("")
    report_lines.append("Retained prediction-instance metrics")
    report_lines.append("-" * 92)
    report_lines.append(f"R2: {retained_metrics['R2']:.6f}")
    report_lines.append(f"RMSE: {retained_metrics['RMSE']:.6f}")
    report_lines.append(f"MAE: {retained_metrics['MAE']:.6f}")
    report_lines.append(f"Prediction-instance N: {int(retained_metrics['N'])}")
    report_lines.append(f"Unique experimental-row N: {int(retained_unique_n)}")
    report_lines.append(f"Dominant retained literature group: {retained_paper}")
    if not math.isnan(paper_n) and paper_n:
        report_lines.append(f"Retained unique rows / {retained_paper} all rows: {retained_unique_n}/{int(paper_n)} = {retained_unique_n / paper_n:.3f}")
    if not math.isnan(all_n) and all_n:
        report_lines.append(f"Retained unique rows / all source rows: {retained_unique_n}/{int(all_n)} = {retained_unique_n / all_n:.3f}")
    report_lines.append("")
    report_lines.append("Main retained-subdomain chemistry window")
    report_lines.append("-" * 92)
    if not numeric_summary.empty:
        retained_num = numeric_summary.loc[numeric_summary["Context"].eq("Retained unique experimental rows")]
        for _, row in retained_num.iterrows():
            report_lines.append(
                f"{row['Variable_Key']} ({row['Variable_Column']}): "
                f"min={row['Min']:.6g}, median={row['Median']:.6g}, max={row['Max']:.6g}, mean={row['Mean']:.6g}, n={int(row['N'])}"
            )
    else:
        report_lines.append("No numeric source-chemistry columns were available.")
    report_lines.append("")
    report_lines.append("Main categorical composition in retained unique rows")
    report_lines.append("-" * 92)
    if not cat_summary.empty:
        retained_cat = cat_summary.loc[cat_summary["Context"].eq("Retained unique experimental rows")]
        for key in ["coagulant", "interfering_ion", "fluoride_form", "paper"]:
            sub = retained_cat.loc[retained_cat["Variable_Key"].eq(key)].head(12)
            if len(sub):
                report_lines.append(f"[{key}]")
                for _, row in sub.iterrows():
                    report_lines.append(f"  {row['Category']}: n={int(row['Count'])}, fraction={row['Fraction']:.3f}")
    else:
        report_lines.append("No categorical source-chemistry columns were available.")
    report_lines.append("")
    report_lines.append("Interpretation guardrails")
    report_lines.append("-" * 92)
    report_lines.append("1. This is a target-free selective AD retained-subdomain result, not all-domain grouped-validation performance.")
    report_lines.append("2. Prediction-instance N and unique experimental-row N are different because repeated grouped splits may retain the same row more than once.")
    report_lines.append("3. If retained rows are dominated by one literature group, describe the result as a high-confidence literature/chemistry-specific subdomain, not universal generalization.")
    report_lines.append("4. The chemical window reported here should be used to explain what operating conditions the selective AD rule actually identifies.")
    report_lines.append("")
    report_lines.append("Generated outputs")
    report_lines.append("-" * 92)
    report_lines.append("Tables: 32g_retained_predictions_with_chemistry.csv; 32g_unique_retained_chemical_rows.csv; 32g_numeric_operating_window.csv; 32g_categorical_composition.csv; 32g_ad_component_summary.csv; 32g_representativeness_summary.csv")
    report_lines.append("Figures: 184_32g_retained_chemistry_overview; 185_32g_operating_window_vs_source; 186_32g_ad_profile_and_rule_context; 187_32g_literature_dominance_and_representativeness")

    report_path = table_dir / "32g_retained_subdomain_decision_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"[REPORT] {report_path}")

    elapsed = time.time() - t0
    log("=" * 92)
    log("32G retained-subdomain chemistry characterization finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    log("=" * 92)


if __name__ == "__main__":
    main()
