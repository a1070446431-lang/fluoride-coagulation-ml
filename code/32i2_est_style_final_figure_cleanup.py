# -*- coding: utf-8 -*-
"""
32i2_est_style_final_figure_cleanup.py

Purpose
-------
Step 32I2 for the fluoride coagulation ML project.

This script does NOT retrain any model, does NOT change the accepted selective
applicability-domain (AD) rule, and does NOT recalculate the modeling workflow.
It only rebuilds Step 32I summary figures in a cleaner, publication-ready style:

1. English-only labels to avoid missing Chinese glyph warnings in PDF/PNG outputs.
2. EST-like white-background figure style with readable font sizes.
3. Better spacing to avoid title/axis/annotation overlap.
4. Direct numeric annotations on bars/lines/scatter points.

Required input tables, searched automatically under outputs/tables:
- 32i_final_evidence_summary.csv
- 32i_transfer_support_top.csv
- 32h_threshold_sensitivity_all.csv
- 32h_threshold_plateau_summary.csv
- 32h_exact_policy_split_metrics.csv
- 32h_exact_policy_paper_dominance.csv
- 32g_numeric_operating_window.csv
- 32g_categorical_composition.csv
- 32g_ad_component_summary.csv

Expected outputs:
- outputs/figures/194_32i2_evidence_chain_summary_clean.png/pdf
- outputs/figures/195_32i2_threshold_plateau_evidence_clean.png/pdf
- outputs/figures/196_32i2_retained_chemistry_window_clean.png/pdf
- outputs/figures/197_32i2_stability_transfer_boundary_clean.png/pdf
- outputs/tables/32i2_figure_cleanup_report.txt

Important
---------
The accepted 32I result remains:
RepeatedBalanced_Extended | XGBoost | Reliability_Zone == Outside AND NN_Risk <= 0.814815.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 0. Global style
# =============================================================================

START_TIME = time.time()


def find_project_root() -> Path:
    """Find project root from this script location or current working directory."""
    candidates: List[Path] = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    for start in candidates:
        for p in [start] + list(start.parents):
            if (p / "outputs" / "tables").exists():
                return p
            if (p / "outputs").exists() and (p / "src").exists():
                return p
    # Fallback: when the script is placed in src but outputs does not exist yet
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Use Times New Roman on Windows when available; otherwise use a safe serif font.
def choose_serif_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Times New Roman", "Times", "DejaVu Serif"]:
        if name in available:
            return name
    return "DejaVu Serif"


FONT_SERIF = choose_serif_font()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [FONT_SERIF, "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "axes.labelcolor": "#111111",
    }
)

# Suppress only non-fatal font/glyph warnings. This script uses English-only labels,
# but some systems may still issue harmless font fallback warnings.
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
warnings.filterwarnings("ignore", message="findfont: Font family.*")


COLORS = {
    "blue": "#4C78A8",
    "light_blue": "#7FA9C9",
    "green": "#6DA77A",
    "light_green": "#A9C7B2",
    "orange": "#D9A04F",
    "red": "#C94E52",
    "gray": "#9AA7B8",
    "light_gray": "#D6D6D6",
    "dark_gray": "#333333",
}


# =============================================================================
# 1. IO helpers
# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def read_csv_required(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required table: {path}\n"
            f"Please make sure Step 32I/32H/32G outputs are present in outputs/tables."
        )
    df = pd.read_csv(path, low_memory=False)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df


def read_csv_optional(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        log(f"[WARN] Optional table not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log(f"[REPORT] Saved {path}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIGURE] Saved {png}")
    log(f"[FIGURE] Saved {pdf}")


# =============================================================================
# 2. Plot helpers
# =============================================================================


def est_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.5, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)


def annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.3f}",
    dy_frac: float = 0.02,
    fontsize: int = 10,
) -> None:
    y0, y1 = ax.get_ylim()
    span = max(abs(y1 - y0), 1e-9)
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        x = bar.get_x() + bar.get_width() / 2
        if h >= 0:
            y = h + dy_frac * span
            va = "bottom"
        else:
            y = h - dy_frac * span
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


def short_strategy(s: object) -> str:
    text = str(s)
    mapping = {
        "RepeatedBalanced_Extended": "RBE",
        "RepeatedOrdinary_GroupedSplit": "RO-GS",
        "GroupKFold_FullyGrouped": "GKF",
        "Balanced_5Seed_Current31F": "B5S",
        "LOLO_LeaveOneLiteratureOut": "LOLO",
    }
    return mapping.get(text, text[:18])


def short_method(s: object) -> str:
    text = str(s)
    mapping = {
        "KRR_CatBoost_Weighted_2Model": "KRR-CB",
        "KRR_XGBoost_Weighted_2Model": "KRR-XGB",
        "XGBoost": "XGB",
        "CatBoost": "CatBoost",
        "CB-XGB": "CB-XGB",
        "KRR_Poly": "KRR-Poly",
        "GPR_Matern": "GPR",
    }
    return mapping.get(text, text.replace("_", "-")[:16])


def metric_ylim(vals: Iterable[float], pad_frac: float = 0.16, lower_zero: bool = False) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if pd.notna(v) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (0.0, 1.0)
    mn, mx = float(arr.min()), float(arr.max())
    if np.isclose(mn, mx):
        mn -= 1.0
        mx += 1.0
    span = mx - mn
    lo = mn - span * pad_frac
    hi = mx + span * pad_frac
    if lower_zero:
        lo = min(0.0, lo)
    return lo, hi


def get_retained_context(df: pd.DataFrame, preferred: Optional[str] = None) -> pd.DataFrame:
    if df.empty or "Context" not in df.columns:
        return df.iloc[0:0].copy()
    ctx = df["Context"].astype(str)
    if preferred is not None:
        exact = ctx.eq(preferred)
        if exact.any():
            return df.loc[exact].copy()
    mask = ctx.str.contains("Retained", case=False, na=False)
    return df.loc[mask].copy()


def variable_label(key: str) -> str:
    mapping = {
        "initial_f": "Initial F- (mg/L)",
        "dose": "Dose (mM)",
        "pH_in": "Influent pH",
        "retention_time": "Retention time (min)",
        "temperature": "Temperature (deg C)",
        "interfering_conc": "Interfering ion conc. (mmol/L)",
        "removal": "Observed F removal (%)",
        "effluent_pH": "Effluent pH",
        "effluent_f": "Effluent F- (mg/L)",
        "free_f_fraction": "Free-F fraction",
    }
    return mapping.get(str(key), str(key).replace("_", " "))


def component_label(name: str) -> str:
    return (
        str(name)
        .replace("_", " ")
        .replace("Risk", "risk")
        .replace("AD", "AD")
        .replace("NN", "NN")
        .replace("PCA", "PCA")
    )


def clean_coagulant_label(label: object) -> str:
    text = str(label)
    # Prefer pre-existing English Plot_Label from 32G when possible, then normalize.
    repl = {
        "CSAF (CSAF, Al/Fe=4:1)": "CSAF, Al/Fe=4:1",
        "Al sulfate (CSAFsingle-Al)": "Al sulfate, CSAF single-Al",
        "Al sulfate (CSAMsingle-Al)": "Al sulfate, CSAM single-Al",
        "PAC (single-Al)": "PAC, single-Al",
        "CSAF (CSAF, Al/Fe=14:1)": "CSAF, Al/Fe=14:1",
        "CSAF (CSAF, Al/Fe=9:1)": "CSAF, Al/Fe=9:1",
        "CSAF (CSAF, Al/Fe=6.5:1)": "CSAF, Al/Fe=6.5:1",
        "CSAF (CSAF, Al/Fe=1:1)": "CSAF, Al/Fe=1:1",
        "PAFC (Al/Fe)": "PAFC, Al/Fe",
    }
    if text in repl:
        return repl[text]
    # Fallback transliterations for common Chinese terms.
    text = text.replace("复合硫酸铝铁", "CSAF")
    text = text.replace("硫酸铝", "Al sulfate")
    text = text.replace("聚合氯化铝铁", "PAFC")
    text = text.replace("聚合氯化铝", "PAC")
    text = text.replace("单铝体系", "single-Al")
    text = text.replace("（", " (").replace("）", ")")
    text = text.replace("，", ", ")
    text = re.sub(r"[^A-Za-z0-9,./:=+()\- ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "Unknown"


# =============================================================================
# 3. Figures
# =============================================================================


def plot_evidence_chain(summary: pd.DataFrame) -> None:
    df = summary.copy()
    order = [
        "All-domain grouped validation",
        "Predefined reliable domain",
        "Target-free selective retained subdomain",
    ]
    df["Evidence_Level"] = pd.Categorical(df["Evidence_Level"], categories=order, ordered=True)
    df = df.sort_values("Evidence_Level").reset_index(drop=True)

    labels = ["All-domain\ngrouped validation", "Predefined\nreliable domain", "Selective AD\nretained subdomain"]
    colors = [COLORS["blue"], COLORS["gray"], COLORS["green"]]
    metrics = [("R2", r"$R^2$", "{:.3f}"), ("RMSE", "RMSE", "{:.2f}"), ("MAE", "MAE", "{:.2f}"), ("Coverage", "Coverage", "{:.3f}")]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    fig.suptitle("32I2 evidence-chain summary for selective AD result", fontsize=18, fontweight="bold", y=1.02)

    for ax, (col, title, fmt) in zip(axes.ravel(), metrics):
        vals = pd.to_numeric(df[col], errors="coerce").values
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor=COLORS["dark_gray"], linewidth=0.8)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=10)
        lo, hi = metric_ylim(vals, lower_zero=(col != "R2"))
        if col == "R2":
            lo = min(lo, -0.25)
            hi = max(hi, 0.88)
            ax.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.3)
            ax.text(-0.45, 0.515, r"$R^2$ target = 0.5", color=COLORS["red"], fontsize=10, va="bottom")
        elif col == "Coverage":
            lo, hi = 0.0, max(1.12, np.nanmax(vals) * 1.2)
        ax.set_ylim(lo, hi)
        annotate_bars(ax, bars, fmt=fmt, fontsize=10)
        est_axis(ax)

    save_figure(fig, "194_32i2_evidence_chain_summary_clean")


def plot_threshold_plateau(thresh: pd.DataFrame, plateau: pd.DataFrame) -> None:
    df = thresh.copy()
    # Focus on the accepted P08 configuration.
    for col, val in [("Strategy", "RepeatedBalanced_Extended"), ("Method", "XGBoost"), ("Zone", "Outside"), ("Score_Column", "NN_Risk")]:
        if col in df.columns:
            df = df[df[col].astype(str) == val]
    df = df.sort_values("Threshold")

    if df.empty:
        log("[WARN] No threshold-sensitivity rows available for accepted policy; threshold figure skipped.")
        return

    tmin = tmax = exact = None
    if not plateau.empty:
        tmin = float(pd.to_numeric(plateau.get("Threshold_Min"), errors="coerce").dropna().iloc[0])
        tmax = float(pd.to_numeric(plateau.get("Threshold_Max"), errors="coerce").dropna().iloc[0])
    if "Is_Exact_Base_Threshold" in df.columns and df["Is_Exact_Base_Threshold"].astype(str).str.lower().eq("true").any():
        exact = float(df.loc[df["Is_Exact_Base_Threshold"].astype(str).str.lower().eq("true"), "Threshold"].iloc[-1])
    else:
        exact = 0.814815

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    fig.suptitle("32I2 threshold-plateau evidence for the accepted selective AD rule", fontsize=17, fontweight="bold", y=1.05)

    panels = [
        ("R2", r"Retained-subdomain $R^2$", r"$R^2$"),
        ("N_Retained", "Retained sample count", "N retained"),
        ("Coverage_within_strategy_method", "Coverage within strategy-method subset", "Coverage"),
    ]

    for ax, (col, ylab, title) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce")
        xs = pd.to_numeric(df["Threshold"], errors="coerce")
        ax.plot(xs, vals, marker="o", markersize=4.2, color=COLORS["blue"], linewidth=1.8, alpha=0.88)
        if tmin is not None and tmax is not None:
            ax.axvspan(tmin, tmax, color=COLORS["green"], alpha=0.13, lw=0)
        ax.axvline(exact, color=COLORS["dark_gray"], linestyle=":", linewidth=1.4)
        if col == "R2":
            ax.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.2)
            ax.set_ylim(-0.03, max(0.85, float(np.nanmax(vals)) * 1.1))
        elif col == "Coverage_within_strategy_method":
            ax.axhline(0.005, color=COLORS["red"], linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_ylim(-0.01, max(0.56, float(np.nanmax(vals)) * 1.12))
        else:
            ax.axhline(30, color=COLORS["red"], linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_ylim(0, max(60, float(np.nanmax(vals)) * 1.12))
        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=9)
        ax.set_xlabel("NN-risk threshold", fontsize=11.5)
        ax.set_ylabel(ylab, fontsize=11.5)
        est_axis(ax)

    save_figure(fig, "195_32i2_threshold_plateau_evidence_clean")


def plot_retained_chemistry(numeric: pd.DataFrame, categorical: pd.DataFrame, ad_components: pd.DataFrame) -> None:
    num = get_retained_context(numeric, preferred="Retained unique experimental rows")
    cat = get_retained_context(categorical, preferred="Retained unique experimental rows")
    ad = get_retained_context(ad_components, preferred="Retained prediction instances")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), constrained_layout=True)
    fig.suptitle("32I2 retained-subdomain chemistry and AD characterization", fontsize=17, fontweight="bold", y=1.02)

    # A. Numeric operating window
    ax = axes[0, 0]
    keep_keys = ["initial_f", "dose", "pH_in", "retention_time", "temperature", "interfering_conc", "removal", "effluent_pH"]
    sub = num[num["Variable_Key"].astype(str).isin(keep_keys)].copy()
    sub["_order"] = sub["Variable_Key"].astype(str).map({k: i for i, k in enumerate(keep_keys)})
    sub = sub.sort_values("_order", ascending=False)
    y = np.arange(len(sub))
    mins = pd.to_numeric(sub["Min"], errors="coerce").values
    meds = pd.to_numeric(sub["Median"], errors="coerce").values
    maxs = pd.to_numeric(sub["Max"], errors="coerce").values
    ax.hlines(y, mins, maxs, color=COLORS["blue"], linewidth=2.0, alpha=0.85)
    ax.scatter(mins, y, s=34, color=COLORS["gray"], edgecolor=COLORS["dark_gray"], linewidth=0.5, zorder=3, label="Min/Max")
    ax.scatter(maxs, y, s=34, color=COLORS["gray"], edgecolor=COLORS["dark_gray"], linewidth=0.5, zorder=3)
    ax.scatter(meds, y, s=52, color=COLORS["green"], edgecolor=COLORS["dark_gray"], linewidth=0.6, zorder=4, label="Median")
    ax.set_yticks(y)
    ax.set_yticklabels([variable_label(k) for k in sub["Variable_Key"].astype(str)], fontsize=10.2)
    ax.set_xlabel("Value on variable-specific scale", fontsize=11.5)
    ax.set_title("Retained chemistry window", fontsize=13.5, fontweight="bold", pad=9)
    for yy, mn, med, mx in zip(y, mins, meds, maxs):
        ax.text(mx + max(0.25, 0.012 * max(maxs)), yy, f"{mn:g}-{med:g}-{mx:g}", fontsize=8.7, va="center")
    est_axis(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # B. Coagulant composition
    ax = axes[0, 1]
    co = cat[cat["Variable_Key"].astype(str).eq("coagulant")].copy()
    co["Count"] = pd.to_numeric(co.get("Count"), errors="coerce").fillna(0)
    co = co.sort_values("Count", ascending=False).head(8)
    if "Plot_Label" in co.columns:
        labels = [clean_coagulant_label(v) for v in co["Plot_Label"]]
    else:
        labels = [clean_coagulant_label(v) for v in co.get("Category", [])]
    vals = co["Count"].values
    y = np.arange(len(co))
    bars = ax.barh(y, vals, color=COLORS["blue"], edgecolor=COLORS["dark_gray"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlabel("Unique retained row count", fontsize=11.5)
    ax.set_title("Top retained coagulant compositions", fontsize=13.5, fontweight="bold", pad=9)
    xmax = max(vals) if len(vals) else 1
    ax.set_xlim(0, xmax * 1.28)
    for bar in bars:
        ax.text(bar.get_width() + xmax * 0.03, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.0f}", va="center", fontsize=9)
    est_axis(ax)

    # C. Literature-group composition
    ax = axes[1, 0]
    paper = cat[cat["Variable_Key"].astype(str).eq("paper")].copy()
    paper["Count"] = pd.to_numeric(paper.get("Count"), errors="coerce").fillna(0)
    paper = paper.sort_values("Count", ascending=False).head(8)
    p_labels = [str(v) for v in paper.get("Plot_Label", paper.get("Category", []))]
    p_vals = paper["Count"].values
    p_frac = pd.to_numeric(paper.get("Fraction"), errors="coerce").fillna(0).values
    y = np.arange(len(paper))
    bars = ax.barh(y, p_vals, color=COLORS["green"], edgecolor=COLORS["dark_gray"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(p_labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Unique retained row count", fontsize=11.5)
    ax.set_title("Retained literature-group composition", fontsize=13.5, fontweight="bold", pad=9)
    xmax = max(p_vals) if len(p_vals) else 1
    ax.set_xlim(0, xmax * 1.34)
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + xmax * 0.03, bar.get_y() + bar.get_height() / 2, f"{p_vals[i]:.0f} ({p_frac[i]*100:.1f}%)", va="center", fontsize=9.5)
    est_axis(ax)

    # D. AD component median levels
    ax = axes[1, 1]
    if not ad.empty:
        ad = ad.sort_values("Median", ascending=True).tail(8)
        labels = [component_label(v) for v in ad["AD_Component"]]
        vals = pd.to_numeric(ad["Median"], errors="coerce").values
        y = np.arange(len(ad))
        bars = ax.barh(y, vals, color=COLORS["gray"], edgecolor=COLORS["dark_gray"], linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.8)
        ax.set_xlim(0, 1.12)
        for bar in bars:
            ax.text(min(bar.get_width() + 0.015, 1.08), bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", fontsize=9)
    ax.set_xlabel("Median component value", fontsize=11.5)
    ax.set_title("Median AD-component levels", fontsize=13.5, fontweight="bold", pad=9)
    est_axis(ax)

    save_figure(fig, "196_32i2_retained_chemistry_window_clean")


def plot_stability_transfer_boundary(split_metrics: pd.DataFrame, transfer: pd.DataFrame, dominance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    fig.suptitle("32I2 stability, transfer support, and interpretation boundary", fontsize=17, fontweight="bold", y=1.05)

    # A. split-level R2
    ax = axes[0]
    sm = split_metrics.copy()
    sm = sm.sort_values("R2", ascending=False)
    labels = [str(s).replace("balanced_ext_", "") for s in sm.get("Split_ID", [])]
    vals = pd.to_numeric(sm.get("R2"), errors="coerce").values
    colors = [COLORS["green"] if v >= 0.5 else COLORS["orange"] for v in vals]
    bars = ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor=COLORS["dark_gray"], linewidth=0.8)
    ax.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.2)
    ax.set_title(r"Split-level retained-subdomain $R^2$", fontsize=13.5, fontweight="bold", pad=9)
    ax.set_ylabel(r"$R^2$", fontsize=12)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.4)
    ax.set_ylim(min(-0.05, np.nanmin(vals) - 0.15), max(0.82, np.nanmax(vals) + 0.12))
    annotate_bars(ax, bars, fmt="{:.2f}", fontsize=9.5)
    est_axis(ax)

    # B. transfer support
    ax = axes[1]
    tf = transfer.copy().sort_values("R2", ascending=True).tail(8)
    if not tf.empty:
        labels = [f"{short_strategy(r.Strategy)}-{short_method(r.Method)}" for _, r in tf.iterrows()]
        vals = pd.to_numeric(tf["R2"], errors="coerce").values
        cov = pd.to_numeric(tf.get("Coverage_within_strategy_method"), errors="coerce").fillna(np.nan).values
        y = np.arange(len(tf))
        bars = ax.barh(y, vals, color=COLORS["blue"], edgecolor=COLORS["dark_gray"], linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.2)
        ax.axvline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.2)
        ax.set_xlim(0, max(0.82, np.nanmax(vals) * 1.22))
        for bar, r2v, cv in zip(bars, vals, cov):
            ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"R2={r2v:.3f}; cov={cv:.3f}", va="center", fontsize=8.4)
    ax.set_title("Transfer support at exact threshold", fontsize=13.5, fontweight="bold", pad=9)
    ax.set_xlabel(r"Retained-subdomain $R^2$", fontsize=11.5)
    est_axis(ax)

    # C. paper dominance
    ax = axes[2]
    dom = dominance.copy()
    if not dom.empty:
        if "Threshold" in dom.columns:
            # Select rows at the accepted exact threshold only. This avoids mixing
            # paper-dominance rows from neighboring thresholds.
            dom["_dist"] = (pd.to_numeric(dom["Threshold"], errors="coerce") - 0.814815).abs()
            min_dist = float(dom["_dist"].min())
            dom = dom.loc[dom["_dist"] <= min_dist + 1e-9].sort_values("Rank").head(6)
        labels = [str(v) for v in dom.get("Literature_Group", dom.get("Dominant_Group", []))]
        if "N" in dom.columns:
            vals = pd.to_numeric(dom["N"], errors="coerce").fillna(0).values
            frac = pd.to_numeric(dom.get("Fraction_within_retained"), errors="coerce").fillna(0).values
        else:
            vals = pd.to_numeric(dom.get("Dominant_Group_N"), errors="coerce").fillna(0).values
            frac = pd.to_numeric(dom.get("Dominant_Group_Fraction"), errors="coerce").fillna(0).values
        order = np.argsort(vals)
        vals = vals[order]
        frac = frac[order]
        labels = [labels[i] for i in order]
        y = np.arange(len(vals))
        bars = ax.barh(y, vals, color=COLORS["gray"], edgecolor=COLORS["dark_gray"], linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        xmax = max(vals) if len(vals) else 1
        ax.set_xlim(0, xmax * 1.28)
        for bar, n, fr in zip(bars, vals, frac):
            ax.text(bar.get_width() + xmax * 0.03, bar.get_y() + bar.get_height() / 2, f"{n:.0f} ({fr*100:.1f}%)", va="center", fontsize=9)
    ax.set_title("Literature-group dominance", fontsize=13.5, fontweight="bold", pad=9)
    ax.set_xlabel("Retained prediction-instance count", fontsize=11.5)
    est_axis(ax)

    save_figure(fig, "197_32i2_stability_transfer_boundary_clean")


# =============================================================================
# 4. Main
# =============================================================================


def main() -> None:
    log("=" * 110)
    log("Running 32i2_est_style_final_figure_cleanup.py")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Font selected: {FONT_SERIF}")
    log("=" * 110)

    summary = read_csv_required("32i_final_evidence_summary.csv")
    transfer = read_csv_required("32i_transfer_support_top.csv")
    threshold = read_csv_required("32h_threshold_sensitivity_all.csv")
    plateau = read_csv_required("32h_threshold_plateau_summary.csv")
    split_metrics = read_csv_required("32h_exact_policy_split_metrics.csv")
    dominance = read_csv_required("32h_exact_policy_paper_dominance.csv")
    numeric = read_csv_required("32g_numeric_operating_window.csv")
    categorical = read_csv_required("32g_categorical_composition.csv")
    ad_components = read_csv_required("32g_ad_component_summary.csv")

    plot_evidence_chain(summary)
    plot_threshold_plateau(threshold, plateau)
    plot_retained_chemistry(numeric, categorical, ad_components)
    plot_stability_transfer_boundary(split_metrics, transfer, dominance)

    # Build concise audit report
    selected = summary[summary["Evidence_Level"].astype(str).str.contains("selective", case=False, na=False)].copy()
    if not selected.empty:
        s = selected.iloc[0]
        selected_line = (
            f"Accepted selective AD result: Strategy={s.get('Strategy')}; Method={s.get('Method')}; "
            f"Rule={s.get('Rule')}; R2={float(s.get('R2')):.6f}; RMSE={float(s.get('RMSE')):.6f}; "
            f"MAE={float(s.get('MAE')):.6f}; Coverage={float(s.get('Coverage')):.6f}; "
            f"N_Retained={s.get('N_Retained')}; Dominant_Group={s.get('Dominant_Group')}."
        )
    else:
        selected_line = "Accepted selective AD row was not found in 32i_final_evidence_summary.csv."

    report = f"""
32I2 EST-style final figure cleanup report
{'=' * 110}

What this script did
--------------------
- Replotted Step 32I figures with English-only labels to avoid missing Chinese glyph warnings.
- Did not retrain models.
- Did not change the accepted selective AD rule.
- Did not change any metric values.
- Improved label spacing, panel layout, and direct numeric annotations.

{selected_line}

Output figures
--------------
- 194_32i2_evidence_chain_summary_clean.png/pdf
- 195_32i2_threshold_plateau_evidence_clean.png/pdf
- 196_32i2_retained_chemistry_window_clean.png/pdf
- 197_32i2_stability_transfer_boundary_clean.png/pdf

Font used
---------
{FONT_SERIF}

Important manuscript boundary
-----------------------------
This is retained-subdomain performance after target-free selective AD filtering, not all-domain grouped-validation performance.
""".strip()
    write_text(TABLE_DIR / "32i2_figure_cleanup_report.txt", report)

    elapsed = time.time() - START_TIME
    log("=" * 110)
    log("32I2 EST-style final figure cleanup finished successfully.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    log("=" * 110)


if __name__ == "__main__":
    main()
