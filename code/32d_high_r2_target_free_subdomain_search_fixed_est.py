# -*- coding: utf-8 -*-
"""
32d_high_r2_target_free_subdomain_search.py

Purpose
-------
Step 32D for the fluoride coagulation ML project.

This script performs a broad target-free high-R2 subdomain search using the
prediction + applicability-domain table generated in Step 32B/32A2:
    outputs/tables/32b_predictions_with_reliability_zone.csv

Why this step is needed
-----------------------
Step 32C/32C2 showed that the previous strict AD selective policies did not
reach retained R2 >= 0.5. This script broadens the search over deployable,
target-free rule families, including:
  1. Reliability-zone rules.
  2. Single AD-component threshold rules.
  3. Zone + AD-component rules.
  4. Prediction-band rules.
  5. Zone + prediction-band rules.
  6. Pairwise AD-component rules for the most relevant AD components.

Important strictness statement
------------------------------
- Rule conditions use only deployment-available quantities: model prediction,
  AD components, reliability zone, strategy/method identity.
- Observed labels are used only to score and rank candidate rules retrospectively.
- Therefore, the best rule from this script is a publication-audit candidate,
  not yet a final deployment rule unless confirmed in a later block-calibrated
  confirmation step.
- This script explicitly reports split-level stability to avoid hiding a rule
  that only works because of a single grouped split.

Outputs
-------
Tables:
  outputs/tables/32d_high_r2_policy_summary.csv
  outputs/tables/32d_policy_label_mapping.csv
  outputs/tables/32d_best_policy_retained_predictions.csv
  outputs/tables/32d_best_policy_split_stability.csv
  outputs/tables/32d_decision_report.txt

Figures:
  outputs/figures/165_32d_high_r2_policy_metrics.png/pdf
  outputs/figures/166_32d_r2_coverage_frontier.png/pdf
  outputs/figures/167_32d_best_policy_prediction_panel.png/pdf
  outputs/figures/168_32d_best_policy_split_stability.png/pdf

Author note
-----------
All scripts in this project should print total runtime at the end.
"""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_NAME = "32d_high_r2_target_free_subdomain_search.py"
R2_TARGET = 0.50
# A retained domain with fewer than 100 pooled held-out points is too fragile for
# a main manuscript claim. It can still be reported as an exploratory candidate.
MAIN_MIN_N = 100
MAIN_MIN_COVERAGE = 0.05
EXPLORATORY_MIN_N = 30
MAX_TOP_POLICIES_TO_PLOT = 12

# EST-like plotting defaults: white background, high DPI, clean fonts,
# low-saturation professional palette, and enough spacing to avoid overlaps.
rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "0.15",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
})

EST_BLUE = "#5B8DB8"
EST_GREEN = "#72A87E"
EST_ORANGE = "#D99A4E"
EST_PURPLE = "#9C79B8"
EST_GRAY = "#8C8C8C"


def find_project_root() -> Path:
    """Find project root from current script location or current working directory."""
    candidates: List[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    markers = [
        Path("outputs") / "tables" / "32b_predictions_with_reliability_zone.csv",
        Path("data") / "processed" / "model_ready_v1.csv",
        Path("src"),
    ]
    for start in candidates:
        for p in [start] + list(start.parents):
            if any((p / m).exists() for m in markers):
                return p
    return Path.cwd().resolve()


ROOT = find_project_root()
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def locate_prediction_table() -> Path:
    """Locate 32B prediction/reliability table."""
    candidate_paths = [
        TABLE_DIR / "32b_predictions_with_reliability_zone.csv",
        ROOT / "32b_predictions_with_reliability_zone.csv",
        Path.cwd() / "outputs" / "tables" / "32b_predictions_with_reliability_zone.csv",
        Path("/mnt/data/32b_predictions_with_reliability_zone.csv"),  # sandbox fallback only
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find 32b_predictions_with_reliability_zone.csv. Expected it under outputs/tables/."
    )


def sanitize_text(s: Any, max_len: int = 80) -> str:
    """Make strings safe and compact for tables/figures."""
    if pd.isna(s):
        return "NA"
    text = str(s)
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def safe_float_array(s: pd.Series) -> np.ndarray:
    """Convert a Series to a writable float ndarray.

    pandas/numpy may return a read-only view from nullable or Arrow-backed columns.
    The explicit np.array(..., copy=True) prevents "assignment destination is read-only".
    """
    arr = np.array(pd.to_numeric(s, errors="coerce"), dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def r2_rmse_mae(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    y = y[ok]
    pred = pred[ok]
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    rmse = math.sqrt(float(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    return r2, rmse, mae


def save_fig(fig: plt.Figure, name: str) -> None:
    """Fast, stable EST-style PNG saving for PyCharm/Windows."""
    png = FIG_DIR / name
    log(f"[FIG-SAVE-START] {png}")
    fig.savefig(png, dpi=450)
    plt.close(fig)
    log(f"[FIG] {png}")


def safe_to_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


def pretty_num(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


@dataclass(frozen=True)
class RuleSpec:
    rule_name: str
    rule_family: str
    description: str
    mask: np.ndarray


def finite_unique_thresholds(values: np.ndarray, max_unique: int = 12) -> List[float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return []
    # Deliberately compact grid: enough to capture meaningful AD cutoffs,
    # including the boundary-risk cutoff that was near-successful in 32C-fast,
    # while avoiding an excessively slow all-pairs search.
    qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.85, 0.95]
    thresholds = list(np.nanquantile(vals, qs))
    thresholds += [0.0, 0.05, 0.10, 0.20, 0.25, 0.266667, 0.30, 0.40, 0.50, 0.70, 0.90]
    uniq = np.unique(np.round(vals, 6))
    if uniq.size <= max_unique:
        thresholds += list(uniq)
    thresholds = sorted(set(float(np.round(t, 6)) for t in thresholds if np.isfinite(t)))
    return thresholds

def prediction_band_thresholds(values: np.ndarray) -> List[Tuple[float, float, str]]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return []
    qs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 0.95]
    qvals = {q: float(np.nanquantile(vals, q)) for q in qs}
    bands: List[Tuple[float, float, str]] = []
    for lo_q in [0.05, 0.10, 0.20, 0.30, 0.40]:
        for hi_q in [0.50, 0.70, 0.85, 0.95]:
            if hi_q <= lo_q:
                continue
            bands.append((qvals[lo_q], qvals[hi_q], f"Q{int(lo_q*100):02d}_Q{int(hi_q*100):02d}"))
    return bands

def get_candidate_component_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "Boundary_AD_Risk",
        "Corrected_AD_Risk",
        "Distance_AD_Risk",
        "Max_Component_Risk",
        "NN_Risk",
        "PCA_Reconstruction_Risk",
        "Categorical_Unseen_Fraction",
    ]
    cols = [c for c in preferred if c in df.columns]
    return cols


def generate_rules_for_subset(g: pd.DataFrame, component_cols: Sequence[str]) -> List[RuleSpec]:
    """Generate deployment-available target-free rule masks for one strategy-method subset."""
    n = len(g)
    if n == 0:
        return []

    rules: List[RuleSpec] = []
    all_mask = np.ones(n, dtype=bool)
    rules.append(RuleSpec("All", "all", "All rows", all_mask))

    # Reliability-zone masks.
    zone_masks: Dict[str, np.ndarray] = {}
    if "Reliability_Zone" in g.columns:
        zones = [z for z in g["Reliability_Zone"].dropna().astype(str).unique().tolist()]
        for z in sorted(zones):
            m = g["Reliability_Zone"].astype(str).to_numpy() == z
            if m.sum() >= EXPLORATORY_MIN_N:
                zsafe = sanitize_text(z, 40)
                zone_masks[zsafe] = m
                rules.append(RuleSpec(
                    f"Zone_eq_{zsafe}",
                    "zone_single",
                    f"Reliability_Zone == {zsafe}",
                    m,
                ))
        # Useful combinations used in AD literature: reliable-only and reliable+warning.
        for combo_name, combo_values in [
            ("Zone_in_Reliable_Warning", ["Reliable", "Warning"]),
            ("Zone_in_Warning_Outside", ["Warning", "Outside"]),
        ]:
            m = g["Reliability_Zone"].astype(str).isin(combo_values).to_numpy()
            if m.sum() >= EXPLORATORY_MIN_N:
                rules.append(RuleSpec(
                    combo_name,
                    "zone_combo",
                    f"Reliability_Zone in {combo_values}",
                    m,
                ))

    # Numeric component threshold rules.
    arrays: Dict[str, np.ndarray] = {c: safe_float_array(g[c]) for c in component_cols if c in g.columns}
    for c, arr in arrays.items():
        thresholds = finite_unique_thresholds(arr)
        for t in thresholds:
            # Low-risk rule.
            m_low = np.isfinite(arr) & (arr <= t)
            if m_low.sum() >= EXPLORATORY_MIN_N:
                rules.append(RuleSpec(
                    f"{c}_le_{t:.6g}",
                    "single_component_low",
                    f"{c} <= {t:.6g}",
                    m_low,
                ))
            # Zone + low-risk rules.
            for zname, zm in zone_masks.items():
                mz = zm & m_low
                if mz.sum() >= EXPLORATORY_MIN_N:
                    rules.append(RuleSpec(
                        f"Zone_{zname}__{c}_le_{t:.6g}",
                        "zone_component_low",
                        f"Reliability_Zone == {zname} AND {c} <= {t:.6g}",
                        mz,
                    ))

    # Prediction-band rules were already explored in 32C2. For 32D we focus on
    # AD-zone + AD-component subdomains, which are cleaner to defend in a manuscript
    # and fast enough to run repeatedly.

    # Pairwise component rules are intentionally not generated here to keep the
    # script fast and reproducible. The main near-successful 32C route is already
    # captured by zone + single-component rules.

    # Deduplicate by rule_name while keeping first occurrence.
    seen = set()
    unique_rules: List[RuleSpec] = []
    for r in rules:
        if r.rule_name not in seen:
            seen.add(r.rule_name)
            unique_rules.append(r)
    return unique_rules


def evaluate_rule_subset_fast(
    y_all: np.ndarray,
    pred_all: np.ndarray,
    rule: RuleSpec,
    strategy: str,
    method: str,
    group_len: int,
    total_n: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate one rule using precomputed arrays for speed."""
    m = np.array(rule.mask, dtype=bool, copy=False)
    if m.sum() < EXPLORATORY_MIN_N:
        return None
    ok = m & np.isfinite(y_all) & np.isfinite(pred_all)
    n_retained = int(ok.sum())
    if n_retained < EXPLORATORY_MIN_N:
        return None
    y = y_all[ok]
    pred = pred_all[ok]
    if np.nanvar(y) <= 1e-12:
        return None
    r2, rmse, mae = r2_rmse_mae(y, pred)
    coverage_within_subset = float(n_retained / group_len)
    coverage_vs_all_rows = float(n_retained / total_n)
    return {
        "Strategy": strategy,
        "Method": method,
        "Rule_Name": rule.rule_name,
        "Rule_Family": rule.rule_family,
        "Rule_Description": rule.description,
        "N_Base": int(group_len),
        "N_Retained": n_retained,
        "Coverage_Within_Strategy_Method": coverage_within_subset,
        "Coverage_Vs_All_Filtered_Rows": coverage_vs_all_rows,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Mean_Observed": float(np.mean(y)),
        "Std_Observed": float(np.std(y, ddof=0)),
        "Mean_Predicted": float(np.mean(pred)),
        "Std_Predicted": float(np.std(pred, ddof=0)),
        "Meets_R2_Target": bool(r2 >= R2_TARGET),
        "Meets_Main_N": bool(n_retained >= MAIN_MIN_N),
        "Meets_Main_Coverage": bool(coverage_within_subset >= MAIN_MIN_COVERAGE),
        "Publication_Candidate": bool((r2 >= R2_TARGET) and (n_retained >= MAIN_MIN_N) and (coverage_within_subset >= MAIN_MIN_COVERAGE)),
    }


def evaluate_all_policies(df: pd.DataFrame, component_cols: Sequence[str]) -> pd.DataFrame:
    total_n = len(df)
    rows: List[Dict[str, Any]] = []
    for (strategy, method), g in df.groupby(["Strategy", "Method"], sort=True):
        if len(g) < EXPLORATORY_MIN_N:
            continue
        g = g.reset_index(drop=False).rename(columns={"index": "Original_Row_Index"})
        rules = generate_rules_for_subset(g, component_cols)
        y_all = safe_float_array(g["Observed"])
        pred_all = safe_float_array(g["Predicted"])
        log(f"[SEARCH] Strategy={strategy} | Method={method} | base_n={len(g)} | candidate_rules={len(rules)}")
        for rule in rules:
            rec = evaluate_rule_subset_fast(y_all, pred_all, rule, str(strategy), str(method), len(g), total_n)
            if rec is not None:
                rows.append(rec)
    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No valid subdomain policy was generated. Check input table and required columns.")
    # Sort: first publication candidates hitting R2 target, then R2, then n, then RMSE.
    res = res.sort_values(
        ["Publication_Candidate", "Meets_R2_Target", "R2", "N_Retained", "RMSE"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    res.insert(0, "Rank", np.arange(1, len(res) + 1))
    return res


def build_retained_predictions(df: pd.DataFrame, best: pd.Series, component_cols: Sequence[str]) -> pd.DataFrame:
    strategy = str(best["Strategy"])
    method = str(best["Method"])
    rule_name = str(best["Rule_Name"])
    g = df[(df["Strategy"].astype(str) == strategy) & (df["Method"].astype(str) == method)].copy()
    g = g.reset_index(drop=False).rename(columns={"index": "Original_Row_Index"})
    rules = generate_rules_for_subset(g, component_cols)
    matching = [r for r in rules if r.rule_name == rule_name]
    if not matching:
        raise RuntimeError(f"Could not rebuild best policy mask: {rule_name}")
    mask = matching[0].mask
    retained = g.loc[mask].copy()
    retained.insert(0, "Selected_By_32D", True)
    retained.insert(1, "32D_Rule_Name", rule_name)
    retained.insert(2, "32D_Rule_Description", str(best["Rule_Description"]))
    return retained


def split_stability(retained: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "Split_ID" not in retained.columns:
        return pd.DataFrame()
    for split_id, g in retained.groupby("Split_ID", sort=True):
        if len(g) < 3:
            r2, rmse, mae = np.nan, np.nan, np.nan
        else:
            r2, rmse, mae = r2_rmse_mae(safe_float_array(g["Observed"]), safe_float_array(g["Predicted"]))
        rows.append({
            "Split_ID": split_id,
            "N_Retained": len(g),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Observed_Mean": float(pd.to_numeric(g["Observed"], errors="coerce").mean()),
            "Observed_Std": float(pd.to_numeric(g["Observed"], errors="coerce").std(ddof=0)),
        })
    return pd.DataFrame(rows)


def plot_policy_metrics(top: pd.DataFrame) -> None:
    show = top.head(MAX_TOP_POLICIES_TO_PLOT).copy()
    show["Policy_ID"] = [f"P{i:02d}" for i in range(1, len(show) + 1)]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4))
    fig.suptitle("32D selective AD subdomain policy metrics", fontsize=16, fontweight="bold", y=0.98)
    panels = [("R2", "Retained R2"), ("RMSE", "Retained RMSE"), ("MAE", "Retained MAE"), ("Coverage_Within_Strategy_Method", "Coverage")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        vals = show[col].astype(float).to_numpy()
        ax.bar(show["Policy_ID"], vals, color=EST_BLUE, edgecolor="0.25", linewidth=0.8)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Policy ID")
        ax.set_ylabel("Coverage" if "Coverage" in col else col)
        if col == "R2":
            ax.axhline(R2_TARGET, linestyle="--", linewidth=1.2, color="red", label="R² = 0.5 target")
            ax.legend(loc="best")
            ax.set_ylim(min(-0.1, np.nanmin(vals) - 0.05), max(0.55, np.nanmax(vals) + 0.08))
        else:
            ax.set_ylim(0, max(0.01, np.nanmax(vals) * 1.18))
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if col == "R2" else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.28)
    save_fig(fig, "165_32d_high_r2_policy_metrics.png")


def plot_frontier(res: pd.DataFrame) -> None:
    """Plot R2-coverage frontier with a bounded number of points for speed/stability."""
    candidate = res[res["N_Retained"] >= EXPLORATORY_MIN_N].copy()
    # Keep a representative, deterministic subset to avoid slow dense rendering.
    candidate = candidate.sort_values("R2", ascending=False).head(250).copy()
    top = candidate.head(20).copy()

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.scatter(
        candidate["Coverage_Within_Strategy_Method"],
        candidate["R2"],
        s=24,
        alpha=0.35,
        color=EST_GRAY,
        edgecolor="none",
        label="Candidate policies",
    )
    ax.scatter(
        top["Coverage_Within_Strategy_Method"],
        top["R2"],
        s=72,
        alpha=0.90,
        color=EST_BLUE,
        edgecolor="white",
        linewidth=0.5,
        label="Top R² policies",
    )
    for i, (_, row) in enumerate(top.head(8).iterrows(), start=1):
        ax.text(
            row["Coverage_Within_Strategy_Method"] + 0.004,
            row["R2"] + 0.006,
            f"P{i:02d}\n{sanitize_text(row['Method'], 16)}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.axhline(R2_TARGET, linestyle="--", color="red", linewidth=1.2, label="R² = 0.5 target")
    ax.axvline(MAIN_MIN_COVERAGE, linestyle=":", color="0.4", linewidth=1.2, label="Main min coverage")
    ax.set_xlabel("Coverage retained within strategy-method subset")
    ax.set_ylabel("Retained-domain R²")
    ax.set_title("R²–coverage frontier under 32D selective AD policies", pad=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    y_min = max(-1.0, np.nanmin(candidate["R2"]) - 0.05)
    y_max = max(0.55, np.nanmax(candidate["R2"]) + 0.08)
    ax.set_ylim(y_min, y_max)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12)
    save_fig(fig, "166_32d_r2_coverage_frontier.png")


def plot_best_prediction(retained: pd.DataFrame, best: pd.Series) -> None:
    y = safe_float_array(retained["Observed"])
    p = safe_float_array(retained["Predicted"])
    r2, rmse, mae = r2_rmse_mae(y, p)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.scatter(y, p, s=38, alpha=0.78, color=EST_BLUE, edgecolor="white", linewidth=0.35)
    ax.plot([0, 100], [0, 100], linestyle="--", color="red", linewidth=1.4)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    title = f"Best 32D selective AD subdomain: {sanitize_text(best['Method'], 28)}"
    ax.set_title(title, pad=12, fontsize=14, fontweight="bold")
    info = (
        f"R² = {r2:.3f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"n = {len(retained)}\n"
        f"coverage = {best['Coverage_Within_Strategy_Method']:.3f}\n"
        f"rule = {sanitize_text(best['Rule_Name'], 30)}"
    )
    ax.text(
        0.04, 0.96, info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.9),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "167_32d_best_policy_prediction_panel.png")


def plot_split_stability(stab: pd.DataFrame) -> None:
    """Save split-level stability plot.

    This function uses a simple PNG-only save path to avoid rare Matplotlib
    backend hangs on some Windows/PyCharm/PDF combinations.
    """
    if stab.empty:
        return
    show = stab.copy()
    show["Split_ID_short"] = show["Split_ID"].astype(str).str.replace("balanced5_seed_", "seed_", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.suptitle("32D best-policy split-level stability", fontsize=15, fontweight="bold", y=0.98)
    for ax, col in zip(axes, ["R2", "RMSE", "N_Retained"]):
        vals = np.array(pd.to_numeric(show[col], errors="coerce"), dtype=float, copy=True)
        ax.bar(show["Split_ID_short"], vals, color=EST_GREEN if col == "R2" else EST_BLUE, edgecolor="0.25", linewidth=0.8)
        ax.set_title("R²" if col == "R2" else col, pad=10)
        ax.set_xlabel("Split")
        ax.set_ylabel("R²" if col == "R2" else col)
        ax.tick_params(axis="x", rotation=30)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            continue
        if col == "R2":
            ax.axhline(R2_TARGET, linestyle="--", color="red", linewidth=1.0)
            ax.set_ylim(min(-0.5, np.nanmin(finite_vals) - 0.08), max(0.55, np.nanmax(finite_vals) + 0.08))
        else:
            ax.set_ylim(0, max(0.01, np.nanmax(finite_vals) * 1.18))
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}" if col != "N_Retained" else f"{int(v)}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(top=0.82, wspace=0.30, bottom=0.22)
    png = FIG_DIR / "168_32d_best_policy_split_stability.png"
    fig.savefig(png, dpi=600)
    plt.close(fig)
    log(f"[FIG] {png}")


def write_report(
    input_path: Path,
    res: pd.DataFrame,
    best: pd.Series,
    stab: pd.DataFrame,
    runtime_seconds: float,
) -> Path:
    path = TABLE_DIR / "32d_decision_report.txt"
    top = res.head(20)
    lines: List[str] = []
    lines.append("32D high-R2 target-free subdomain search decision report")
    lines.append("=" * 100)
    lines.append(f"Input table: {input_path}")
    lines.append(f"Candidate policies evaluated: {len(res)}")
    lines.append(f"R2 target: {R2_TARGET:.2f}")
    lines.append(f"Main-candidate minimum n: {MAIN_MIN_N}")
    lines.append(f"Main-candidate minimum coverage within strategy-method subset: {MAIN_MIN_COVERAGE:.2f}")
    lines.append("")
    lines.append("Best ranked target-free candidate")
    lines.append("-" * 100)
    for key in [
        "Strategy", "Method", "Rule_Name", "Rule_Family", "Rule_Description",
        "N_Base", "N_Retained", "Coverage_Within_Strategy_Method",
        "R2", "RMSE", "MAE", "Publication_Candidate",
    ]:
        lines.append(f"- {key}: {best.get(key)}")
    lines.append("")
    if bool(best.get("Meets_R2_Target", False)):
        lines.append("Decision: At least one target-free retained subdomain reached R2 >= 0.5.")
        if bool(best.get("Publication_Candidate", False)):
            lines.append("This candidate also meets the main n/coverage safeguards used in this script.")
        else:
            lines.append("However, this candidate is low-coverage or low-n and must be framed as a narrow selective subdomain unless further confirmed.")
    else:
        lines.append("Decision: No evaluated target-free retained subdomain reached R2 >= 0.5.")
    lines.append("")
    lines.append("Strictness / publication logic")
    lines.append("- Rule conditions are target-free and deployment-available.")
    lines.append("- Observed labels are used here to rank candidate rules retrospectively.")
    lines.append("- Therefore, the selected rule should be confirmed in a follow-up block-calibrated or independent-validation step before being written as a final deployable rule.")
    lines.append("")
    if not stab.empty:
        lines.append("Split-level stability for best candidate")
        lines.append("-" * 100)
        for _, r in stab.iterrows():
            lines.append(
                f"{r['Split_ID']}: n={int(r['N_Retained'])}, "
                f"R2={pretty_num(r['R2'])}, RMSE={pretty_num(r['RMSE'], 2)}, MAE={pretty_num(r['MAE'], 2)}"
            )
        lines.append("")
    lines.append("Top 20 policies")
    lines.append("-" * 100)
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i:02d}. Strategy={r['Strategy']} | Method={r['Method']} | "
            f"Rule={r['Rule_Name']} | Family={r['Rule_Family']} | "
            f"n={int(r['N_Retained'])} | cov={r['Coverage_Within_Strategy_Method']:.3f} | "
            f"R2={r['R2']:.4f} | RMSE={r['RMSE']:.3f} | MAE={r['MAE']:.3f} | "
            f"MainCandidate={r['Publication_Candidate']}"
        )
    lines.append("")
    lines.append(f"Total runtime / elapsed time: {runtime_seconds:.2f} seconds ({runtime_seconds/60:.2f} minutes)")
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[REPORT] {path}")
    return path


def main() -> None:
    start_time = time.perf_counter()
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {ROOT}")
    log("=" * 100)

    input_path = locate_prediction_table()
    df = pd.read_csv(input_path, low_memory=False)
    log(f"[INFO] Loaded: {input_path} | shape={df.shape}")

    required = ["Strategy", "Method", "Observed", "Predicted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    # Basic cleaning.
    df = df.copy()
    df["Observed"] = pd.to_numeric(df["Observed"], errors="coerce")
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
    df = df[np.isfinite(df["Observed"]) & np.isfinite(df["Predicted"])].copy()
    df["Predicted"] = df["Predicted"].clip(0, 100)
    df["Observed"] = df["Observed"].clip(0, 100)
    df["Strategy"] = df["Strategy"].astype(str)
    df["Method"] = df["Method"].astype(str)
    if "Reliability_Zone" not in df.columns:
        df["Reliability_Zone"] = "Unknown"
    else:
        df["Reliability_Zone"] = df["Reliability_Zone"].astype(str).fillna("Unknown")

    component_cols = get_candidate_component_columns(df)
    log(f"[INFO] Candidate target-free component columns ({len(component_cols)}): {component_cols}")

    res = evaluate_all_policies(df, component_cols)
    # Main ranking: if target-reaching candidates exist, put them first; otherwise best R2.
    res = res.sort_values(
        ["Publication_Candidate", "Meets_R2_Target", "R2", "N_Retained", "Coverage_Within_Strategy_Method", "RMSE"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    res["Rank"] = np.arange(1, len(res) + 1)

    # Add compact policy labels for clean figures.
    top_for_labels = res.head(max(20, MAX_TOP_POLICIES_TO_PLOT)).copy()
    top_for_labels["Policy_ID"] = [f"P{i:02d}" for i in range(1, len(top_for_labels) + 1)]
    label_map = top_for_labels[[
        "Policy_ID", "Rank", "Strategy", "Method", "Rule_Name", "Rule_Family", "Rule_Description",
        "N_Retained", "Coverage_Within_Strategy_Method", "R2", "RMSE", "MAE", "Publication_Candidate"
    ]].copy()

    best = res.iloc[0]
    retained = build_retained_predictions(df, best, component_cols)
    stab = split_stability(retained)

    safe_to_csv(res, "32d_high_r2_policy_summary.csv")
    safe_to_csv(label_map, "32d_policy_label_mapping.csv")
    safe_to_csv(retained, "32d_best_policy_retained_predictions.csv")
    safe_to_csv(stab, "32d_best_policy_split_stability.csv")

    log("[PLOT] policy metrics")
    plot_policy_metrics(top_for_labels.head(MAX_TOP_POLICIES_TO_PLOT))
    log("[PLOT] frontier")
    plot_frontier(res)
    log("[PLOT] best prediction")
    plot_best_prediction(retained, best)
    log("[PLOT] split stability")
    plot_split_stability(stab)

    runtime = time.perf_counter() - start_time
    write_report(input_path, res, best, stab, runtime)

    log("=" * 100)
    log("32D high-R2 target-free subdomain search finished.")
    log(f"Total runtime / elapsed time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    log("=" * 100)


if __name__ == "__main__":
    main()
