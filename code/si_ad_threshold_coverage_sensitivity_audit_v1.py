# -*- coding: utf-8 -*-
"""论文1：P08 选择性适用域阈值、覆盖率、迁移性与化学窗口补充审计。

将本文件放入 fluoride_coagulation_ml_project/src/ 后直接运行。
脚本只读取既有 Step 32B/32D/32F2/32G/32H 结果，不训练模型，不改写正文图。
输出沿用 outputs/jece_final_package/{tables,figures,reports,logs}/。
依赖：Python >= 3.9, numpy, pandas, matplotlib。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

STEM = "SI_ad_threshold_coverage_sensitivity_audit_v1"
STRATEGY = "RepeatedBalanced_Extended"
METHOD = "XGBoost"
ZONE = "Outside"
SCORE = "NN_Risk"
THRESHOLD = 0.814815
MIN_N = 30
MIN_COVERAGE = 0.005
R2_TARGET = 0.5
RANDOM_STATE = 20260608

INPUT_NAMES = [
    "32b_predictions_with_reliability_zone.csv",
    "32d_policy_label_mapping.csv",
    "32f2_policy_reproduction_audit.csv",
    "32f2_best_reproduced_retained_predictions.csv",
    "32f2_best_policy_split_stability.csv",
    "32f2_best_policy_literature_group_dominance.csv",
    "32g_retained_predictions_with_chemistry.csv",
    "32g_unique_retained_chemical_rows.csv",
    "32g_numeric_operating_window.csv",
    "32g_categorical_composition.csv",
    "32g_ad_component_summary.csv",
    "32g_representativeness_summary.csv",
    "32h_threshold_sensitivity_all.csv",
    "32h_exact_policy_metrics.csv",
    "32h_exact_policy_split_metrics.csv",
    "32h_exact_policy_paper_dominance.csv",
    "32h_threshold_plateau_summary.csv",
    "32h_strategy_method_transfer_summary.csv",
    "32h_bootstrap_uncertainty_exact_policy.csv",
    "32h_bootstrap_uncertainty_exact_policy_raw.csv",
]

METHOD_ORDER = [
    "KRR_Poly", "CatBoost", "XGBoost", "CB-XGB",
    "KRR_CatBoost_Weighted_2Model", "KRR_XGBoost_Weighted_2Model",
    "GPR_Matern", "PLS",
]
METHOD_LABEL = {
    "KRR_Poly": "KRR-Poly", "CatBoost": "CatBoost", "XGBoost": "XGBoost",
    "CB-XGB": "CB-XGB", "KRR_CatBoost_Weighted_2Model": "KRR + CatBoost",
    "KRR_XGBoost_Weighted_2Model": "KRR + XGBoost",
    "GPR_Matern": "GPR-Matern", "PLS": "PLS",
}
STRATEGY_ORDER = [
    "Balanced_5Seed_Current31F", "RepeatedBalanced_Extended",
    "RepeatedOrdinary_GroupedSplit", "GroupKFold_FullyGrouped",
    "LOLO_LeaveOneLiteratureOut",
]
STRATEGY_LABEL = {
    "Balanced_5Seed_Current31F": "Balanced (5)",
    "RepeatedBalanced_Extended": "Balanced (15)",
    "RepeatedOrdinary_GroupedSplit": "Ordinary (15)",
    "GroupKFold_FullyGrouped": "GroupKFold (5)",
    "LOLO_LeaveOneLiteratureOut": "LOLO (27)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(frame["Observed"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame["Predicted"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    e = p - y
    den = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(e ** 2) / den if len(y) > 1 and den > 0 else np.nan
    return {"R2": float(r2), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAE": float(np.mean(np.abs(e))), "N": int(len(y))}


def resolve_inputs(project_root: Path, table_dir: Path, formal_tables: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in INPUT_NAMES:
        candidates = [table_dir / name, formal_tables / name]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            raise FileNotFoundError("Required existing result is missing. Candidate paths:\n" +
                                    "\n".join(str(p) for p in candidates))
        hashes = {sha256(p) for p in existing}
        if len(hashes) > 1:
            raise ValueError(f"Conflicting copies of {name}: {existing}")
        found[name] = existing[0]
    return found


def threshold_grid(zone_scores: np.ndarray) -> list[float]:
    coarse = np.round(np.arange(0.50, 1.0001, 0.025), 6)
    local = np.round(np.arange(THRESHOLD - 0.080, THRESHOLD + 0.0801, 0.005), 6)
    quant = np.round(np.quantile(zone_scores, np.linspace(0.02, 0.98, 25)), 6)
    return sorted(set(float(x) for x in np.concatenate([coarse, local, quant, [THRESHOLD]])
                      if np.isfinite(x) and 0 <= x <= 1.5))


def evaluate(frame_all: pd.DataFrame, frame_zone: pd.DataFrame, threshold: float,
             strategy: str, method: str) -> dict[str, object]:
    kept = frame_zone[pd.to_numeric(frame_zone[SCORE], errors="coerce") <= threshold].copy()
    m = metrics(kept)
    n = m.pop("N")
    unique_n = int(kept["Data_Index"].nunique()) if n else 0
    counts = kept["文献编号"].astype(str).value_counts() if n else pd.Series(dtype=float)
    dominant = str(counts.index[0]) if len(counts) else ""
    dominant_n = int(counts.iloc[0]) if len(counts) else 0
    return {
        "Strategy": strategy, "Method": method, "Zone": ZONE,
        "Score_Column": SCORE, "Threshold": float(threshold), **m,
        "N_Retained": n, "Unique_Row_N": unique_n,
        "Coverage_within_strategy_method": n / len(frame_all),
        "Unique_Row_Coverage_within_strategy_method": unique_n / len(frame_all),
        "N_Literature_Groups": int(len(counts)), "Dominant_Group": dominant,
        "Dominant_Group_N": dominant_n,
        "Dominant_Group_Fraction": dominant_n / n if n else np.nan,
        "Pass_R2_0p5": bool(np.isfinite(m["R2"]) and m["R2"] >= R2_TARGET),
        "Pass_Min_N": bool(n >= MIN_N),
        "Pass_Min_Coverage": bool(n / len(frame_all) >= MIN_COVERAGE),
        "Is_Exact_Base_Threshold": bool(np.isclose(threshold, THRESHOLD, atol=1e-9)),
    }


def compare_numeric(checks: list[dict[str, object]], label: str, actual: object,
                    expected: object, atol: float = 1e-8) -> None:
    a = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([expected]), errors="coerce").iloc[0]
    passed = bool((pd.isna(a) and pd.isna(b)) or np.isclose(float(a), float(b), rtol=1e-7,
                                                             atol=atol, equal_nan=True))
    checks.append({"Check": label, "Recomputed": a, "Recorded": b, "Passed": passed})
    if not passed:
        raise ValueError(f"Numeric audit failed: {label}; {a} != {b}")


def compare_frame(checks: list[dict[str, object]], actual: pd.DataFrame,
                  expected: pd.DataFrame, keys: list[str], numbers: list[str], label: str) -> None:
    a = actual.copy().sort_values(keys).reset_index(drop=True)
    b = expected.copy().sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        raise ValueError(f"{label} row count differs: {len(a)} != {len(b)}")
    for i in range(len(a)):
        for key in keys:
            if str(a.loc[i, key]) != str(b.loc[i, key]):
                raise ValueError(f"{label} key mismatch at row {i}: {key}")
        for col in numbers:
            compare_numeric(checks, f"{label}[{i}] {col}", a.loc[i, col], b.loc[i, col])


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.3, "ytick.labelsize": 8.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str, products: list[Path]) -> None:
    for ext in ("png", "pdf"):
        path = figure_dir / f"{STEM}_{stem}.{ext}"
        fig.savefig(path, dpi=500, facecolor="white")
        products.append(path)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "src" / "config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}. Place this complete file in project src/.")
    spec = importlib.util.spec_from_file_location("project_config_si_ad", config_path)
    config = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(config)
    if Path(config.PROJECT_ROOT).resolve() != project_root:
        raise ValueError("config.PROJECT_ROOT does not match the project containing this script.")

    package = project_root / "outputs" / "jece_final_package"
    dirs = {name: package / name for name in ("tables", "figures", "reports", "logs")}
    inputs = resolve_inputs(project_root, Path(config.TABLE_DIR), dirs["tables"])
    input_hashes = {str(p.relative_to(project_root)): sha256(p) for p in inputs.values()}
    data = {name: pd.read_csv(path, low_memory=False) for name, path in inputs.items()}
    full = data["32b_predictions_with_reliability_zone.csv"]
    if len(full) != 66224 or full["Data_Index"].nunique() != 1011 or full["文献编号"].nunique() != 27:
        raise ValueError("32B prediction dimensions differ from the audited 1011-row, 27-source run.")
    if not np.allclose(pd.to_numeric(full.Residual), pd.to_numeric(full.Observed) - pd.to_numeric(full.Predicted)):
        raise ValueError("Residual convention is inconsistent with Observed - Predicted.")

    policy = data["32d_policy_label_mapping.csv"]
    p08 = policy[policy.Policy_ID.astype(str) == "P08"]
    if len(p08) != 1:
        raise ValueError("Exactly one P08 row is required in 32d_policy_label_mapping.csv.")
    p08 = p08.iloc[0]
    expected_rule = "Reliability_Zone == Outside AND NN_Risk <= 0.814815"
    if (str(p08.Strategy) != STRATEGY or str(p08.Method) != METHOD or
            str(p08.Rule_Description) != expected_rule):
        raise ValueError("P08 definition differs from the locked audit target.")

    subset_all = full[(full.Strategy == STRATEGY) & (full.Method == METHOD)].copy()
    subset_zone = subset_all[subset_all.Reliability_Zone.astype(str).str.lower() == ZONE.lower()].copy()
    if len(subset_all) != 3546 or len(subset_zone) != 1775:
        raise ValueError("P08 denominator or Outside-zone row count differs from the source run.")

    threshold_rows = [evaluate(subset_all, subset_zone, t, STRATEGY, METHOD)
                      for t in threshold_grid(pd.to_numeric(subset_zone[SCORE]).dropna().to_numpy(float))]
    threshold = pd.DataFrame(threshold_rows).sort_values("Threshold").reset_index(drop=True)
    exact = threshold[threshold.Is_Exact_Base_Threshold].copy()
    if len(exact) != 1:
        raise ValueError("The exact P08 threshold is missing or duplicated.")
    retained = subset_zone[pd.to_numeric(subset_zone[SCORE]) <= THRESHOLD].copy()
    if len(retained) != 52 or retained.Data_Index.nunique() != 44:
        raise ValueError("Exact P08 retained event/unique-row counts differ from 52/44.")

    split_rows = []
    for split_id, split_all in subset_all.groupby("Split_ID", sort=False):
        kept = retained[retained.Split_ID == split_id]
        if kept.empty:
            continue
        row = evaluate(split_all, kept.assign(Reliability_Zone=ZONE), np.inf, STRATEGY, METHOD)
        row.update({"Split_ID": split_id,
                    "Coverage_within_split": len(kept) / len(split_all)})
        split_rows.append(row)
    split = pd.DataFrame(split_rows)

    transfer_rows = []
    for (strategy, method), all_sm in full.groupby(["Strategy", "Method"], sort=False):
        zone_sm = all_sm[all_sm.Reliability_Zone.astype(str).str.lower() == ZONE.lower()]
        if len(zone_sm):
            transfer_rows.append(evaluate(all_sm, zone_sm, THRESHOLD, str(strategy), str(method)))
    transfer = pd.DataFrame(transfer_rows)

    dominance_rows = []
    dominance_thresholds = sorted(set(threshold.Threshold.tolist()))
    for t in dominance_thresholds:
        kept = subset_zone[pd.to_numeric(subset_zone[SCORE]) <= t]
        for rank, (group, n) in enumerate(kept["文献编号"].astype(str).value_counts().head(10).items(), 1):
            dominance_rows.append({"Threshold": t, "Rank": rank, "Literature_Group": group,
                                   "N": int(n), "Fraction_within_retained": n / len(kept),
                                   "Coverage_within_strategy_method": len(kept) / len(subset_all),
                                   "Total_Retained_N": len(kept)})
    dominance = pd.DataFrame(dominance_rows)

    rng = np.random.default_rng(RANDOM_STATE)
    boot_rows = []
    for i in range(1000):
        sample = retained.iloc[rng.integers(0, len(retained), size=len(retained))]
        boot_rows.append({"Bootstrap_ID": i + 1, **metrics(sample)})
    bootstrap_raw = pd.DataFrame(boot_rows)
    bootstrap_summary = []
    for col in ("R2", "RMSE", "MAE"):
        values = bootstrap_raw[col].dropna().to_numpy(float)
        bootstrap_summary.append({"Metric": col, "Mean": values.mean(),
                                  "Median": np.median(values), "CI_2p5": np.quantile(values, .025),
                                  "CI_97p5": np.quantile(values, .975), "N_Bootstrap": len(values)})
    bootstrap_summary = pd.DataFrame(bootstrap_summary)

    checks: list[dict[str, object]] = []
    compare_frame(checks, threshold, data["32h_threshold_sensitivity_all.csv"],
                  ["Threshold"], ["R2", "RMSE", "MAE", "N_Retained", "Unique_Row_N",
                                  "Coverage_within_strategy_method", "Dominant_Group_N",
                                  "Dominant_Group_Fraction"], "threshold sensitivity")
    compare_frame(checks, transfer, data["32h_strategy_method_transfer_summary.csv"],
                  ["Strategy", "Method"], ["R2", "RMSE", "MAE", "N_Retained", "Unique_Row_N",
                                             "Coverage_within_strategy_method", "Dominant_Group_N",
                                             "Dominant_Group_Fraction"], "strategy-method transfer")
    # Lower-ranked sources can tie in count; pandas versions may order tied labels differently.
    # Compare the threshold/rank counts and fractions, and audit the dominant source separately.
    compare_frame(checks, dominance, data["32h_exact_policy_paper_dominance.csv"],
                  ["Threshold", "Rank"],
                  ["N", "Fraction_within_retained", "Coverage_within_strategy_method",
                   "Total_Retained_N"], "paper dominance")
    expected_top = data["32h_exact_policy_paper_dominance.csv"].query("Rank == 1")
    actual_top = dominance.query("Rank == 1")
    if list(actual_top.sort_values("Threshold").Literature_Group) != list(expected_top.sort_values("Threshold").Literature_Group):
        raise ValueError("Dominant literature group differs at one or more thresholds.")
    compare_frame(checks, bootstrap_raw, data["32h_bootstrap_uncertainty_exact_policy_raw.csv"],
                  ["Bootstrap_ID"], ["R2", "RMSE", "MAE", "N"], "event bootstrap")
    compare_frame(checks, bootstrap_summary, data["32h_bootstrap_uncertainty_exact_policy.csv"],
                  ["Metric"], ["Mean", "Median", "CI_2p5", "CI_97p5", "N_Bootstrap"],
                  "bootstrap summary")
    recorded_retained = data["32f2_best_reproduced_retained_predictions.csv"]
    compare_frame(checks, retained, recorded_retained,
                  ["Strategy", "Split_ID", "Method", "Data_Index"],
                  ["Observed", "Predicted", "NN_Risk", "Corrected_AD_Risk"],
                  "retained predictions")
    for _, row in exact.iterrows():
        for col in ("R2", "RMSE", "MAE", "N_Retained", "Unique_Row_N",
                    "Coverage_within_strategy_method", "Dominant_Group_N",
                    "Dominant_Group_Fraction"):
            compare_numeric(checks, f"exact P08 {col}", row[col],
                            data["32h_exact_policy_metrics.csv"].iloc[0][col])

    chemistry = data["32g_unique_retained_chemical_rows.csv"]
    if len(chemistry) != 44 or chemistry._Data_Index_Int.nunique() != 44:
        raise ValueError("32G unique retained chemistry table differs from 44 unique rows.")
    numeric_window = data["32g_numeric_operating_window.csv"]
    for _, row in numeric_window.iterrows():
        source = chemistry if str(row.Context).startswith("Retained") else None
        if source is not None and row.Variable_Column in source:
            vals = pd.to_numeric(source[row.Variable_Column], errors="coerce").dropna()
            for col, value in (("N", len(vals)), ("Mean", vals.mean()), ("Std", vals.std(ddof=1)),
                               ("Min", vals.min()), ("Median", vals.median()), ("Max", vals.max())):
                compare_numeric(checks, f"chemistry {row.Variable_Key} {col}", value, row[col])

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    outputs = {
        "exact_policy_metrics": exact,
        "retained_prediction_events": retained,
        "retained_unique_chemistry_rows": chemistry,
        "threshold_sensitivity": threshold,
        "threshold_paper_dominance": dominance,
        "split_stability": split,
        "strategy_method_transfer": transfer,
        "numeric_operating_window": numeric_window,
        "categorical_composition": data["32g_categorical_composition.csv"],
        "ad_component_summary": data["32g_ad_component_summary.csv"],
        "representativeness": data["32g_representativeness_summary.csv"],
        "event_level_bootstrap_raw": bootstrap_raw,
        "event_level_bootstrap_summary": bootstrap_summary,
        "numeric_consistency_checks": pd.DataFrame(checks),
    }
    for suffix, frame in outputs.items():
        path = dirs["tables"] / f"{STEM}_{suffix}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        products.append(path)

    style()
    color = "#177E89"
    orange = "#D97706"
    red = "#B33A3A"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(left=.09, right=.96, bottom=.10, top=.92, wspace=.27, hspace=.35)
    ax = axes[0, 0]
    ax.plot(threshold.Threshold, threshold.R2, color=color, lw=1.8, marker="o", ms=2.5)
    ax.axhline(.5, color=red, ls="--", lw=1, label="R² criterion")
    ax.axvline(THRESHOLD, color="black", ls=":", lw=1.1, label="P08 threshold")
    ax.axvspan(.804815, .814815, color="#F2C14E", alpha=.22, label="qualifying plateau")
    ax.set(xlabel="NN_Risk threshold", ylabel="Retained-subset R²", title="a  Threshold sensitivity")
    ax.legend(frameon=False, fontsize=7.8)
    ax = axes[0, 1]
    ax.plot(threshold.Threshold, threshold.N_Retained, color=orange, lw=1.8, label="Prediction events")
    ax.plot(threshold.Threshold, threshold.Unique_Row_N, color=color, lw=1.5, label="Unique rows")
    ax.axhline(MIN_N, color=red, ls="--", lw=1, label="minimum N")
    ax.axvline(THRESHOLD, color="black", ls=":", lw=1.1)
    ax.set(xlabel="NN_Risk threshold", ylabel="Retained count", title="b  Coverage-count trade-off")
    ax2 = ax.twinx(); ax2.plot(threshold.Threshold,
                              threshold.Coverage_within_strategy_method * 100,
                              color="#7B2CBF", lw=1.2, alpha=.8)
    ax2.set_ylabel("Event coverage (%)", color="#7B2CBF")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax = axes[1, 0]
    ax.plot(threshold.Threshold, threshold.Dominant_Group_Fraction * 100, color=red, lw=1.8)
    ax.axvline(THRESHOLD, color="black", ls=":", lw=1.1)
    ax.set(xlabel="NN_Risk threshold", ylabel="Dominant-source fraction (%)",
           ylim=(0, 105), title="c  Source concentration")
    ax.text(.03, .08, "P08: Paper_010 = 100%", transform=ax.transAxes, fontsize=8.5)
    ax = axes[1, 1]
    ax.scatter(retained.Observed, retained.Predicted, s=24, alpha=.65, c=color,
               edgecolor="white", linewidth=.25)
    low = min(float(retained[["Observed", "Predicted"]].min().min()) - 3, 0)
    high = max(float(retained[["Observed", "Predicted"]].max().max()) + 3, 100)
    ax.plot([low, high], [low, high], "k--", lw=1)
    m = metrics(retained)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Observed removal (%)",
           ylabel="Predicted removal (%)", title="d  Exact P08 retained predictions")
    ax.text(.04, .96, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}\nEvents/unique = 52/44",
            transform=ax.transAxes, va="top", fontsize=8.2)
    fig.suptitle("Selective AD threshold evidence and coverage boundary", fontsize=14, fontweight="bold")
    save_figure(fig, dirs["figures"], "threshold_evidence", products)

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 5.4))
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.87, wspace=.38)
    exact_split = split.sort_values("Split_ID")
    bars = axes[0].bar([s.replace("balanced_ext_", "") for s in exact_split.Split_ID],
                       exact_split.R2, color=[color if v >= .5 else "#A7B0B5" for v in exact_split.R2])
    axes[0].axhline(.5, color=red, ls="--", lw=1)
    axes[0].set(ylabel="R²", title="a  Split-level stability", ylim=(0, .85))
    for bar, r2, n in zip(bars, exact_split.R2, exact_split.N_Retained):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                     f"{r2:.3f}\nN={int(n)}", ha="center", fontsize=8)
    axes[1].hist(bootstrap_raw.R2.dropna(), bins=28, density=True, alpha=.62, color=color)
    axes[1].axvline(float(bootstrap_summary.loc[bootstrap_summary.Metric == "R2", "Median"].iloc[0]), color="black", ls=":", lw=1)
    axes[1].set(title="b  Bootstrap R²", xlabel="R²", ylabel="Density")
    axes[2].hist(bootstrap_raw.RMSE.dropna(), bins=28, density=True, alpha=.48, color=orange, label="RMSE")
    axes[2].hist(bootstrap_raw.MAE.dropna(), bins=28, density=True, alpha=.42, color="#7B2CBF", label="MAE")
    axes[2].set(title="c  Bootstrap errors", xlabel="Removal percentage points", ylabel="Density")
    axes[2].legend(frameon=False, fontsize=8)
    coverage_values = [44/1011*100, 44/184*100, 52/3546*100]
    labels = ["Unique / all data", "Unique / Paper_010", "Events / strategy-method"]
    b = axes[3].barh(labels, coverage_values, color=["#A7B0B5", color, orange])
    axes[3].invert_yaxis()
    axes[3].set(xlabel="Coverage (%)", title="d  Representativeness", xlim=(0, 27))
    for bar, value in zip(b, coverage_values):
        axes[3].text(value+.45, bar.get_y()+bar.get_height()/2, f"{value:.2f}%", va="center", fontsize=8)
    fig.suptitle("Stability, uncertainty and representativeness of P08", fontsize=14, fontweight="bold")
    fig.text(.08, .035, "Bootstrap resamples 52 recorded prediction events; repeated rows and single-source dominance limit inferential interpretation.", fontsize=8.5)
    save_figure(fig, dirs["figures"], "stability_uncertainty_representativeness", products)

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 6.6))
    fig.subplots_adjust(left=.14, right=.96, bottom=.20, top=.88, wspace=.31)
    input_keys = ["initial_f", "dose", "pH_in", "retention_time", "temperature", "interfering_conc", "free_f_fraction"]
    w = numeric_window[(numeric_window.Context == "Retained unique experimental rows") &
                       numeric_window.Variable_Key.isin(input_keys)].copy()
    w = w.set_index("Variable_Key").reindex(input_keys).reset_index()
    labels = ["Initial F", "Dose", "Influent pH", "Retention time", "Temperature", "Interferent conc.", "Free-F fraction"]
    source_window = numeric_window[(numeric_window.Context == "Paper_010 all source rows") &
                                   numeric_window.Variable_Key.isin(input_keys)].set_index("Variable_Key").reindex(input_keys)
    y_pos = np.arange(len(w))
    for i, row in w.iterrows():
        source_row = source_window.loc[row.Variable_Key]
        span = float(source_row.Max - source_row.Min)
        if span > 0:
            lo = (float(row.Min) - float(source_row.Min)) / span
            hi = (float(row.Max) - float(source_row.Min)) / span
            md = (float(row.Median) - float(source_row.Min)) / span
        else:
            lo = hi = md = .5
        axes[0].plot([lo, hi], [i, i], color="#80AEB5", lw=3, solid_capstyle="round")
        axes[0].plot(md, i, "o", color=color, ms=6)
        axes[0].text(1.03, i, f"{row.Min:g}-{row.Median:g}-{row.Max:g}", va="center", fontsize=7.4)
    axes[0].set_yticks(y_pos, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(-.03, 1.38), xlabel="Position within the Paper_010 source range (0-1)",
                title="a  Retained input-condition ranges")
    axes[0].axvline(0, color="#A7B0B5", lw=.7); axes[0].axvline(1, color="#A7B0B5", lw=.7)
    cats = data["32g_categorical_composition.csv"]
    co = cats[(cats.Context == "Retained unique experimental rows") & (cats.Variable_Key == "coagulant")]
    axes[1].barh(range(len(co)), co.Fraction*100, color=color)
    short = co.Plot_Label.astype(str).tolist()
    axes[1].set_yticks(range(len(co)), short, fontsize=7.4)
    axes[1].invert_yaxis(); axes[1].set(xlabel="Fraction of 44 unique rows (%)", title="b  Coagulant composition")
    fig.suptitle("Chemistry-specific operating window retained by P08", fontsize=14, fontweight="bold")
    fig.text(.11, .045, "All 44 unique rows: Paper_010; initial F = 10 mg/L; retention time = 40 min; temperature = 25 °C; SO₄²⁻/CO₃²⁻ = 0.935 mmol/L.", fontsize=8.5)
    save_figure(fig, dirs["figures"], "chemistry_window", products)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    fig.subplots_adjust(left=.17, right=.97, bottom=.22, top=.88, wspace=.32)
    for ax, value_col, title, cmap, limits in [
        (axes[0], "R2", "a  R² after transferring the exact threshold", "RdBu", (-.5, .8)),
        (axes[1], "Coverage_within_strategy_method", "b  Retained event coverage", "YlGnBu", (0, None)),
    ]:
        mat = transfer.pivot(index="Method", columns="Strategy", values=value_col).reindex(index=METHOD_ORDER, columns=STRATEGY_ORDER)
        if value_col == "Coverage_within_strategy_method":
            mat = mat * 100.0
        arr = mat.to_numpy(float)
        kwargs = {"vmin": limits[0]}
        if limits[1] is not None: kwargs["vmax"] = limits[1]
        image = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, **kwargs)
        ax.set_xticks(range(len(STRATEGY_ORDER)), [STRATEGY_LABEL[s] for s in STRATEGY_ORDER], rotation=42, ha="right")
        ax.set_yticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m] for m in METHOD_ORDER] if ax is axes[0] else [""]*len(METHOD_ORDER))
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                text_value = "NA" if not np.isfinite(v) else (f"{v:.3f}" if value_col == "R2" else f"{v:.2f}%")
                ax.text(j, i, text_value, ha="center", va="center", fontsize=7.5,
                        color="white" if np.isfinite(v) and image.norm(v) > .72 else "black")
        fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
    fig.suptitle("Transfer audit of the P08 rule: Outside and NN_Risk ≤ 0.814815", fontsize=14, fontweight="bold")
    fig.text(.17, .035, "The transfer table reports every available Step 32B strategy-method combination. NA indicates that the combination was not evaluated.", fontsize=8.5)
    save_figure(fig, dirs["figures"], "strategy_method_transfer", products)

    table_pdf = dirs["reports"] / f"{STEM}_summary_tables.pdf"
    plateau_row = data["32h_threshold_plateau_summary.csv"].iloc[0]
    plateau_boot = pd.DataFrame([
        {"Section": "Threshold plateau", "Item": "Threshold span", "Estimate": f"{plateau_row.Threshold_Min:.6f} to {plateau_row.Threshold_Max:.6f}"},
        {"Section": "Threshold plateau", "Item": "Evaluated thresholds", "Estimate": int(plateau_row.Threshold_Count)},
        {"Section": "Threshold plateau", "Item": "R² range", "Estimate": f"{plateau_row.R2_Min:.4f} to {plateau_row.R2_Max:.4f}"},
        {"Section": "Threshold plateau", "Item": "Retained N range", "Estimate": f"{int(plateau_row.N_Min)} to {int(plateau_row.N_Max)}"},
        {"Section": "Threshold plateau", "Item": "Event coverage range", "Estimate": f"{100*plateau_row.Coverage_Min:.3f}% to {100*plateau_row.Coverage_Max:.3f}%"},
        {"Section": "Threshold plateau", "Item": "Dominant source", "Estimate": f"{plateau_row.Dominant_Group_Mode} (100%)"},
        *[{"Section": "Event bootstrap", "Item": str(row.Metric),
           "Estimate": f"median {row.Median:.4f}; 95% percentile interval {row.CI_2p5:.4f} to {row.CI_97p5:.4f}"}
          for row in bootstrap_summary.itertuples(index=False)],
    ])
    transfer_display = transfer.copy()
    transfer_display["Strategy"] = transfer_display.Strategy.map(STRATEGY_LABEL)
    transfer_display["Method"] = transfer_display.Method.map(METHOD_LABEL)
    transfer_display = transfer_display.rename(columns={
        "N_Retained": "Events", "Unique_Row_N": "Unique",
        "Coverage_within_strategy_method": "Coverage",
        "N_Literature_Groups": "Sources", "Dominant_Group_Fraction": "Dominant_fraction",
    }).sort_values(["Strategy", "R2"], ascending=[True, False])
    transfer_columns = ["Strategy", "Method", "R2", "RMSE", "MAE", "Events", "Unique", "Coverage", "Sources", "Dominant_fraction"]
    exact_display = pd.concat([exact.assign(Record="Pooled"), split.assign(Record=split.Split_ID)], ignore_index=True, sort=False).rename(columns={
        "N_Retained": "Events", "Unique_Row_N": "Unique", "Coverage_within_strategy_method": "Coverage",
        "Dominant_Group": "Dominant_source", "Dominant_Group_Fraction": "Dominant_fraction",
    })
    pages = [
        ("Exact P08 policy and split stability", exact_display,
         ["Record", "R2", "RMSE", "MAE", "Events", "Unique", "Coverage", "Dominant_source", "Dominant_fraction"]),
        ("Qualifying threshold plateau and event-level bootstrap", plateau_boot, None),
        ("Exact-threshold transfer across strategies and methods (1/2)", transfer_display.iloc[:17], transfer_columns),
        ("Exact-threshold transfer across strategies and methods (2/2)", transfer_display.iloc[17:], transfer_columns),
        ("Retained chemistry: numeric operating window", w, ["Variable_Key", "N", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"]),
        ("Retained chemistry: categorical composition", co, ["Variable_Key", "Plot_Label", "Count", "Fraction"]),
    ]
    with PdfPages(table_pdf) as pdf:
        for page_no, (title, frame, columns) in enumerate(pages, 1):
            view = frame.copy()
            if columns is not None:
                view = view[[c for c in columns if c in view.columns]]
            if len(view) > 20:
                view = view.head(20)
            display = []
            for row in view.itertuples(index=False, name=None):
                display.append([f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                                ("NA" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else str(v)) for v in row])
            fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis("off")
            ax.set_title(title, fontsize=15, pad=20)
            widths = [max(0.07, min(.24, max(len(str(c)), *(len(str(v)) for v in view[c].head(20))) / 75)) for c in view.columns]
            widths = [x/sum(widths) for x in widths]
            table = ax.table(cellText=display, colLabels=list(view.columns), cellLoc="center",
                             colWidths=widths, bbox=[0, .24, 1, .62])
            table.auto_set_font_size(False); table.set_fontsize(7.2 if len(view.columns) > 8 else 8.5)
            for (i, _), cell in table.get_celld().items():
                cell.set_edgecolor("#CCD3D7"); cell.set_linewidth(.45)
                if i == 0:
                    cell.set_facecolor("#244957"); cell.get_text().set_color("white")
                elif i % 2:
                    cell.set_facecolor("#EDF3F5")
            note = "Metrics are reconstructed from existing Step 32B predictions. RMSE/MAE are removal percentage points."
            if page_no == 2:
                note = "Bootstrap intervals resample 52 prediction events and are descriptive because rows repeat and all events come from Paper_010."
            if page_no in (5, 6):
                note = "The chemistry window is descriptive. Effluent pH, removal and effluent F are excluded from the displayed input-condition summary."
            fig.text(.08, .13, note, fontsize=9)
            fig.text(.91, .06, f"Page {page_no} of {len(pages)}", ha="right", fontsize=8, color="#58666B")
            pdf.savefig(fig); plt.close(fig)
    products.append(table_pdf)

    report = f"""# P08选择性适用域阈值、覆盖率与敏感性补充审计

## 审计对象
P08来自既有候选策略筛选：Strategy={STRATEGY}，Method={METHOD}，规则为`Reliability_Zone == Outside AND NN_Risk <= {THRESHOLD}`。规则执行只使用Reliability_Zone（可靠性区域）与NN_Risk（最近邻风险），不直接使用目标值或测试误差。阈值候选最初依据保留子集表现进行筛选，因此属于post hoc selection（事后候选选择）；本次邻近阈值、迁移与来源组成分析不能替代独立外部确认。早期32D映射表将P08的Publication_Candidate记录为False，后续32F2只确认其为可复现的选择性AD结果；这两段来源记录均应保留。

## 精确复现结果
从完整32B预测表重算得到R²={float(exact.iloc[0].R2):.6f}、RMSE={float(exact.iloc[0].RMSE):.6f}、MAE={float(exact.iloc[0].MAE):.6f}，保留52次预测事件和44个唯一实验行。事件覆盖率为{float(exact.iloc[0].Coverage_within_strategy_method):.6f}，唯一行相对于3546次策略-模型事件的比例为{float(exact.iloc[0].Unique_Row_Coverage_within_strategy_method):.6f}。44个唯一行占全部1011行的4.352%，占Paper_010的184行的23.913%。

## 稳定性与边界
保留事件仅来自2个划分，其中1/2个划分达到R²≥0.5：balanced_ext_seed_53为R²=0.742227、N=44；balanced_ext_seed_66为R²=0.340975、N=8。满足R²≥0.5、N≥30和事件覆盖率≥0.5%的连续记录阈值平台为0.804815–0.814815，共4个已评估阈值，N为34–52。平台内仍全部由Paper_010构成。

全部52次预测事件和44个唯一行均来自Paper_010。因此，R²≈0.748应表述为chemistry/source-specific retained subdomain（化学条件/来源特异的保留子域），不能表述为普适跨文献可靠域或全域分组验证性能。跨策略迁移中某些组合仍超过R²=0.5，但保留来源组成和误差水平明显变化，不能将这些结果合并成一个普适性能指标。

## 化学窗口
保留唯一行的初始氟为10 mg/L，停留时间为40 min，水温为25 °C，游离氟占比为1.0，干扰离子体系为SO₄²⁻/CO₃²⁻且浓度为0.935 mmol/L；进水pH为7–9，混凝剂剂量为0–43.74 mM。该窗口描述规则实际保留的数据条件。出水pH、氟去除率和出水氟仅作descriptive outcome（结果描述），未用于P08规则。

## Bootstrap（自助法）说明
既有1000次bootstrap（自助抽样）以52次预测事件为抽样单位。由于8次事件对应重复实验行且全部事件来自同一来源，其95%区间只反映该事件集合内部的抽样波动，不能作为跨文献泛化不确定性。应优先同时报告划分结果、唯一行数和来源集中度。

## English figure captions
Threshold evidence and coverage boundary. Sensitivity of retained-subset R², retained counts, event coverage and dominant-source fraction to the NN_Risk threshold, together with the exact P08 prediction panel. The highlighted interval contains the recorded thresholds satisfying R²≥0.5, N≥30 and event coverage≥0.5%.

Stability, uncertainty and representativeness. Split-level results, event-level bootstrap distributions and three coverage denominators for the exact P08 retained subset. Bootstrap intervals are descriptive because prediction events include repeated rows and originate from one source.

Chemistry-specific operating window. Ranges of prediction-time input conditions and coagulant composition among 44 unique retained rows. Outcome-derived variables are excluded from the plotted input-condition window.

Strategy-method transfer audit. Performance and retained event coverage obtained by applying the exact P08 rule threshold to every available Step 32B strategy-method combination. NA denotes an unevaluated combination.

XGBoost: extreme gradient boosting（极端梯度提升）；KRR: kernel ridge regression（核岭回归）；GPR: Gaussian process regression（高斯过程回归）；PLS: partial least squares（偏最小二乘）；AD: applicability domain（适用域）；LOLO: leave-one-literature-out（留一文献外验证）。最终Table S/Figure S编号在补充材料总编排时统一分配。
"""
    report_path = dirs["reports"] / f"{STEM}_supplementary_text.md"
    report_path.write_text(report, encoding="utf-8")
    products.append(report_path)

    if not all(sha256(project_root / rel) == digest for rel, digest in input_hashes.items()):
        raise RuntimeError("At least one original input changed during the audit.")
    audit = {
        "script": Path(__file__).name, "script_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes, "inputs_unchanged": True,
        "n_full_prediction_events": len(full), "n_unique_rows": int(full.Data_Index.nunique()),
        "n_sources": int(full["文献编号"].nunique()), "p08_prediction_events": len(retained),
        "p08_unique_rows": int(retained.Data_Index.nunique()),
        "n_numeric_checks": len(checks), "all_numeric_checks_passed": True,
        "output_sha256": {str(p.relative_to(project_root)): sha256(p) for p in products},
    }
    audit_path = dirs["logs"] / f"{STEM}_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PASS: {len(checks)} numeric consistency checks; complete 32B base = {len(full)} events.")
    print("Exact P08: R2=0.747676, RMSE=11.474674, MAE=9.155856, events/unique=52/44.")
    print("Threshold plateau: 0.804815-0.814815; 2 retained splits; 1/2 passes R2 >= 0.5.")
    print("Dominant source: Paper_010 (100% of retained events).")
    print("Outputs:", package)
    print("Original data, predictions, tables and main figures unchanged; no models trained.")
    print(f"Total runtime: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
