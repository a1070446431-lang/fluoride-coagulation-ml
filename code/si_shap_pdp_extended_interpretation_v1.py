# -*- coding: utf-8 -*-
"""Supplementary SHAP/PDP extension and formal Figure 8 consistency audit.

Place this complete file in ``fluoride_coagulation_ml_project/src/`` and run it.
The script reads only the final Figure 8 v8 exported tables and the final
source-bearing matrix. It does not train a model or alter any existing result.

Outputs are written to the existing
``outputs/jece_final_package/{figures,tables,reports,logs}/`` directories.
Dependencies: Python >= 3.9, numpy, pandas, matplotlib.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


STEM = "SI_shap_pdp_extended_interpretation_v1"
FORMAL_STEM = "Figure_8_shap_and_pdp_integrated_interpretation_v8"
DPI = 600
EXPECTED_N = 1011
EXPECTED_SOURCES = 27
EXPECTED_PAPER010_N = 184
EXPECTED_RETAINED_N = 44

FEATURES = [
    "氟形态", "游离氟占比", "初始氟_mg_L", "干扰离子",
    "干扰离子浓度_mmol_L", "水温_C", "混凝剂", "剂量_mM",
    "进水pH", "停留时间_min",
]
LABELS = {
    "氟形态": "Fluoride form",
    "游离氟占比": "Free-fluoride fraction",
    "初始氟_mg_L": "Initial fluoride",
    "干扰离子": "Interfering ion",
    "干扰离子浓度_mmol_L": "Interferent concentration",
    "水温_C": "Water temperature",
    "混凝剂": "Coagulant type",
    "剂量_mM": "Coagulant dosage",
    "进水pH": "Influent pH",
    "停留时间_min": "Residence time",
}
CONTEXTS = ["All domain", "Paper 010", "Retained domain"]
VALUE_COLS = [
    "mean_abs_shap_all", "mean_abs_shap_paper010", "mean_abs_shap_retained"
]
GROUPS = {
    "Water matrix": ["氟形态", "游离氟占比", "初始氟_mg_L", "干扰离子", "干扰离子浓度_mmol_L", "水温_C"],
    "Coagulant system": ["混凝剂", "剂量_mM"],
    "Operating conditions": ["进水pH", "停留时间_min"],
}
GROUP_COLORS = {
    "Water matrix": "#8FC9D5",
    "Coagulant system": "#EFD48C",
    "Operating conditions": "#C9D0D7",
}
PAIR_LABELS = {
    ("剂量_mM", "初始氟_mg_L"): ("Coagulant dosage (mM)", "Initial fluoride (mg L$^{-1}$)"),
    ("剂量_mM", "进水pH"): ("Coagulant dosage (mM)", "Influent pH"),
}

# Locked checkpoints copied from the accepted formal v8 export. They are used
# only for audit; calculations and displayed results always come from the CSV.
EXPECTED_IMPORTANCE = {
    "混凝剂": (6.295743, 6.879003, 9.174687),
    "剂量_mM": (5.237490, 9.469503, 9.734752),
    "干扰离子": (5.027713, 10.694785, 12.183828),
    "初始氟_mg_L": (4.062988, 1.662114, 1.139056),
    "进水pH": (4.042900, 2.060245, 1.881522),
    "游离氟占比": (2.375070, 1.913607, 2.227040),
    "氟形态": (2.341197, 1.222635, 1.487017),
    "停留时间_min": (1.813812, 0.694350, 0.928875),
    "干扰离子浓度_mmol_L": (0.509205, 0.384651, 0.339488),
    "水温_C": (0.297887, 0.142299, 0.174345),
}
EXPECTED_PDP = {
    "n": 612,
    "sum": 40668.27331523386,
    "sum_sq": 2711962.217525782,
    "min": 52.33361600830664,
    "max": 75.69723698582558,
}


def locate_root() -> Path:
    here = Path(__file__).resolve()
    checked = []
    for candidate in [here.parent, *here.parents]:
        checked.append(str(candidate))
        if (candidate / "data" / "processed").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError("Project root not found. Checked: " + "; ".join(checked))


def require_file(path: Path, alternatives: list[Path] | None = None) -> Path:
    if path.is_file():
        return path
    candidates = [path, *(alternatives or [])]
    for candidate in candidates[1:]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Required input is missing. Candidate paths:\n" + "\n".join(map(str, candidates)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")


def audit_inputs(data: pd.DataFrame, importance: pd.DataFrame,
                 pdp: pd.DataFrame, retained: pd.DataFrame) -> list[str]:
    checks: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    check(data.shape[0] == EXPECTED_N, f"data rows = {EXPECTED_N}")
    check("文献编号" in data.columns, "source column present")
    check(data["文献编号"].nunique(dropna=True) == EXPECTED_SOURCES,
          f"source groups = {EXPECTED_SOURCES}")
    check(data["文献编号"].astype(str).eq("Paper_010").sum() == EXPECTED_PAPER010_N,
          f"Paper_010 rows = {EXPECTED_PAPER010_N}")
    check(set(FEATURES).issubset(data.columns), "all ten predictive features present")
    forbidden = {"文献编号", "出水pH", "氟去除率_pct", "出水氟_mg_L", "由浓度反算去除率_pct", "去除率差值_pct", "原始行号"}
    check(not (set(FEATURES) & forbidden), "no target, post-treatment or source field in explanatory features")

    check(set(importance.columns) == {"feature", *VALUE_COLS}, "formal SHAP table schema")
    check(len(importance) == 10 and set(importance["feature"]) == set(FEATURES), "formal SHAP table has ten unique features")
    indexed = importance.set_index("feature")
    for feature, expected in EXPECTED_IMPORTANCE.items():
        actual = indexed.loc[feature, VALUE_COLS].to_numpy(dtype=float)
        check(np.allclose(actual, expected, rtol=0, atol=5e-6), f"locked SHAP checkpoint: {feature}")

    check(set(pdp.columns) == {"feature_x", "feature_y", "x", "y", "predicted_removal"}, "formal PDP table schema")
    check(len(pdp) == EXPECTED_PDP["n"], "formal PDP rows = 612")
    z = pd.to_numeric(pdp["predicted_removal"], errors="coerce").to_numpy(dtype=float)
    check(np.isfinite(z).all(), "all PDP predictions finite")
    check(np.isclose(z.sum(), EXPECTED_PDP["sum"], atol=1e-7), "locked PDP sum checkpoint")
    check(np.isclose(np.square(z).sum(), EXPECTED_PDP["sum_sq"], atol=1e-6), "locked PDP squared-sum checkpoint")
    check(np.isclose(z.min(), EXPECTED_PDP["min"], atol=1e-9), "locked PDP minimum checkpoint")
    check(np.isclose(z.max(), EXPECTED_PDP["max"], atol=1e-9), "locked PDP maximum checkpoint")
    pairs = set(zip(pdp["feature_x"], pdp["feature_y"]))
    check(pairs == set(PAIR_LABELS), "exact two formal PDP pairs")
    for pair, subset in pdp.groupby(["feature_x", "feature_y"], sort=False):
        check(subset["x"].nunique() == 18 and subset["y"].nunique() == 17 and len(subset) == 306,
              f"complete 18 x 17 grid: {pair}")

    retained_index = pd.to_numeric(retained.get("Data_Index"), errors="coerce").dropna().astype(int)
    check(retained_index.nunique() == EXPECTED_RETAINED_N, f"retained unique rows = {EXPECTED_RETAINED_N}")
    check(retained_index.between(0, EXPECTED_N - 1).all(), "retained indices inside final matrix")
    return checks


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(ra, rb)[0, 1])


def make_importance_tables(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = importance.set_index("feature").loc[FEATURES].copy()
    out = pd.DataFrame({"feature": FEATURES, "feature_label": [LABELS[f] for f in FEATURES]})
    for context, col in zip(CONTEXTS, VALUE_COLS):
        values = work[col].to_numpy(dtype=float)
        out[f"mean_abs_shap_{context.lower().replace(' ', '_')}"] = values
        out[f"rank_{context.lower().replace(' ', '_')}"] = pd.Series(-values).rank(method="min").astype(int).to_numpy()
        out[f"share_pct_{context.lower().replace(' ', '_')}"] = 100 * values / values.sum()
    out["rank_range"] = out[[c for c in out if c.startswith("rank_")]].max(axis=1) - out[[c for c in out if c.startswith("rank_")]].min(axis=1)
    out = out.sort_values("rank_all_domain").reset_index(drop=True)

    group_rows = []
    for context, col in zip(CONTEXTS, VALUE_COLS):
        total = float(work[col].sum())
        for group, features in GROUPS.items():
            value = float(work.loc[features, col].sum())
            group_rows.append({"context": context, "feature_group": group,
                               "sum_mean_abs_shap": value, "contribution_pct": 100 * value / total})
    group_table = pd.DataFrame(group_rows)

    corr_rows = []
    for i, (context_i, col_i) in enumerate(zip(CONTEXTS, VALUE_COLS)):
        for context_j, col_j in list(zip(CONTEXTS, VALUE_COLS))[i + 1:]:
            corr_rows.append({"context_1": context_i, "context_2": context_j,
                              "spearman_rank_correlation": rank_correlation(work[col_i], work[col_j])})
    return out, group_table, pd.DataFrame(corr_rows)


def grid_edges(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.unique(values.astype(float)))
    mids = (values[:-1] + values[1:]) / 2
    return np.r_[-np.inf, mids, np.inf]


def make_pdp_tables(pdp: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, profile_rows, support_rows = [], [], []
    for (fx, fy), subset in pdp.groupby(["feature_x", "feature_y"], sort=False):
        gx = np.sort(subset["x"].unique().astype(float))
        gy = np.sort(subset["y"].unique().astype(float))
        pivot = subset.pivot(index="y", columns="x", values="predicted_removal").loc[gy, gx]
        z = pivot.to_numpy(dtype=float)
        minimum = np.unravel_index(np.argmin(z), z.shape)
        maximum = np.unravel_index(np.argmax(z), z.shape)
        summary_rows.append({
            "feature_x": fx, "feature_y": fy, "grid_n_x": len(gx), "grid_n_y": len(gy),
            "x_min": gx.min(), "x_max": gx.max(), "y_min": gy.min(), "y_max": gy.max(),
            "prediction_min": z.min(), "prediction_max": z.max(), "prediction_range": np.ptp(z),
            "minimum_x": gx[minimum[1]], "minimum_y": gy[minimum[0]],
            "maximum_x": gx[maximum[1]], "maximum_y": gy[maximum[0]],
        })
        for j, x in enumerate(gx):
            profile_rows.append({"feature_x": fx, "feature_y": fy, "profile_axis": fx,
                                 "grid_value": x, "pdp_mean": z[:, j].mean(),
                                 "pdp_min": z[:, j].min(), "pdp_max": z[:, j].max()})
        for i, y in enumerate(gy):
            profile_rows.append({"feature_x": fx, "feature_y": fy, "profile_axis": fy,
                                 "grid_value": y, "pdp_mean": z[i, :].mean(),
                                 "pdp_min": z[i, :].min(), "pdp_max": z[i, :].max()})

        xv = pd.to_numeric(data[fx], errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(data[fy], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(xv) & np.isfinite(yv)
        counts, _, _ = np.histogram2d(yv[valid], xv[valid], bins=[grid_edges(gy), grid_edges(gx)])
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                support_rows.append({"feature_x": fx, "feature_y": fy, "x": x, "y": y,
                                     "predicted_removal": z[i, j], "paired_observation_count": int(counts[i, j]),
                                     "support_class": "observed" if counts[i, j] > 0 else "empty"})
    return pd.DataFrame(summary_rows), pd.DataFrame(profile_rows), pd.DataFrame(support_rows)


def figure_importance(metrics: pd.DataFrame, groups: pd.DataFrame, out_dir: Path) -> None:
    ordered = metrics.sort_values("rank_all_domain")
    values = ordered[["mean_abs_shap_all_domain", "mean_abs_shap_paper_010", "mean_abs_shap_retained_domain"]].to_numpy()
    ranks = ordered[["rank_all_domain", "rank_paper_010", "rank_retained_domain"]].to_numpy()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 5.3), gridspec_kw={"width_ratios": [1.28, 1.0]})
    im = ax_a.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=np.ceil(values.max()))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.58 * values.max() else "#20262D"
            ax_a.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7.4, color=color,
                      fontweight="bold" if ranks[i, j] <= 3 else "normal")
    ax_a.set_xticks(range(3), CONTEXTS)
    ax_a.set_yticks(range(len(ordered)), ordered["feature_label"])
    ax_a.set_title("Context-specific mean absolute SHAP values")
    ax_a.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax_a.set_yticks(np.arange(-.5, len(ordered), 1), minor=True)
    ax_a.grid(which="minor", color="white", linewidth=1.0)
    ax_a.tick_params(which="minor", bottom=False, left=False)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.035, pad=0.025)
    cb.set_label("Mean |SHAP value|")
    panel_label(ax_a, "a")

    y = np.arange(3)
    left = np.zeros(3)
    for group in ["Water matrix", "Coagulant system", "Operating conditions"]:
        vals = groups.loc[groups["feature_group"].eq(group), "contribution_pct"].to_numpy()
        ax_b.barh(y, vals, left=left, height=0.55, color=GROUP_COLORS[group],
                  edgecolor="#66717A", linewidth=0.55, label=group)
        for yi, start, value in zip(y, left, vals):
            if value >= 8:
                ax_b.text(start + value / 2, yi, f"{value:.1f}%", ha="center", va="center", fontsize=7.5)
        left += vals
    ax_b.set_yticks(y, CONTEXTS)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Contribution based on mean |SHAP| (%)")
    ax_b.set_title("Feature-group contribution by interpretation context")
    ax_b.grid(axis="x", color="#DDE2E7", linestyle="--", linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.205, right=0.965, top=0.91, bottom=0.20, wspace=0.42)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{STEM}_context_importance.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


def figure_pdp_support(pdp: pd.DataFrame, support: pd.DataFrame, data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    z_min, z_max = pdp["predicted_removal"].min(), pdp["predicted_removal"].max()
    pred_mesh = None
    count_mesh = None
    pairs = list(PAIR_LABELS)
    for col, (fx, fy) in enumerate(pairs):
        subset = pdp.loc[pdp["feature_x"].eq(fx) & pdp["feature_y"].eq(fy)]
        gx = np.sort(subset["x"].unique())
        gy = np.sort(subset["y"].unique())
        z = subset.pivot(index="y", columns="x", values="predicted_removal").loc[gy, gx].to_numpy()
        ax = axes[0, col]
        pred_mesh = ax.pcolormesh(gx, gy, z, shading="auto", cmap="YlGnBu", vmin=z_min, vmax=z_max)
        contour = ax.contour(gx, gy, z, levels=6, colors="white", linewidths=0.65)
        ax.clabel(contour, fmt="%.0f", fontsize=6.5)
        xv = pd.to_numeric(data[fx], errors="coerce")
        yv = pd.to_numeric(data[fy], errors="coerce")
        mask = xv.between(gx.min(), gx.max()) & yv.between(gy.min(), gy.max())
        points = np.flatnonzero(mask.to_numpy())
        if len(points) > 350:
            points = np.random.default_rng(42).choice(points, 350, replace=False)
        ax.scatter(xv.iloc[points], yv.iloc[points], s=5, facecolors="none", edgecolors="#29343D",
                   linewidths=0.35, alpha=0.45)
        ax.set_title("Formal two-dimensional PDP")
        ax.set_xlabel(PAIR_LABELS[(fx, fy)][0])
        ax.set_ylabel(PAIR_LABELS[(fx, fy)][1])
        span_x = float(gx.max() - gx.min())
        ax.set_xlim(gx.min() - 0.015 * span_x, gx.max() + 0.035 * span_x)
        ax.set_xticks([0, 60, 120, 180, 240, 300])
        ax.get_xticklabels()[-1].set_ha("right")
        panel_label(ax, "a" if col == 0 else "b")

        ss = support.loc[support["feature_x"].eq(fx) & support["feature_y"].eq(fy)]
        counts = ss.pivot(index="y", columns="x", values="paired_observation_count").loc[gy, gx].to_numpy()
        ax = axes[1, col]
        count_mesh = ax.pcolormesh(gx, gy, counts + 1, shading="auto", cmap="magma_r",
                                   norm=LogNorm(vmin=1, vmax=max(2, counts.max() + 1)))
        empty_pct = 100 * np.mean(counts == 0)
        ax.set_title(f"Empirical paired support (empty grid cells: {empty_pct:.1f}%)")
        ax.set_xlabel(PAIR_LABELS[(fx, fy)][0])
        ax.set_ylabel(PAIR_LABELS[(fx, fy)][1])
        ax.set_xlim(gx.min() - 0.015 * span_x, gx.max() + 0.035 * span_x)
        ax.set_xticks([0, 60, 120, 180, 240, 300])
        ax.get_xticklabels()[-1].set_ha("right")
        panel_label(ax, "c" if col == 0 else "d")
    cbar1 = fig.colorbar(pred_mesh, ax=axes[0, :].tolist(), fraction=0.022, pad=0.025)
    cbar1.set_label("Predicted fluoride removal (%)")
    cbar2 = fig.colorbar(count_mesh, ax=axes[1, :].tolist(), fraction=0.022, pad=0.025)
    count_labels = [0, 1, 2, 5, 10, 20, 50, 100]
    count_labels = [value for value in count_labels if value + 1 <= count_mesh.norm.vmax]
    cbar2.set_ticks([value + 1 for value in count_labels])
    cbar2.set_ticklabels([str(value) for value in count_labels])
    cbar2.set_label("Observations assigned to grid cell")
    fig.subplots_adjust(left=0.09, right=0.90, top=0.94, bottom=0.09, wspace=0.30, hspace=0.38)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{STEM}_pdp_support.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


def figure_profiles(profiles: pd.DataFrame, data: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    pairs = list(PAIR_LABELS)
    for col, (fx, fy) in enumerate(pairs):
        for row, axis_feature in enumerate([fx, fy]):
            ax = axes[row, col]
            subset = profiles.loc[
                profiles["feature_x"].eq(fx) & profiles["feature_y"].eq(fy) &
                profiles["profile_axis"].eq(axis_feature)
            ].sort_values("grid_value")
            x = subset["grid_value"].to_numpy(dtype=float)
            mean = subset["pdp_mean"].to_numpy(dtype=float)
            low = subset["pdp_min"].to_numpy(dtype=float)
            high = subset["pdp_max"].to_numpy(dtype=float)
            ax.fill_between(x, low, high, color="#A8D5E2", alpha=0.55,
                            label="Range across the paired feature")
            ax.plot(x, mean, color="#145C8E", lw=2.0, marker="o", ms=3.0,
                    label="Marginal mean of formal 2D PDP")
            raw = pd.to_numeric(data[axis_feature], errors="coerce").dropna()
            raw = raw.loc[raw.between(x.min(), x.max())]
            ymin, ymax = ax.get_ylim()
            rug_y = ymin + 0.015 * (ymax - ymin)
            if len(raw) > 250:
                raw = raw.sample(250, random_state=42)
            ax.plot(raw, np.full(len(raw), rug_y), "|", color="#4E5963", ms=4, alpha=0.35)
            axis_label = LABELS[axis_feature]
            if axis_feature == "剂量_mM":
                axis_label += " (mM)"
            elif axis_feature == "初始氟_mg_L":
                axis_label += " (mg L$^{-1}$)"
            ax.set_xlabel(axis_label)
            ax.set_ylabel("Predicted fluoride removal (%)")
            surface_name = "initial-fluoride" if fy == "初始氟_mg_L" else "influent-pH"
            ax.set_title(f"Marginal profile within the {surface_name} surface")
            ax.grid(color="#DDE2E7", linestyle="--", linewidth=0.6)
            ax.set_axisbelow(True)
            panel_label(ax, chr(ord("a") + row * 2 + col))
    axes[0, 0].legend(loc="best", frameon=False, fontsize=7.3)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.94, bottom=0.09, wspace=0.28, hspace=0.38)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{STEM}_pdp_marginal_profiles.{ext}", dpi=DPI if ext == "png" else None)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.{decimals}f}")
    headers = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(path: Path, metrics: pd.DataFrame, groups: pd.DataFrame,
                 correlations: pd.DataFrame, summary: pd.DataFrame,
                 support: pd.DataFrame, checks: list[str]) -> None:
    top = metrics.sort_values("rank_all_domain").iloc[0]
    p010_top = metrics.sort_values("rank_paper_010").iloc[0]
    retained_top = metrics.sort_values("rank_retained_domain").iloc[0]
    empty = (support.groupby(["feature_x", "feature_y"])["paired_observation_count"]
             .apply(lambda s: 100 * (s == 0).mean()).reset_index(name="empty_grid_pct"))
    lines = [
        "# Supplementary SHAP and PDP extension (draft)", "",
        "## Scope and safeguards", "",
        "This analysis extends the accepted Figure 8 v8 exports without retraining the model or changing the main figure. "
        "SHAP values describe the fitted CatBoost model, whereas PDP surfaces are model-averaged response summaries. "
        "Neither quantity establishes causality or an independently validated chemical mechanism.", "",
        "The source identifier, response, effluent measurements, reconstructed response and row identifiers are excluded "
        "from the ten explanatory features. Paper 010 and the retained domain are interpretation contexts only.", "",
        "## SHAP context comparison", "",
        f"Across all 1,011 observations, {top.feature_label} ranked first (mean |SHAP| = {top.mean_abs_shap_all_domain:.3f}). "
        f"The leading feature changed to {p010_top.feature_label} within Paper 010 (n = 184; {p010_top.mean_abs_shap_paper_010:.3f}) "
        f"and remained {retained_top.feature_label} in the retained 44-row source-specific domain ({retained_top.mean_abs_shap_retained_domain:.3f}). "
        "This rank reordering indicates context dependence and must not be interpreted as universal feature dominance.", "",
        "Feature-group percentages are normalized within each context and therefore compare contribution composition, not prediction accuracy.", "",
        markdown_table(groups, decimals=3), "",
        "Pairwise rank agreement across contexts:", "",
        markdown_table(correlations, decimals=3), "",
        "## PDP extension and support boundary", "",
        "The formal two-dimensional PDP grids span the marginal 10th-90th percentile ranges used in Figure 8. "
        "The paired-support maps count observations assigned to the same grid cells. Empty or sparsely populated cells flag "
        "combinations for which the PDP is mainly model interpolation over substituted feature values; such regions require cautious interpretation.", "",
        markdown_table(summary, decimals=4), "",
        markdown_table(empty, decimals=2), "",
        "The marginal profiles are averages of the already exported two-dimensional surfaces, not newly fitted one-dimensional PDPs. "
        "Their shaded envelopes show variation across the paired feature and should not be read as confidence intervals.", "",
        "## Reproducibility audit", "",
        f"All {len(checks)} structural and numerical checks passed. The locked Figure 8 SHAP values, PDP row count, grid dimensions, "
        "aggregate numerical signatures and extrema were reproduced from the formal v8 exports.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    configure_style()
    root = locate_root()
    package = root / "outputs" / "jece_final_package"
    figure_dir, table_dir = package / "figures", package / "tables"
    report_dir, log_dir = package / "reports", package / "logs"
    for directory in [figure_dir, table_dir, report_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data_path = require_file(root / "data" / "processed" / "model_ready_v1_with_source.csv")
    importance_path = require_file(table_dir / f"{FORMAL_STEM}_feature_importance.csv")
    pdp_path = require_file(table_dir / f"{FORMAL_STEM}_pdp_surfaces.csv")
    retained_path = require_file(
        root / "outputs" / "tables" / "32g_unique_retained_chemical_rows.csv",
        [table_dir / "32g_unique_retained_chemical_rows.csv"]
    )

    data = pd.read_csv(data_path)
    importance = pd.read_csv(importance_path)
    pdp = pd.read_csv(pdp_path)
    retained = pd.read_csv(retained_path)
    checks = audit_inputs(data, importance, pdp, retained)
    metrics, groups, correlations = make_importance_tables(importance)
    summary, profiles, support = make_pdp_tables(pdp, data)

    outputs = {
        f"{STEM}_feature_context_metrics.csv": metrics,
        f"{STEM}_feature_group_contributions.csv": groups,
        f"{STEM}_context_rank_correlations.csv": correlations,
        f"{STEM}_pdp_surface_summary.csv": summary,
        f"{STEM}_pdp_marginal_profiles.csv": profiles,
        f"{STEM}_pdp_paired_support_grid.csv": support,
    }
    for filename, frame in outputs.items():
        frame.to_csv(table_dir / filename, index=False, encoding="utf-8-sig")

    figure_importance(metrics, groups, figure_dir)
    figure_pdp_support(pdp, support, data, figure_dir)
    figure_profiles(profiles, data, figure_dir)
    write_report(report_dir / f"{STEM}_supplementary_text_draft.md", metrics, groups,
                 correlations, summary, support, checks)

    audit = {
        "script": STEM,
        "status": "PASS",
        "numeric_and_structural_checks": len(checks),
        "check_messages": checks,
        "data_rows": len(data),
        "source_groups": int(data["文献编号"].nunique()),
        "paper010_rows": int(data["文献编号"].astype(str).eq("Paper_010").sum()),
        "retained_unique_rows": int(pd.to_numeric(retained["Data_Index"], errors="coerce").nunique()),
        "formal_importance_sha256": sha256(importance_path),
        "formal_pdp_sha256": sha256(pdp_path),
        "input_hashes": {str(p.relative_to(root)): sha256(p) for p in [data_path, importance_path, pdp_path, retained_path]},
        "model_trained": False,
        "main_figure_modified": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime_s": round(time.perf_counter() - started, 3),
    }
    audit_path = report_dir / f"{STEM}_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines = [f"{key}: {value}" for key, value in audit.items() if key not in {"check_messages", "input_hashes"}]
    (log_dir / f"{STEM}.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    runtime = time.perf_counter() - started
    print(f"PASS: {len(checks)} locked, structural and leakage-prevention checks.")
    print(f"Contexts: all={len(data)}, Paper_010={EXPECTED_PAPER010_N}, retained={EXPECTED_RETAINED_N}; sources={EXPECTED_SOURCES}.")
    print("Created 6 tables, 3 PNG/PDF figure pairs, 1 text draft and audit records.")
    print(f"Outputs: {package}")
    print("Formal Figure 8 files and model results unchanged; no model was trained.")
    print(f"Total runtime: {runtime:.2f} s")


if __name__ == "__main__":
    main()
