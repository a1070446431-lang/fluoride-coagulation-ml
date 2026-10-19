# -*- coding: utf-8 -*-
"""
Figure 6 — selective applicability-domain evidence and claim boundary.

Place this file in:
    fluoride_coagulation_ml_project/src/

Run directly in PyCharm. The script reads existing project tables and writes:
    outputs/jece_final_package/figures/
    outputs/jece_final_package/tables/
    outputs/jece_final_package/reports/
    outputs/jece_final_package/logs/
"""
from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

SCRIPT_NAME = "34c_est_figure6_selective_ad_evidence_v5"
FIGURE_STEM = "Figure_6_selective_ad_evidence_and_chemistry_window_v5"
DPI = 600
FIG_W, FIG_H = 12.9, 8.75

COL = {
    "blue": "#5F86A5",
    "blue_light": "#DCE8F0",
    "teal": "#4F8C91",
    "teal_light": "#D8E8E6",
    "gold": "#C99B55",
    "gold_light": "#F2E3C9",
    "sage": "#8FA38F",
    "sage_light": "#E2E9E1",
    "rose": "#B86F68",
    "rose_light": "#EED9D6",
    "gray": "#6E7780",
    "gray_mid": "#A8B0B7",
    "gray_light": "#F2F5F6",
    "grid": "#D8DEE3",
    "ink": "#252A2E",
    "white": "#FFFFFF",
}


@dataclass(frozen=True)
class Paths:
    root: Path
    fig_dir: Path
    table_dir: Path
    report_dir: Path
    log_dir: Path


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.4,
        "axes.titlesize": 10.8,
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def locate_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (p / "outputs" / "tables").exists() and (p / "src").exists():
            return p
        if p.name == "fluoride_coagulation_ml_project" and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Could not locate fluoride_coagulation_ml_project root.")


def build_paths() -> Paths:
    root = locate_project_root()
    out = root / "outputs" / "jece_final_package"
    paths = Paths(
        root=root,
        fig_dir=out / "figures",
        table_dir=out / "tables",
        report_dir=out / "reports",
        log_dir=out / "logs",
    )
    for d in (paths.fig_dir, paths.table_dir, paths.report_dir, paths.log_dir):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def read_table(paths: Paths, name: str) -> pd.DataFrame:
    p = paths.root / "outputs" / "tables" / name
    if not p.exists():
        raise FileNotFoundError(f"Required table not found: {p}")
    return pd.read_csv(p)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=COL["ink"], clip_on=False)


def polish_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=COL["grid"], linestyle="--", linewidth=0.65, alpha=0.80)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    mae = float(np.mean(np.abs(obs - pred)))
    return float(r2), rmse, mae


def stage_label(text: str) -> str:
    mapping = {
        "All-domain grouped validation": "All-domain\ngrouped",
        "Predefined reliable domain": "Predefined\nreliable domain",
        "Target-free selective retained subdomain": "Selective retained\nsubdomain",
    }
    return mapping.get(text, text)


def panel_a(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    add_panel_label(ax, "a", x=-0.13, y=1.12)
    e = evidence.copy()
    order = [
        "All-domain grouped validation",
        "Predefined reliable domain",
        "Target-free selective retained subdomain",
    ]
    e["Evidence_Level"] = pd.Categorical(e["Evidence_Level"], order, ordered=True)
    e = e.sort_values("Evidence_Level")

    x = np.arange(len(e))
    r2 = e["R2"].to_numpy(float)
    coverage = e["Coverage"].to_numpy(float)
    colors = [COL["blue"], COL["gray_mid"], COL["gold"]]
    sizes = 110 + 520 * np.sqrt(np.clip(coverage, 0, 1))

    ax.plot(x, r2, color=COL["gray_mid"], linewidth=1.25, zorder=1)
    ax.scatter(x, r2, s=sizes, c=colors, edgecolor="white", linewidth=1.1, zorder=3)
    ax.axhline(0.50, color=COL["sage"], linestyle="--", linewidth=0.95)
    ax.text(1.98, 0.515, "$R^2=0.50$", ha="left", va="center", fontsize=7.4, color=COL["sage"])

    for i, row in e.reset_index(drop=True).iterrows():
        ax.text(i, row["R2"] + 0.075, f"$R^2$={row['R2']:.3f}",
                ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=COL["ink"])
        ax.text(i, -0.265,
                f"RMSE {row['RMSE']:.1f}\ncoverage {row['Coverage']*100:.2f}%",
                ha="center", va="bottom", fontsize=7.0, color=COL["gray"], linespacing=1.20)

    ax.set_xticks(x, [stage_label(v) for v in e["Evidence_Level"].astype(str)])
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.36, 0.92)
    ax.set_xlim(-0.45, 2.42)
    ax.set_title("Evidence hierarchy and coverage trade-off", pad=11)
    polish_axis(ax, "y")



def panel_b(ax: plt.Axes, sensitivity: pd.DataFrame, plateau: pd.DataFrame, exact: pd.DataFrame) -> None:
    add_panel_label(ax, "b", x=-0.095, y=1.085)
    use = sensitivity[sensitivity["N_Retained"] > 0].copy().sort_values("Threshold")
    use = use[np.isfinite(use["R2"])].copy()

    p = plateau.iloc[0]
    exact_row = exact.iloc[0]
    thr = use["Threshold"].to_numpy(float)
    r2 = use["R2"].to_numpy(float)
    n = use["N_Retained"].to_numpy(float)

    ax.fill_betweenx([-0.05, 0.85], p["Threshold_Min"], p["Threshold_Max"],
                     color=COL["gold_light"], alpha=0.82, zorder=0)
    ax.plot(thr, r2, color=COL["teal"], linewidth=1.65, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.85, zorder=3)
    ax.axvline(exact_row["Threshold"], color=COL["gold"], linewidth=1.15, linestyle="--")
    ax.scatter([exact_row["Threshold"]], [exact_row["R2"]], s=58,
               color=COL["gold"], edgecolor="white", linewidth=0.9, zorder=5)

    ax2 = ax.twinx()
    ax2.plot(thr, n, color=COL["blue"], linewidth=1.10, alpha=0.84)
    ax2.fill_between(thr, 0, n, color=COL["blue_light"], alpha=0.30)
    ax2.set_ylabel("Retained prediction instances", color=COL["blue"], labelpad=8)
    ax2.tick_params(axis="y", colors=COL["blue"], labelsize=7.8, pad=3)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(n) * 1.10)

    ax.set_xlim(0.775, 1.005)
    ax.set_ylim(-0.05, 0.85)
    ax.set_xlabel("NN-risk threshold")
    ax.set_ylabel("Retained-domain $R^2$")
    ax.set_title("Threshold sensitivity and accepted plateau", pad=10)
    polish_axis(ax, "both")

    # Move plateau description to a data-sparse region and use a short leader line.
    note = (f"Accepted plateau: {p['Threshold_Min']:.3f}-{p['Threshold_Max']:.3f}\n"
            f"$R^2$ = {p['R2_Min']:.3f}-{p['R2_Max']:.3f}; N = {int(p['N_Min'])}-{int(p['N_Max'])}")
    ax.annotate(note,
                xy=((p["Threshold_Min"] + p["Threshold_Max"]) / 2, p["R2_Max"]),
                xytext=(0.865, 0.805), textcoords="data",
                ha="left", va="top", fontsize=7.4, color=COL["gray"],
                arrowprops=dict(arrowstyle="-", color=COL["gray_mid"], lw=0.7),
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor=COL["gray_mid"], linewidth=0.55, alpha=0.94))
    ax.text(exact_row["Threshold"] + 0.004, exact_row["R2"] - 0.045,
            "accepted threshold", ha="left", va="top", fontsize=7.2, color=COL["gold"])

    # Direct labels replace a floating legend and avoid overlap with the curves.
    ax.text(thr[-1] - 0.004, r2[-1] + 0.035, "$R^2$", ha="right", va="bottom",
            fontsize=7.4, color=COL["teal"], fontweight="bold")
    ax2.text(thr[-1] - 0.004, n[-1] - 35, "retained N", ha="right", va="top",
             fontsize=7.3, color=COL["blue"], fontweight="bold")


def panel_c(subspec, predictions: pd.DataFrame) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subspec,
        width_ratios=[4.25, 1.10], height_ratios=[1.05, 4.0],
        wspace=0.06, hspace=0.06,
    )
    ax_top = plt.subplot(inner[0, 0])
    ax = plt.subplot(inner[1, 0])
    ax_right = plt.subplot(inner[1, 1], sharey=ax)
    ax_blank = plt.subplot(inner[0, 1])
    ax_blank.axis("off")
    add_panel_label(ax_top, "c", x=-0.13, y=1.42)

    df = predictions.copy()
    obs = pd.to_numeric(df["Observed"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(df["Predicted"], errors="coerce").to_numpy(float)
    r2, rmse, mae = regression_metrics(obs, pred)

    split_order = list(df["Split_ID"].dropna().astype(str).unique())
    split_colors = [COL["teal"], COL["gold"], COL["blue"], COL["rose"]]
    markers = ["o", "s", "^", "D"]
    for i, split in enumerate(split_order):
        sub = df[df["Split_ID"].astype(str) == split]
        label = split.replace("balanced_ext_seed_", "Seed ")
        ax.scatter(sub["Observed"], sub["Predicted"], s=31,
                   color=split_colors[i % len(split_colors)], marker=markers[i % len(markers)],
                   alpha=0.76, edgecolor="white", linewidth=0.55, label=label)

    lo = float(np.nanmin([obs.min(), pred.min()]))
    hi = float(np.nanmax([obs.max(), pred.max()]))
    lo = min(-2.0, np.floor(lo / 5) * 5)
    hi = max(75.0, np.ceil(hi / 5) * 5)
    xx = np.linspace(lo, hi, 150)
    coef = np.polyfit(obs, pred, 1)
    ax.plot(xx, xx, linestyle="--", color=COL["gray"], linewidth=1.0)
    ax.plot(xx, coef[0] * xx + coef[1], color=COL["gold"], linewidth=1.25)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual fluoride removal efficiency (%)")
    ax.set_ylabel("Predicted value (%)")
    polish_axis(ax, "both")
    ax.legend(frameon=False, loc="lower right", fontsize=7.1)

    metric_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nN = {len(df)} (44 unique)"
    ax.text(0.04, 0.96, metric_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                    edgecolor=COL["gray_mid"], linewidth=0.7, alpha=0.95))

    bins = np.linspace(lo, hi, 13)
    ax_top.hist(obs, bins=bins, color="#C9D1D8", alpha=0.78, edgecolor="white")
    ax_top.hist(pred, bins=bins, color=COL["teal"], alpha=0.48, edgecolor="white")
    ax_top.set_xlim(lo, hi)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=7.0)
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.set_title("Retained-domain pooled predictions", pad=8)

    ax_right.hist(pred, bins=bins, orientation="horizontal", color=COL["teal"], alpha=0.48, edgecolor="white")
    ax_right.hist(obs, bins=bins, orientation="horizontal", color="#C9D1D8", alpha=0.78, edgecolor="white")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=7.0)
    ax_right.spines[["top", "right"]].set_visible(False)



def panel_d(subspec, window: pd.DataFrame) -> None:
    variables = [
        ("initial_f", "Initial F (mg L$^{-1}$)"),
        ("dose", "Dose (mM)"),
        ("pH_in", "Influent pH"),
        ("retention_time", "Residence time (min)"),
    ]
    inner = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=subspec,
                                             height_ratios=[0.95, 1, 1, 1, 1], hspace=0.78)
    ax_title = plt.subplot(inner[0, 0])
    ax_title.axis("off")
    add_panel_label(ax_title, "d", x=-0.105, y=1.02)
    ax_title.text(0.5, 0.82, "Retained chemistry and operating window",
                  ha="center", va="center", fontsize=10.8, color=COL["ink"])

    contexts = ["All source data", "Paper_010 all source rows", "Retained unique experimental rows"]
    context_colors = [COL["gray_mid"], COL["teal"], COL["gold"]]
    legend_handles = [
        Line2D([0], [0], color=context_colors[0], lw=5, label="All-data IQR"),
        Line2D([0], [0], color=context_colors[1], lw=5, label="Paper_010 IQR"),
        Line2D([0], [0], color=context_colors[2], lw=2, marker="o", label="Retained range / median"),
    ]
    ax_title.legend(handles=legend_handles, frameon=False, loc="lower center",
                    bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=6.6,
                    handlelength=1.9, columnspacing=1.0)

    for idx, (key, label) in enumerate(variables, start=1):
        ax = plt.subplot(inner[idx, 0])
        sub = window[window["Variable_Key"] == key].set_index("Context")
        all_row = sub.loc[contexts[0]]
        paper_row = sub.loc[contexts[1]]
        ret_row = sub.loc[contexts[2]]

        yvals = [2, 1, 0]
        ax.plot([all_row["Q25"], all_row["Q75"]], [2, 2], color=context_colors[0], lw=7, solid_capstyle="butt")
        ax.plot([paper_row["Q25"], paper_row["Q75"]], [1, 1], color=context_colors[1], lw=7, solid_capstyle="butt")
        ax.plot([ret_row["Min"], ret_row["Max"]], [0, 0], color=context_colors[2], lw=2.2)
        ax.scatter([ret_row["Median"]], [0], s=28, color=context_colors[2], edgecolor="white", linewidth=0.6, zorder=3)

        vals = [all_row["Q25"], all_row["Q75"], paper_row["Q25"], paper_row["Q75"], ret_row["Min"], ret_row["Max"]]
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        span = vmax - vmin
        if span == 0:
            span = max(abs(vmax) * 0.3, 1.0)
        ax.set_xlim(vmin - 0.10 * span, vmax + 0.15 * span)
        ax.set_ylim(-0.55, 2.55)
        ax.set_yticks(yvals, ["All", "Paper 010", "Retained"])
        ax.set_title(label, fontsize=8.4, loc="left", pad=3.0)
        ax.grid(axis="x", color=COL["grid"], linestyle="--", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=6.8)
        ax.tick_params(axis="x", labelsize=6.8)



def panel_e(ax: plt.Axes, split_metrics: pd.DataFrame, dominance: pd.DataFrame, represent: pd.DataFrame) -> None:
    add_panel_label(ax, "e", x=-0.10, y=1.055)
    s = split_metrics.copy().sort_values("R2", ascending=False)
    y = np.arange(len(s))[::-1]
    labels = s["Split_ID"].str.replace("balanced_ext_seed_", "Seed ", regex=False)
    pass_mask = s["R2_Target_Pass"].astype(bool).to_numpy()
    colors = [COL["teal"] if p else COL["rose"] for p in pass_mask]
    sizes = 70 + 5.0 * s["N_Retained"].to_numpy(float)

    ax.hlines(y, 0, s["R2"], color=COL["gray_mid"], linewidth=1.0)
    ax.scatter(s["R2"], y, s=sizes, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.axvline(0.50, color=COL["sage"], linestyle="--", linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-1.15, len(s) - 0.25)
    ax.set_xlabel("Split-level $R^2$")
    ax.set_title("Split stability and source support", pad=10)
    polish_axis(ax, "x")

    for yi, (_, row) in zip(y, s.iterrows()):
        ax.text(row["R2"] + 0.025, yi,
                f"N={int(row['N_Retained'])}; RMSE={row['RMSE']:.1f}",
                va="center", ha="left", fontsize=7.1, color=COL["gray"], clip_on=False)

    retained = represent.loc[represent["Context"] == "Retained unique rows"].iloc[0]
    support_text = ("Paper_010 = 100% of retained predictions\n"
                    f"44 unique rows = {retained['Fraction_of_All_Data']*100:.2f}% of all source data")
    ax.text(0.50, 0.135, support_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, color=COL["ink"], linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=COL["gold_light"],
                      edgecolor=COL["gold"], linewidth=0.8, alpha=0.96))


def create_figure(paths: Paths) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    evidence = read_table(paths, "32i_final_evidence_summary.csv")
    sensitivity = read_table(paths, "32h_threshold_sensitivity_all.csv")
    plateau = read_table(paths, "32h_threshold_plateau_summary.csv")
    exact = read_table(paths, "32h_exact_policy_metrics.csv")
    predictions = read_table(paths, "32f2_best_reproduced_retained_predictions.csv")
    window = read_table(paths, "32g_numeric_operating_window.csv")
    split_metrics = read_table(paths, "32f2_best_policy_split_stability.csv")
    dominance = read_table(paths, "32f2_best_policy_literature_group_dominance.csv")
    represent = read_table(paths, "32g_representativeness_summary.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=False)
    outer = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.02, 1.38, 1.02],
        height_ratios=[0.98, 1.12],
        left=0.055, right=0.978, top=0.965, bottom=0.082,
        wspace=0.40, hspace=0.42,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1:])
    panel_a(ax_a, evidence)
    panel_b(ax_b, sensitivity, plateau, exact)
    panel_c(outer[1, 0], predictions)
    panel_d(outer[1, 1], window)
    ax_e = fig.add_subplot(outer[1, 2])
    panel_e(ax_e, split_metrics, dominance, represent)

    return fig, {
        "evidence": evidence,
        "exact": exact,
        "plateau": plateau,
        "split_metrics": split_metrics,
        "dominance": dominance,
    }


def write_outputs(paths: Paths, trace: dict[str, pd.DataFrame], runtime_s: float) -> None:
    trace["evidence"].to_csv(paths.table_dir / f"{FIGURE_STEM}_evidence.csv",
                             index=False, encoding="utf-8-sig")
    trace["split_metrics"].to_csv(paths.table_dir / f"{FIGURE_STEM}_split_stability.csv",
                                  index=False, encoding="utf-8-sig")
    audit = (
        f"{SCRIPT_NAME}\n"
        f"figure={FIGURE_STEM}\n"
        f"exact_policy_R2={trace['exact'].iloc[0]['R2']:.6f}\n"
        f"exact_policy_RMSE={trace['exact'].iloc[0]['RMSE']:.6f}\n"
        f"exact_policy_MAE={trace['exact'].iloc[0]['MAE']:.6f}\n"
        f"exact_policy_coverage={trace['exact'].iloc[0]['Coverage_within_strategy_method']:.6f}\n"
        f"dominant_group={trace['dominance'].iloc[0]['Literature_Group']}\n"
        f"dominant_group_fraction={trace['dominance'].iloc[0]['Fraction_of_Retained']:.6f}\n"
        f"runtime_s={runtime_s:.2f}\n"
        f"python={sys.version.split()[0]}\n"
        f"platform={platform.platform()}\n"
    )
    (paths.report_dir / f"{SCRIPT_NAME}_audit.txt").write_text(audit, encoding="utf-8")
    (paths.log_dir / f"{SCRIPT_NAME}.log").write_text(audit, encoding="utf-8")


def main() -> None:
    t0 = time.time()
    configure_matplotlib()
    paths = build_paths()
    fig, trace = create_figure(paths)

    png = paths.fig_dir / f"{FIGURE_STEM}.png"
    pdf = paths.fig_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    runtime = time.time() - t0
    write_outputs(paths, trace, runtime)
    print(f"[{SCRIPT_NAME}] PNG: {png}")
    print(f"[{SCRIPT_NAME}] PDF: {pdf}")
    print(f"[{SCRIPT_NAME}] Total runtime: {runtime:.2f} s")


if __name__ == "__main__":
    main()
