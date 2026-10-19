# -*- coding: utf-8 -*-
"""
Figure 8 - journal-layout SHAP + PDP interpretation.

Recommended location:
    fluoride_coagulation_ml_project/src/

Outputs:
    outputs/jece_final_package/figures/
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from catboost import Pool

SCRIPT_NAME = "34e_est_figure8_shap_pdp_journal_layout_v8"
FIG_STEM = "Figure_8_shap_and_pdp_integrated_interpretation_v8"
DPI = 600

COLORS = {
    "water": "#8FC9D5",
    "coag": "#EFD48C",
    "operation": "#C9D0D7",
    "axis": "#22272E",
    "grid": "#DDE2E7",
    "rose": "#E64675",
    "blue": "#1676CF",
    "darkblue": "#2D5D86",
    "teal": "#2A7F8A",
    "gold": "#C79543",
    "gray": "#6E7881",
}

FEATURE_LABELS = {
    "氟形态": "Fluoride form",
    "游离氟占比": "Free-fluoride fraction",
    "初始氟_mg_L": "Initial fluoride",
    "干扰离子": "Interfering ion",
    "干扰离子浓度_mmol_L": "Interfering-ion concentration",
    "水温_C": "Water temperature",
    "混凝剂": "Coagulant type",
    "剂量_mM": "Coagulant dosage",
    "进水pH": "Influent pH",
    "停留时间_min": "Residence time",
}

GROUPS = {
    "Water matrix": ["氟形态", "游离氟占比", "初始氟_mg_L", "干扰离子", "干扰离子浓度_mmol_L", "水温_C"],
    "Coagulant system": ["混凝剂", "剂量_mM"],
    "Operating conditions": ["进水pH", "停留时间_min"],
}
GROUP_COLORS = {
    "Water matrix": COLORS["water"],
    "Coagulant system": COLORS["coag"],
    "Operating conditions": COLORS["operation"],
}

PDP_PAIRS = [
    ("剂量_mM", "初始氟_mg_L"),
    ("剂量_mM", "进水pH"),
]


def locate_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "processed").exists() and (p / "outputs").exists():
            return p
    raise FileNotFoundError("Project root not found.")


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
    })


def clean_input(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = df[bundle["input_features"]].copy()
    for col in bundle["num_cols"]:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(float(bundle["medians"][col]))
    for col in bundle["cat_cols"]:
        X[col] = X[col].astype(str).fillna("Missing")
    return X


def get_shap(model, X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    pool = Pool(X, cat_features=cat_cols)
    values = model.get_feature_importance(pool, type="ShapValues")
    return np.asarray(values[:, :-1], dtype=float)


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def normalized_feature_values(X: pd.DataFrame, feature: str) -> np.ndarray:
    s = X[feature]
    if pd.api.types.is_numeric_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            return np.full(len(arr), 0.5)
        lo, hi = np.nanquantile(arr[finite], [0.05, 0.95])
        if hi <= lo:
            return np.full(len(arr), 0.5)
        return np.clip((arr - lo) / (hi - lo), 0, 1)
    codes, _ = pd.factorize(s.astype(str), sort=True)
    return np.full(len(codes), 0.5) if codes.max() <= 0 else codes / codes.max()


def panel_label(ax, label: str) -> None:
    # Put labels just inside each panel to avoid clipping after journal scaling.
    if hasattr(ax, "text2D"):
        try:
            ax.text2D(0.01, 0.99, label, transform=ax.transAxes, fontsize=11.5,
                      fontweight="bold", va="top", ha="left")
            return
        except Exception:
            pass
    ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left")


def plot_shap_panel(ax, X: pd.DataFrame, shap_values: np.ndarray,
                    feature_order: list[str], title: str, label: str,
                    seed: int, shap_lim: float, bar_lim: float) -> None:
    panel_label(ax, label)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(300, len(X)), replace=False) if len(X) > 300 else np.arange(len(X))
    shap_sub = shap_values[idx]
    X_sub = X.iloc[idx].reset_index(drop=True)
    feature_index = {feature: i for i, feature in enumerate(X.columns)}
    y_positions = np.arange(len(feature_order))[::-1]

    ax_top = ax.twiny()
    means = np.array([np.mean(np.abs(shap_values[:, feature_index[feature]])) for feature in feature_order])
    bar_colors = [GROUP_COLORS[next(group for group, feats in GROUPS.items() if feature in feats)]
                  for feature in feature_order]
    ax_top.barh(y_positions, means, height=0.68, color=bar_colors,
                alpha=0.54, edgecolor="none", zorder=0)
    ax_top.set_xlim(0, bar_lim)
    ax_top.set_xlabel("Mean |SHAP|", labelpad=2)
    ax_top.tick_params(axis="x", labelsize=6.7, pad=1)
    for side in ["left", "right", "bottom"]:
        ax_top.spines[side].set_visible(False)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["top"].set_color("#B8BFC6")
    ax_top.patch.set_alpha(0)

    cmap = LinearSegmentedColormap.from_list(
        "feature_value", [COLORS["blue"], "#F7F7F7", COLORS["rose"]]
    )
    for row, feature in zip(y_positions, feature_order):
        j = feature_index[feature]
        values = shap_sub[:, j]
        color_values = normalized_feature_values(X_sub, feature)
        jitter = rng.normal(0, 0.085, size=len(values))
        ax.scatter(values, row + jitter, c=color_values, cmap=cmap, vmin=0, vmax=1,
                   s=6.5, alpha=0.88, linewidths=0, zorder=3)

    ax.set_xlim(-shap_lim, shap_lim)
    ax.axvline(0, color="#5E6972", lw=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_LABELS[feature] for feature in feature_order])
    ax.set_xlabel("SHAP value")
    ax.set_title(title, pad=20)
    ax.grid(axis="x")
    ax.set_axisbelow(True)


def group_percentages(shap_values: np.ndarray, features: list[str]) -> dict[str, float]:
    means = np.mean(np.abs(shap_values), axis=0)
    feature_map = {feature: means[i] for i, feature in enumerate(features)}
    total = sum(feature_map.values())
    return {
        group: 100 * sum(feature_map[feature] for feature in group_features) / total
        for group, group_features in GROUPS.items()
    }


def plot_composition(ax, shap_sets: dict[str, np.ndarray], features: list[str], label: str) -> None:
    panel_label(ax, label)
    row_names = list(shap_sets.keys())
    y_positions = np.arange(len(row_names))[::-1]
    for y, name in zip(y_positions, row_names):
        percentages = group_percentages(shap_sets[name], features)
        left = 0.0
        for group in ["Operating conditions", "Water matrix", "Coagulant system"]:
            width = percentages[group]
            ax.barh(y, width, left=left, height=0.48, color=GROUP_COLORS[group],
                    edgecolor="#7B838B", linewidth=0.5)
            if width >= 9.5:
                ax.text(left + width / 2, y, f"{width:.1f}%", ha="center", va="center",
                        fontsize=7.2, fontweight="bold")
            left += width

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, len(row_names) - 0.15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_names)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Contribution based on mean |SHAP| (%)")
    ax.set_title("Feature-group contribution", pad=10)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=GROUP_COLORS[group], edgecolor="#7B838B", label=group)
        for group in ["Water matrix", "Coagulant system", "Operating conditions"]
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
              ncol=3, fontsize=6.8, handlelength=1.1, columnspacing=1.0,
              borderaxespad=0.0)


def compute_pdp_surface(model, X_ref: pd.DataFrame, feature_x: str, feature_y: str,
                        n_x: int = 18, n_y: int = 17):
    raw_x = pd.to_numeric(X_ref[feature_x], errors="coerce")
    raw_y = pd.to_numeric(X_ref[feature_y], errors="coerce")
    grid_x = np.linspace(np.nanquantile(raw_x, 0.10), np.nanquantile(raw_x, 0.90), n_x)
    grid_y = np.linspace(np.nanquantile(raw_y, 0.10), np.nanquantile(raw_y, 0.90), n_y)
    surface = np.zeros((len(grid_y), len(grid_x)))
    for i, value_y in enumerate(grid_y):
        for j, value_x in enumerate(grid_x):
            X_grid = X_ref.copy()
            X_grid[feature_x] = value_x
            X_grid[feature_y] = value_y
            surface[i, j] = predict_model(model, X_grid).mean()
    return grid_x, grid_y, surface


def plot_heatmap_pdp(ax, grid_x, grid_y, surface, feature_x, feature_y,
                     title, label, norm, cmap) -> None:
    panel_label(ax, label)
    mesh = ax.pcolormesh(grid_x, grid_y, surface, shading="auto", cmap=cmap, norm=norm)
    contours = ax.contour(grid_x, grid_y, surface, levels=5, colors="white",
                          linewidths=0.65, alpha=0.85)
    ax.clabel(contours, inline=True, fontsize=6.2, fmt="%.0f")
    ax.set_xlabel(FEATURE_LABELS[feature_x])
    ax.set_ylabel(FEATURE_LABELS[feature_y])
    ax.set_title(title, pad=9)
    return mesh


def plot_surface_pdp(ax, grid_x, grid_y, surface, feature_x, feature_y,
                     title, label, norm, cmap, z_min, z_max) -> None:
    panel_label(ax, label)
    X_mesh, Y_mesh = np.meshgrid(grid_x, grid_y)
    face_colors = cmap(norm(surface))
    ax.plot_surface(
        X_mesh, Y_mesh, surface,
        facecolors=face_colors,
        rstride=1, cstride=1,
        linewidth=0.15,
        edgecolor=(1, 1, 1, 0.16),
        antialiased=True,
        shade=False,
        alpha=0.99,
    )
    floor = z_min - 2.0
    ax.contour(
        X_mesh, Y_mesh, surface,
        zdir="z", offset=floor,
        levels=6, cmap=cmap, norm=norm,
        linewidths=0.72,
    )
    ax.set_zlim(floor, z_max + 0.7)
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-55)

    # A larger axes box with restrained zoom keeps the complete surface,
    # ticks and labels inside the journal page boundary.
    ax.set_box_aspect((1.68, 1.00, 0.66), zoom=1.17)
    ax.set_anchor("C")

    ax.set_xlabel("Dose (mM)", labelpad=0.5)
    y_label = "Initial F (mg L$^{-1}$)" if feature_y == "初始氟_mg_L" else "Influent pH"
    ax.set_ylabel(y_label, labelpad=0.5)
    ax.set_zlabel("")
    ax.text2D(
        0.965, 0.51, "Predicted removal (%)",
        transform=ax.transAxes, rotation=90,
        ha="center", va="center", fontsize=7.7,
    )
    ax.set_title(title, pad=8)

    x_ticks = np.linspace(grid_x.min(), grid_x.max(), 3)
    y_ticks = np.linspace(grid_y.min(), grid_y.max(), 3)
    z_ticks = np.linspace(z_min, z_max, 4)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.set_xticklabels([f"{value:.0f}" for value in x_ticks])
    if feature_y == "初始氟_mg_L":
        ax.set_yticklabels([f"{value:.0f}" for value in y_ticks])
    else:
        ax.set_yticklabels([f"{value:.1f}" for value in y_ticks])
    ax.set_zticklabels([f"{value:.0f}" for value in z_ticks])
    ax.tick_params(axis="x", pad=-1, labelsize=6.8)
    ax.tick_params(axis="y", pad=-1, labelsize=6.8)
    ax.tick_params(axis="z", pad=-1, labelsize=6.8)

    pane_color = (0.985, 0.988, 0.992, 1.0)
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_facecolor(pane_color)
    ax.xaxis.pane.set_edgecolor("#D7DDE3")
    ax.yaxis.pane.set_edgecolor("#D7DDE3")
    ax.zaxis.pane.set_edgecolor("#D7DDE3")
    ax.grid(False)


def main() -> None:
    start = time.time()
    configure_style()
    root = locate_root()
    data_path = root / "data" / "processed" / "model_ready_v1_with_source.csv"
    model_path = root / "outputs" / "models" / "catboost_tuned_model.joblib"
    retained_path = root / "outputs" / "tables" / "32g_unique_retained_chemical_rows.csv"

    out_dir = root / "outputs" / "jece_final_package" / "figures"
    table_dir = root / "outputs" / "jece_final_package" / "tables"
    report_dir = root / "outputs" / "jece_final_package" / "reports"
    log_dir = root / "outputs" / "jece_final_package" / "logs"
    for directory in [out_dir, table_dir, report_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    bundle = joblib.load(model_path)
    model = bundle["catboost_tuned_model"]
    X_all = clean_input(df, bundle)
    shap_all = get_shap(model, X_all, bundle["cat_cols"])

    paper_mask = df["文献编号"].astype(str).eq("Paper_010")
    X_paper = X_all.loc[paper_mask].reset_index(drop=True)
    shap_paper = shap_all[paper_mask.to_numpy()]

    retained = pd.read_csv(retained_path)
    retained_indices = pd.to_numeric(retained["Data_Index"], errors="coerce").dropna().astype(int).unique()
    retained_indices = retained_indices[(retained_indices >= 0) & (retained_indices < len(df))]
    X_retained = X_all.iloc[retained_indices].reset_index(drop=True)
    shap_retained = shap_all[retained_indices]

    importance = np.mean(np.abs(shap_all), axis=0)
    feature_order = [bundle["input_features"][i] for i in np.argsort(importance)[::-1][:8]]
    shap_lim = max(1.0, 1.08 * max(
        np.nanquantile(np.abs(shap_all), 0.995),
        np.nanquantile(np.abs(shap_paper), 0.995),
        np.nanquantile(np.abs(shap_retained), 0.995),
    ))
    bar_lim = 1.10 * max(
        np.mean(np.abs(shap_all), axis=0).max(),
        np.mean(np.abs(shap_paper), axis=0).max(),
        np.mean(np.abs(shap_retained), axis=0).max(),
    )

    X_reference = X_all.sample(n=min(140, len(X_all)), random_state=42).reset_index(drop=True)
    pdp_results = {
        pair: compute_pdp_surface(model, X_reference, pair[0], pair[1])
        for pair in PDP_PAIRS
    }
    all_surface_values = np.concatenate([result[2].ravel() for result in pdp_results.values()])
    z_min = float(np.nanmin(all_surface_values))
    z_max = float(np.nanmax(all_surface_values))
    norm = Normalize(vmin=z_min, vmax=z_max)
    pdp_cmap = mpl.colormaps["YlGnBu"]

    # Journal-ready 3-row layout. The taller third row enlarges the 3D PDPs
    # without forcing their axes, titles or labels beyond the page boundary.
    fig = plt.figure(figsize=(13.2, 11.15))
    grid = GridSpec(
        3, 6, figure=fig,
        height_ratios=[1.03, 0.82, 1.30],
        wspace=0.72, hspace=0.50,
        left=0.065, right=0.962, top=0.958, bottom=0.045,
    )

    ax_a = fig.add_subplot(grid[0, 0:2])
    plot_shap_panel(ax_a, X_all, shap_all, feature_order,
                    f"All-domain SHAP (n={len(X_all):,})", "a", 1, shap_lim, bar_lim)
    ax_b = fig.add_subplot(grid[0, 2:4])
    plot_shap_panel(ax_b, X_paper, shap_paper, feature_order,
                    f"Paper 010 SHAP (n={len(X_paper):,})", "b", 2, shap_lim, bar_lim)
    ax_c = fig.add_subplot(grid[0, 4:6])
    plot_shap_panel(ax_c, X_retained, shap_retained, feature_order,
                    f"Retained-domain SHAP (n={len(X_retained):,})", "c", 3, shap_lim, bar_lim)

    shap_cmap = LinearSegmentedColormap.from_list(
        "feature_value", [COLORS["blue"], "#F7F7F7", COLORS["rose"]]
    )
    shap_scalar = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=shap_cmap)
    cax_shap = ax_c.inset_axes([1.035, 0.24, 0.026, 0.50])
    cb_shap = fig.colorbar(shap_scalar, cax=cax_shap)
    cb_shap.set_ticks([0, 1])
    cb_shap.set_ticklabels(["Low", "High"])
    cb_shap.set_label("Feature value", fontsize=7.2, labelpad=4)
    cb_shap.ax.tick_params(labelsize=6.7)

    ax_d = fig.add_subplot(grid[1, 0:2])
    plot_composition(
        ax_d,
        {"All domain": shap_all, "Paper 010": shap_paper, "Retained": shap_retained},
        bundle["input_features"], "d"
    )

    grid_x_1, grid_y_1, surface_1 = pdp_results[("剂量_mM", "初始氟_mg_L")]
    ax_e = fig.add_subplot(grid[1, 2:4])
    heatmap_1 = plot_heatmap_pdp(
        ax_e, grid_x_1, grid_y_1, surface_1,
        "剂量_mM", "初始氟_mg_L",
        "2D PDP: dosage x initial fluoride", "e", norm, pdp_cmap
    )

    grid_x_2, grid_y_2, surface_2 = pdp_results[("剂量_mM", "进水pH")]
    ax_f = fig.add_subplot(grid[1, 4:6])
    plot_heatmap_pdp(
        ax_f, grid_x_2, grid_y_2, surface_2,
        "剂量_mM", "进水pH",
        "2D PDP: dosage x influent pH", "f", norm, pdp_cmap
    )
    cax_pdp = ax_f.inset_axes([1.035, 0.16, 0.027, 0.66])
    cb_pdp = fig.colorbar(heatmap_1, cax=cax_pdp)
    cb_pdp.set_label("Predicted removal (%)", fontsize=7.2, labelpad=4)
    cb_pdp.ax.tick_params(labelsize=6.7)

    ax_g = fig.add_subplot(grid[2, 0:3], projection="3d")
    plot_surface_pdp(
        ax_g, grid_x_1, grid_y_1, surface_1,
        "剂量_mM", "初始氟_mg_L",
        "3D PDP: dosage x initial fluoride", "g",
        norm, pdp_cmap, z_min, z_max
    )

    ax_h = fig.add_subplot(grid[2, 3:6], projection="3d")
    plot_surface_pdp(
        ax_h, grid_x_2, grid_y_2, surface_2,
        "剂量_mM", "进水pH",
        "3D PDP: dosage x influent pH", "h",
        norm, pdp_cmap, z_min, z_max
    )

    png_path = out_dir / f"{FIG_STEM}.png"
    pdf_path = out_dir / f"{FIG_STEM}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "feature": bundle["input_features"],
        "mean_abs_shap_all": np.mean(np.abs(shap_all), axis=0),
        "mean_abs_shap_paper010": np.mean(np.abs(shap_paper), axis=0),
        "mean_abs_shap_retained": np.mean(np.abs(shap_retained), axis=0),
    }).sort_values("mean_abs_shap_all", ascending=False).to_csv(
        table_dir / f"{FIG_STEM}_feature_importance.csv",
        index=False, encoding="utf-8-sig"
    )

    rows = []
    for (feature_x, feature_y), (grid_x, grid_y, surface) in pdp_results.items():
        for i, value_y in enumerate(grid_y):
            for j, value_x in enumerate(grid_x):
                rows.append({
                    "feature_x": feature_x,
                    "feature_y": feature_y,
                    "x": value_x,
                    "y": value_y,
                    "predicted_removal": surface[i, j],
                })
    pd.DataFrame(rows).to_csv(
        table_dir / f"{FIG_STEM}_pdp_surfaces.csv",
        index=False, encoding="utf-8-sig"
    )

    audit = (
        f"{SCRIPT_NAME}\n"
        f"all_n={len(X_all)}\n"
        f"paper010_n={len(X_paper)}\n"
        f"retained_n={len(X_retained)}\n"
        f"model=catboost_tuned_model.joblib\n"
        f"runtime_s={time.time() - start:.2f}\n"
        f"python={sys.version.split()[0]}\n"
        f"platform={platform.platform()}\n"
    )
    (report_dir / f"{SCRIPT_NAME}_audit.txt").write_text(audit, encoding="utf-8")
    (log_dir / f"{SCRIPT_NAME}.log").write_text(audit, encoding="utf-8")

    print(f"Figure PNG: {png_path}")
    print(f"Figure PDF: {pdf_path}")
    print(f"Runtime: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
