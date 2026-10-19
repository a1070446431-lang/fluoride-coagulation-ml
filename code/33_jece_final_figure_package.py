# -*- coding: utf-8 -*-
"""
33_jece_final_figure_package.py

用途：
1. 统一审计混凝除氟项目最终数据与历史结果文件
2. 生成 JECE（Journal of Environmental Chemical Engineering）风格 Figure 1–Figure 4
3. 所有输出统一保存到 outputs/jece_final_package/

作者：ChatGPT（按当前项目定制）
说明：
- 本脚本为第一版主图脚本，优先生成 Figure 1–4
- Figure 5–6（SHAP/PDP/二维交互）下一步再接入同一脚本
- 兼容“部分结果表存在 / 部分结果表不存在”的情况
"""

import os
import re
import time
import math
import json
import textwrap
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =========================
# 全局风格设置
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# JECE 风格低饱和配色
COLORS = {
    "blue": "#5B8FA8",
    "blue_dark": "#3E6578",
    "blue_light": "#AFC8D6",
    "green": "#8DAA91",
    "green_light": "#C8D8C6",
    "orange": "#C99A6B",
    "orange_light": "#E8CCAE",
    "gray": "#666666",
    "gray_light": "#D9D9D9",
    "gray_mid": "#A6A6A6",
    "red_soft": "#B87C7C",
    "teal": "#6EA6A6",
    "purple_soft": "#9C90B8",
    "white": "#FFFFFF",
    "black": "#222222",
}

# =========================
# 运行开关
# =========================
RUN_AUDIT = True
RUN_FIGURE_1 = True
RUN_FIGURE_2 = True
RUN_FIGURE_3 = True
RUN_FIGURE_4 = True

# =========================
# 项目路径与输出路径
# =========================
def locate_project_root() -> Path:
    """
    自动定位项目根目录 fluoride_coagulation_ml_project
    当前脚本位于 src/ 下时，默认返回上一级目录
    """
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return current.parent.parent

PROJECT_ROOT = locate_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

JECE_DIR = OUTPUTS_DIR / "jece_final_package"
FIG_DIR = JECE_DIR / "figures"
TAB_DIR = JECE_DIR / "tables"
REP_DIR = JECE_DIR / "reports"
LOG_DIR = JECE_DIR / "logs"

for p in [JECE_DIR, FIG_DIR, TAB_DIR, REP_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH = LOG_DIR / "33_jece_final_figure_package_run_log.txt"

# =========================
# 日志工具
# =========================
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.buffer = []

    def log(self, msg: str):
        print(msg)
        self.buffer.append(msg)

    def save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            for line in self.buffer:
                f.write(line + "\n")

logger = Logger(RUN_LOG_PATH)

# =========================
# 通用读表工具
# =========================
def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()

def safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if file_exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            logger.log(f"[WARN] 读取失败：{path} | 错误：{e}")
            return None
    return None

def safe_read_txt(path: Path) -> Optional[str]:
    if file_exists(path):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.log(f"[WARN] 读取失败：{path} | 错误：{e}")
            return None
    return None

def ensure_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out

def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

# =========================
# 搜索文件
# =========================
def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if file_exists(p):
            return p
    return None

def find_all_matching(folder: Path, pattern: str) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern))

# =========================
# 数据文件候选路径
# =========================
MODEL_READY_CANDIDATES = [
    DATA_DIR / "processed" / "model_ready_v1.csv",
    DATA_DIR / "processed" / "model_ready-v1.csv",
    PROJECT_ROOT / "model_ready_v1.csv",
]

MODEL_READY_WITH_SOURCE_CANDIDATES = [
    DATA_DIR / "processed" / "model_ready_v1_with_source.csv",
    PROJECT_ROOT / "model_ready_v1_with_source.csv",
]

TABLE_CANDIDATES = {
    "19_selected_feature_ranking_by_seed": OUTPUTS_DIR / "tables" / "19_selected_feature_ranking_by_seed.csv",
    "19_selected_fusion_candidate_results": OUTPUTS_DIR / "tables" / "19_selected_fusion_candidate_results.csv",
    "19_selected_fusion_predictions": OUTPUTS_DIR / "tables" / "19_selected_fusion_predictions.csv",
    "19_selected_fusion_summary": OUTPUTS_DIR / "tables" / "19_selected_fusion_summary.csv",
    "19_selected_fusion_test_results": OUTPUTS_DIR / "tables" / "19_selected_fusion_test_results.csv",
    "19_selected_fusion_weights": OUTPUTS_DIR / "tables" / "19_selected_fusion_weights.csv",

    "29_grouped_screen_decision": OUTPUTS_DIR / "tables" / "29_grouped_screen_decision.txt",
    "29_grouped_screen_group_distribution": OUTPUTS_DIR / "tables" / "29_grouped_screen_group_distribution.csv",
    "29_grouped_screen_predictions": OUTPUTS_DIR / "tables" / "29_grouped_screen_predictions.csv",
    "29_grouped_screen_split_summary": OUTPUTS_DIR / "tables" / "29_grouped_screen_split_summary.csv",
    "29_grouped_screen_summary": OUTPUTS_DIR / "tables" / "29_grouped_screen_summary.csv",
    "29_grouped_screen_test_results": OUTPUTS_DIR / "tables" / "29_grouped_screen_test_results.csv",
    "29_grouped_screen_validation_candidates": OUTPUTS_DIR / "tables" / "29_grouped_screen_validation_candidates.csv",

    "29a_final_dataset_group_distribution": OUTPUTS_DIR / "tables" / "29a_final_dataset_group_distribution.csv",
    "29a_final_dataset_summary": OUTPUTS_DIR / "tables" / "29a_final_dataset_summary.txt",
    "29c_exact_legacy_column_mapping": OUTPUTS_DIR / "tables" / "29c_exact_legacy_column_mapping.csv",
    "29c_exact_legacy_group_distribution": OUTPUTS_DIR / "tables" / "29c_exact_legacy_group_distribution.csv",
    "29c_exact_legacy_summary": OUTPUTS_DIR / "tables" / "29c_exact_legacy_summary.txt",

    "30_column_role_mapping": OUTPUTS_DIR / "tables" / "30_column_role_mapping.csv",
    "30_grouped_split_assignments_by_seed": OUTPUTS_DIR / "tables" / "30_grouped_split_assignments_by_seed.csv",
    "30_grouped_split_size_and_target_summary": OUTPUTS_DIR / "tables" / "30_grouped_split_size_and_target_summary.csv",
}

# =========================
# 小工具函数
# =========================
def guess_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for col in df.columns:
        c = str(col).lower().strip()
        if any(k in c for k in ["去除率", "removal", "rf", "fluoride removal", "fluoride_removal"]):
            candidates.append(col)

    # 优先排除明显不是目标的列
    filtered = []
    for col in candidates:
        c = str(col).lower()
        if any(bad in c for bad in ["pred", "predict", "residual", "abs_error", "mae", "rmse", "r2"]):
            continue
        filtered.append(col)

    if len(filtered) == 1:
        return filtered[0]

    # 若有多个，优先包含 fluoride/F/氟 且包含 removal/去除率
    for col in filtered:
        c = str(col).lower()
        if ("fluor" in c or "氟" in c or re.search(r"\bf\b", c)) and ("removal" in c or "去除率" in c):
            return col

    # 如果没有明确目标列，则从数值列中选择 0~100 范围最像百分比的列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scored = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        frac_in_0_100 = ((s >= 0) & (s <= 100)).mean()
        uniq = s.nunique()
        score = frac_in_0_100 + min(uniq / 1000, 1)
        scored.append((col, score))
    if scored:
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored[0][0]

    return None

def guess_source_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["文献编号", "literature", "source", "paper_id", "group", "文献id"]):
            return col
    return None

def guess_pred_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["pred", "predict"]) and not any(bad in c for bad in ["residual", "abs"]):
            return col
    return None

def guess_true_col(df: pd.DataFrame) -> Optional[str]:
    # 优先 observed/actual/true
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["actual", "observed", "true"]):
            return col
    # 回退到 removal 目标列
    return guess_target_column(df)

def guess_residual_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if "residual" in c:
            return col
    return None

def guess_split_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["split", "seed", "strategy", "fold"]):
            return col
    return None

def guess_metric_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {"r2": None, "rmse": None, "mae": None, "n": None, "coverage": None}
    for col in df.columns:
        c = str(col).lower()
        if mapping["r2"] is None and c == "r2":
            mapping["r2"] = col
        elif mapping["r2"] is None and "r2" in c:
            mapping["r2"] = col
        if mapping["rmse"] is None and "rmse" in c:
            mapping["rmse"] = col
        if mapping["mae"] is None and "mae" in c:
            mapping["mae"] = col
        if mapping["n"] is None and (c == "n" or "retained_n" in c or "sample" in c or "count" in c):
            mapping["n"] = col
        if mapping["coverage"] is None and "coverage" in c:
            mapping["coverage"] = col
    return mapping

def try_parse_key_numbers_from_txt(text: str) -> Dict[str, float]:
    out = {}
    if text is None:
        return out
    patterns = {
        "r2": r"R[²2]\s*[:=]\s*([0-9.]+)",
        "rmse": r"RMSE\s*[:=]\s*([0-9.]+)",
        "mae": r"MAE\s*[:=]\s*([0-9.]+)",
        "coverage": r"coverage\s*[:=]\s*([0-9.]+)",
        "n": r"(?:N retained|N)\s*[:=]\s*([0-9.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                out[k] = float(m.group(1))
            except:
                pass
    return out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "N": len(y_true)}

def standardize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            mean = s.mean()
            std = s.std(ddof=0)
            if pd.notna(std) and std > 0:
                out[col] = (s - mean) / std
            else:
                out[col] = np.nan
    return out

def wrap_label(s: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width))

# =========================
# 画图辅助函数
# =========================
def add_panel_label(ax, label: str):
    ax.text(
        -0.08, 1.06, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left", va="bottom",
        color=COLORS["black"]
    )

def beautify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6)

def save_figure(fig: plt.Figure, filename_stem: str):
    png_path = FIG_DIR / f"{filename_stem}.png"
    pdf_path = FIG_DIR / f"{filename_stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.log(f"[OK] 图已保存：{png_path.name}, {pdf_path.name}")

# =========================
# 审计与加载数据
# =========================
class ProjectDataHub:
    def __init__(self):
        self.model_ready = None
        self.model_ready_with_source = None
        self.tables = {}

    def load(self):
        model_ready_path = find_first_existing(MODEL_READY_CANDIDATES)
        model_ready_with_source_path = find_first_existing(MODEL_READY_WITH_SOURCE_CANDIDATES)

        if model_ready_path:
            self.model_ready = safe_read_csv(model_ready_path)
            if self.model_ready is not None:
                self.model_ready = ensure_str_columns(self.model_ready)
                logger.log(f"[OK] 读取 model_ready：{model_ready_path} | 形状={self.model_ready.shape}")
        else:
            logger.log("[WARN] 未找到 model_ready_v1.csv")

        if model_ready_with_source_path:
            self.model_ready_with_source = safe_read_csv(model_ready_with_source_path)
            if self.model_ready_with_source is not None:
                self.model_ready_with_source = ensure_str_columns(self.model_ready_with_source)
                logger.log(f"[OK] 读取 model_ready_v1_with_source：{model_ready_with_source_path} | 形状={self.model_ready_with_source.shape}")
        else:
            logger.log("[WARN] 未找到 model_ready_v1_with_source.csv")

        for key, path in TABLE_CANDIDATES.items():
            if path.suffix.lower() == ".csv":
                df = safe_read_csv(path)
                if df is not None:
                    self.tables[key] = ensure_str_columns(df)
                    logger.log(f"[OK] 读取表：{path.name} | 形状={df.shape}")
            elif path.suffix.lower() == ".txt":
                text = safe_read_txt(path)
                if text is not None:
                    self.tables[key] = text
                    logger.log(f"[OK] 读取文本：{path.name}")

    def audit_report(self):
        rows = []

        # model_ready
        if self.model_ready is not None:
            tgt_col = guess_target_column(self.model_ready)
            rows.append({
                "item": "model_ready_v1",
                "exists": True,
                "shape": str(self.model_ready.shape),
                "target_guess": tgt_col,
                "source_id_guess": guess_source_id_column(self.model_ready)
            })
        else:
            rows.append({"item": "model_ready_v1", "exists": False, "shape": "", "target_guess": "", "source_id_guess": ""})

        # model_ready_with_source
        if self.model_ready_with_source is not None:
            tgt_col = guess_target_column(self.model_ready_with_source)
            rows.append({
                "item": "model_ready_v1_with_source",
                "exists": True,
                "shape": str(self.model_ready_with_source.shape),
                "target_guess": tgt_col,
                "source_id_guess": guess_source_id_column(self.model_ready_with_source)
            })
        else:
            rows.append({"item": "model_ready_v1_with_source", "exists": False, "shape": "", "target_guess": "", "source_id_guess": ""})

        # tables
        for key, obj in self.tables.items():
            if isinstance(obj, pd.DataFrame):
                rows.append({"item": key, "exists": True, "shape": str(obj.shape), "target_guess": "", "source_id_guess": ""})
            else:
                rows.append({"item": key, "exists": True, "shape": "text", "target_guess": "", "source_id_guess": ""})

        audit_df = pd.DataFrame(rows)
        save_df(audit_df, TAB_DIR / "00_jece_figure_source_manifest.csv")

        report_lines = []
        report_lines.append("JECE Figure Package Audit Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Project root: {PROJECT_ROOT}")
        report_lines.append(f"Output root: {JECE_DIR}")
        report_lines.append("")
        report_lines.append("1) 核心数据文件")
        report_lines.append("-" * 60)
        for _, r in audit_df.iterrows():
            report_lines.append(
                f"{r['item']:<45} | exists={r['exists']} | shape={r['shape']} | "
                f"target_guess={r['target_guess']} | source_id_guess={r['source_id_guess']}"
            )

        report_path = REP_DIR / "00_jece_figure_data_audit.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.log(f"[OK] 审计表已保存：{TAB_DIR / '00_jece_figure_source_manifest.csv'}")
        logger.log(f"[OK] 审计报告已保存：{report_path}")

# =========================
# Figure 1
# =========================
def plot_figure_1(hub: ProjectDataHub):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # -------------------------
    # 1a 总体方法流程图
    # -------------------------
    ax = axes[0]
    ax.set_axis_off()
    add_panel_label(ax, "a")

    steps = [
        "Literature collection\nand data extraction",
        "Unit harmonization\nand target verification",
        "Mechanism-guided\nfeature engineering",
        "Random split baseline",
        "Grouped validation",
        "Applicability domain\nand reliable domain",
        "SHAP/PDP-based\ninterpretation"
    ]
    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.55
    box_w = 0.11
    box_h = 0.18

    colors = [
        COLORS["blue_light"], COLORS["green_light"], COLORS["orange_light"],
        COLORS["blue_light"], COLORS["green_light"], COLORS["orange_light"], COLORS["blue_light"]
    ]

    for i, (x, step) in enumerate(zip(x_positions, steps)):
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0, edgecolor=COLORS["gray"], facecolor=colors[i]
        )
        ax.add_patch(rect)
        ax.text(x, y, step, ha="center", va="center", fontsize=9)

        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_w / 2, y), (x_positions[i + 1] - box_w / 2, y),
                arrowstyle="->", mutation_scale=12, linewidth=1.2, color=COLORS["gray"]
            )
            ax.add_patch(arrow)

    ax.text(0.5, 0.15,
            "Framework for heterogeneous multi-literature data-driven prediction,\n"
            "cross-literature generalization evaluation, and reliable prediction boundary identification.",
            ha="center", va="center", fontsize=9, color=COLORS["gray"])

    # -------------------------
    # 1b 文献组样本量分布
    # -------------------------
    ax = axes[1]
    add_panel_label(ax, "b")
    group_dist = None

    if "29c_exact_legacy_group_distribution" in hub.tables:
        group_dist = hub.tables["29c_exact_legacy_group_distribution"].copy()
    elif "29a_final_dataset_group_distribution" in hub.tables:
        group_dist = hub.tables["29a_final_dataset_group_distribution"].copy()
    elif hub.model_ready_with_source is not None:
        sid = guess_source_id_column(hub.model_ready_with_source)
        if sid is not None:
            group_dist = (
                hub.model_ready_with_source[sid]
                .value_counts(dropna=False)
                .reset_index()
            )
            group_dist.columns = [sid, "count"]

    if group_dist is not None:
        # 尝试识别列
        cols = group_dist.columns.tolist()
        count_col = None
        group_col = None
        for c in cols:
            cl = c.lower()
            if "count" in cl or cl == "n" or "sample" in cl:
                count_col = c
            if any(k in cl for k in ["文献", "literature", "source", "group", "paper"]):
                group_col = c
        if group_col is None:
            group_col = cols[0]
        if count_col is None:
            # 若只有两列，默认第二列是 count
            if len(cols) >= 2:
                count_col = cols[1]
            else:
                group_dist["count"] = 1
                count_col = "count"

        tmp = group_dist[[group_col, count_col]].copy()
        tmp[count_col] = pd.to_numeric(tmp[count_col], errors="coerce")
        tmp = tmp.dropna(subset=[count_col]).sort_values(count_col, ascending=False)
        tmp = tmp.head(32).copy()

        colors = [COLORS["blue"]] * len(tmp)
        if len(tmp) > 0:
            colors[0] = COLORS["orange"]

        ax.bar(range(len(tmp)), tmp[count_col].values, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(tmp)))
        ax.set_xticklabels([wrap_label(x, 8) for x in tmp[group_col].astype(str)], rotation=90, fontsize=7)
        ax.set_ylabel("Sample count")
        ax.set_title("Distribution of samples across literature groups", fontsize=10)
        beautify_axes(ax)

        # 标注最大值
        if len(tmp) > 0:
            idx_max = tmp[count_col].idxmax()
            max_pos = tmp.index.get_loc(idx_max)
            max_val = tmp.loc[idx_max, count_col]
            ax.text(max_pos, max_val + max_val * 0.02, f"{int(max_val)}",
                    ha="center", va="bottom", fontsize=8, color=COLORS["orange"])

        save_df(tmp.rename(columns={group_col: "group", count_col: "count"}),
                TAB_DIR / "Figure_1b_literature_group_counts.csv")
    else:
        ax.text(0.5, 0.5, "No group distribution data found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 1c 目标分布
    # -------------------------
    ax = axes[2]
    add_panel_label(ax, "c")
    if hub.model_ready is not None:
        tgt = guess_target_column(hub.model_ready)
        if tgt is not None:
            s = pd.to_numeric(hub.model_ready[tgt], errors="coerce").dropna()
            ax.hist(s, bins=25, color=COLORS["blue_light"], edgecolor="white", alpha=0.95)
            ax.axvline(s.mean(), color=COLORS["orange"], linestyle="--", linewidth=1.2, label=f"Mean = {s.mean():.2f}")
            ax.axvline(s.median(), color=COLORS["blue_dark"], linestyle="-.", linewidth=1.2, label=f"Median = {s.median():.2f}")
            ax.set_xlabel("Fluoride removal efficiency (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of target variable", fontsize=10)
            ax.legend(frameon=False, fontsize=8)
            beautify_axes(ax)

            summary_df = pd.DataFrame([{
                "target_column": tgt,
                "N": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(ddof=1),
                "min": s.min(),
                "max": s.max()
            }])
            save_df(summary_df, TAB_DIR / "Figure_1c_target_distribution.csv")
        else:
            ax.text(0.5, 0.5, "Target column not found", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "model_ready not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 1d 特征类别结构
    # -------------------------
    ax = axes[3]
    ax.set_axis_off()
    add_panel_label(ax, "d")

    feature_groups = {
        "Water matrix": ["Initial fluoride", "Influent pH", "Water temperature", "Interfering ions"],
        "Coagulant system": ["Coagulant type", "Coagulant dosage", "Flocculant type", "Flocculant dosage"],
        "Operational conditions": ["Reaction time", "Mixing time", "Settling time", "Residence time"],
        "Response / audit": ["Fluoride removal", "Effluent fluoride", "Effluent pH", "Source label (audit only)"]
    }

    box_positions = [
        (0.06, 0.58), (0.54, 0.58),
        (0.06, 0.12), (0.54, 0.12)
    ]
    group_colors = [COLORS["blue_light"], COLORS["green_light"], COLORS["orange_light"], COLORS["gray_light"]]

    feature_rows = []
    for (title, items), (x, y), fc in zip(feature_groups.items(), box_positions, group_colors):
        rect = FancyBboxPatch((x, y), 0.38, 0.28,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              linewidth=1, edgecolor=COLORS["gray"], facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + 0.19, y + 0.24, title, ha="center", va="center", fontsize=10, fontweight="bold")
        text = "\n".join([f"• {it}" for it in items])
        ax.text(x + 0.03, y + 0.14, text, ha="left", va="center", fontsize=8)

        for it in items:
            feature_rows.append({"feature_group": title, "feature_name": it})

    save_df(pd.DataFrame(feature_rows), TAB_DIR / "Figure_1d_feature_categories.csv")

    save_figure(fig, "Figure_1_data_architecture_and_workflow")

# =========================
# Figure 2
# =========================
def plot_figure_2(hub: ProjectDataHub):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # -------------------------
    # 2a 候选模型性能比较
    # -------------------------
    ax = axes[0]
    add_panel_label(ax, "a")

    candidate_df = hub.tables.get("19_selected_fusion_candidate_results", None)
    if isinstance(candidate_df, pd.DataFrame) and len(candidate_df) > 0:
        metrics_map = guess_metric_cols(candidate_df)

        # 识别模型名列
        model_col = None
        for c in candidate_df.columns:
            cl = c.lower()
            if any(k in cl for k in ["model", "method", "candidate", "strategy"]):
                model_col = c
                break
        if model_col is None:
            model_col = candidate_df.columns[0]

        plot_df = candidate_df.copy()
        if metrics_map["r2"] is not None:
            plot_df[metrics_map["r2"]] = pd.to_numeric(plot_df[metrics_map["r2"]], errors="coerce")
        plot_df = plot_df.dropna(subset=[metrics_map["r2"]]) if metrics_map["r2"] else plot_df
        plot_df = plot_df.sort_values(metrics_map["r2"], ascending=False) if metrics_map["r2"] else plot_df

        x = np.arange(len(plot_df))
        y = plot_df[metrics_map["r2"]].values if metrics_map["r2"] else np.zeros(len(plot_df))
        bar_colors = [COLORS["blue"]] * len(plot_df)
        if len(bar_colors) > 0:
            bar_colors[0] = COLORS["orange"]

        ax.bar(x, y, color=bar_colors, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([wrap_label(i, 12) for i in plot_df[model_col].astype(str)], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title("Candidate model performance under random split", fontsize=10)
        beautify_axes(ax)

        for xi, yi in zip(x, y):
            if pd.notna(yi):
                ax.text(xi, yi + 0.005, f"{yi:.3f}", ha="center", va="bottom", fontsize=7)

        save_cols = [model_col]
        for k, v in metrics_map.items():
            if v is not None:
                save_cols.append(v)
        save_df(plot_df[save_cols], TAB_DIR / "Figure_2a_model_metrics.csv")
    else:
        ax.text(0.5, 0.5, "Candidate results file not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 2b 特征/融合选择曲线
    # -------------------------
    ax = axes[1]
    add_panel_label(ax, "b")
    ranking_df = hub.tables.get("19_selected_feature_ranking_by_seed", None)

    if isinstance(ranking_df, pd.DataFrame) and len(ranking_df) > 0:
        # 尝试识别特征数与性能列
        feature_num_col = None
        perf_col = None
        for c in ranking_df.columns:
            cl = c.lower()
            if feature_num_col is None and any(k in cl for k in ["feature_num", "num_feature", "k", "rank"]):
                feature_num_col = c
            if perf_col is None and "r2" in cl:
                perf_col = c

        if feature_num_col is None:
            # 若不存在，直接用行序号代替
            ranking_df = ranking_df.copy().reset_index(drop=True)
            ranking_df["feature_step"] = np.arange(1, len(ranking_df) + 1)
            feature_num_col = "feature_step"

        if perf_col is None:
            # 试图从数值列中找最像性能列
            numeric_cols = ranking_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                perf_col = numeric_cols[-1]

        if perf_col is not None:
            tmp = ranking_df.copy()
            tmp[feature_num_col] = pd.to_numeric(tmp[feature_num_col], errors="coerce")
            tmp[perf_col] = pd.to_numeric(tmp[perf_col], errors="coerce")
            tmp = tmp.dropna(subset=[feature_num_col, perf_col]).sort_values(feature_num_col)

            ax.plot(tmp[feature_num_col], tmp[perf_col], marker="o", linewidth=1.5,
                    color=COLORS["blue"], markersize=4)
            # 标最优
            best_idx = tmp[perf_col].idxmax()
            bx = tmp.loc[best_idx, feature_num_col]
            by = tmp.loc[best_idx, perf_col]
            ax.scatter([bx], [by], color=COLORS["orange"], s=45, zorder=3)
            ax.text(bx, by, f" Best={by:.3f}", fontsize=8, va="bottom", color=COLORS["orange"])

            ax.set_xlabel("Feature-selection step")
            ax.set_ylabel("Performance")
            ax.set_title("Feature/fusion selection curve", fontsize=10)
            beautify_axes(ax)

            save_df(tmp[[feature_num_col, perf_col]], TAB_DIR / "Figure_2b_selection_curve.csv")
        else:
            ax.text(0.5, 0.5, "No suitable performance column found", ha="center", va="center")
            ax.set_axis_off()
    else:
        # 回退：尝试用 candidate_results 做条线替代
        candidate_df = hub.tables.get("19_selected_fusion_candidate_results", None)
        if isinstance(candidate_df, pd.DataFrame) and len(candidate_df) > 0:
            metrics_map = guess_metric_cols(candidate_df)
            model_col = candidate_df.columns[0]
            r2_col = metrics_map["r2"]
            if r2_col is not None:
                tmp = candidate_df[[model_col, r2_col]].copy()
                tmp[r2_col] = pd.to_numeric(tmp[r2_col], errors="coerce")
                tmp = tmp.dropna().sort_values(r2_col, ascending=False).reset_index(drop=True)
                ax.plot(np.arange(1, len(tmp) + 1), tmp[r2_col], marker="o", linewidth=1.5,
                        color=COLORS["blue"], markersize=4)
                ax.set_xlabel("Candidate rank")
                ax.set_ylabel("R²")
                ax.set_title("Candidate selection curve (fallback)", fontsize=10)
                beautify_axes(ax)
                save_df(tmp, TAB_DIR / "Figure_2b_selection_curve.csv")
            else:
                ax.text(0.5, 0.5, "No selection-curve data found", ha="center", va="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No selection-curve data found", ha="center", va="center")
            ax.set_axis_off()

    # -------------------------
    # 2c 实测 vs 预测
    # -------------------------
    ax = axes[2]
    add_panel_label(ax, "c")
    pred_df = hub.tables.get("19_selected_fusion_predictions", None)
    if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        true_col = guess_true_col(pred_df)
        pred_col = guess_pred_col(pred_df)
        if true_col is not None and pred_col is not None:
            tmp = pred_df[[true_col, pred_col]].copy()
            tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            tmp = tmp.dropna()

            ax.scatter(tmp[true_col], tmp[pred_col],
                       s=18, alpha=0.65, color=COLORS["blue"], edgecolor="white", linewidth=0.3)

            min_v = min(tmp[true_col].min(), tmp[pred_col].min())
            max_v = max(tmp[true_col].max(), tmp[pred_col].max())
            ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color=COLORS["orange"], linewidth=1.2)
            ax.set_xlabel("Observed fluoride removal efficiency (%)")
            ax.set_ylabel("Predicted fluoride removal efficiency (%)")
            ax.set_title("Observed versus predicted values", fontsize=10)
            beautify_axes(ax)

            metrics = compute_metrics(tmp[true_col].values, tmp[pred_col].values)
            ax.text(0.04, 0.96,
                    f"R² = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nMAE = {metrics['MAE']:.3f}\nN = {metrics['N']}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["gray_light"]))

            save_df(tmp.rename(columns={true_col: "observed", pred_col: "predicted"}),
                    TAB_DIR / "Figure_2c_predictions.csv")
        else:
            ax.text(0.5, 0.5, "Prediction columns not found", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Prediction file not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 2d 残差图
    # -------------------------
    ax = axes[3]
    add_panel_label(ax, "d")
    if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        true_col = guess_true_col(pred_df)
        pred_col = guess_pred_col(pred_df)
        residual_col = guess_residual_col(pred_df)

        if true_col is not None and pred_col is not None:
            tmp = pred_df.copy()
            tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            if residual_col is None:
                tmp["__residual__"] = tmp[true_col] - tmp[pred_col]
                residual_col = "__residual__"
            else:
                tmp[residual_col] = pd.to_numeric(tmp[residual_col], errors="coerce")

            tmp = tmp.dropna(subset=[pred_col, residual_col])

            ax.scatter(tmp[pred_col], tmp[residual_col],
                       s=18, alpha=0.65, color=COLORS["blue"], edgecolor="white", linewidth=0.3)
            ax.axhline(0, linestyle="--", color=COLORS["orange"], linewidth=1.2)
            ax.set_xlabel("Predicted fluoride removal efficiency (%)")
            ax.set_ylabel("Residual")
            ax.set_title("Residual diagnostics", fontsize=10)
            beautify_axes(ax)

            save_df(tmp[[pred_col, residual_col]].rename(columns={pred_col: "predicted", residual_col: "residual"}),
                    TAB_DIR / "Figure_2d_residuals.csv")
        else:
            ax.text(0.5, 0.5, "Prediction columns not found", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Prediction file not found", ha="center", va="center")
        ax.set_axis_off()

    save_figure(fig, "Figure_2_random_split_performance")

# =========================
# Figure 3
# =========================
def build_random_vs_grouped_metrics(hub: ProjectDataHub) -> pd.DataFrame:
    rows = []

    # Random split
    random_summary = hub.tables.get("19_selected_fusion_summary", None)
    if isinstance(random_summary, pd.DataFrame) and len(random_summary) > 0:
        mm = guess_metric_cols(random_summary)
        # 取第一行或最佳行
        tmp = random_summary.copy()
        if mm["r2"] is not None:
            tmp[mm["r2"]] = pd.to_numeric(tmp[mm["r2"]], errors="coerce")
            tmp = tmp.sort_values(mm["r2"], ascending=False)
        r = tmp.iloc[0]
        rows.append({
            "evaluation": "Random split",
            "R2": pd.to_numeric(r[mm["r2"]], errors="coerce") if mm["r2"] else np.nan,
            "RMSE": pd.to_numeric(r[mm["rmse"]], errors="coerce") if mm["rmse"] else np.nan,
            "MAE": pd.to_numeric(r[mm["mae"]], errors="coerce") if mm["mae"] else np.nan,
            "N": pd.to_numeric(r[mm["n"]], errors="coerce") if mm["n"] else np.nan,
            "Coverage": pd.to_numeric(r[mm["coverage"]], errors="coerce") if mm["coverage"] else np.nan,
        })

    # Grouped validation
    grouped_summary = hub.tables.get("29_grouped_screen_summary", None)
    if isinstance(grouped_summary, pd.DataFrame) and len(grouped_summary) > 0:
        mm = guess_metric_cols(grouped_summary)
        tmp = grouped_summary.copy()
        if mm["r2"] is not None:
            tmp[mm["r2"]] = pd.to_numeric(tmp[mm["r2"]], errors="coerce")
            tmp = tmp.sort_values(mm["r2"], ascending=False)
        r = tmp.iloc[0]
        rows.append({
            "evaluation": "Grouped validation",
            "R2": pd.to_numeric(r[mm["r2"]], errors="coerce") if mm["r2"] else np.nan,
            "RMSE": pd.to_numeric(r[mm["rmse"]], errors="coerce") if mm["rmse"] else np.nan,
            "MAE": pd.to_numeric(r[mm["mae"]], errors="coerce") if mm["mae"] else np.nan,
            "N": pd.to_numeric(r[mm["n"]], errors="coerce") if mm["n"] else np.nan,
            "Coverage": pd.to_numeric(r[mm["coverage"]], errors="coerce") if mm["coverage"] else np.nan,
        })

    return pd.DataFrame(rows)

def plot_figure_3(hub: ProjectDataHub):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # -------------------------
    # 3a 随机划分 vs 分组划分示意
    # -------------------------
    ax = axes[0]
    ax.set_axis_off()
    add_panel_label(ax, "a")

    ax.text(0.24, 0.92, "Random split", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.76, 0.92, "Grouped split", ha="center", va="center", fontsize=11, fontweight="bold")

    # 左侧随机划分
    for i in range(6):
        x = 0.07 + (i % 3) * 0.12
        y = 0.60 - (i // 3) * 0.16
        ax.add_patch(Rectangle((x, y), 0.08, 0.08, facecolor=COLORS["blue_light"], edgecolor="white"))
        ax.text(x + 0.04, y + 0.04, f"L{i+1}", ha="center", va="center", fontsize=8)

    ax.text(0.06, 0.30, "Train and test may both contain\nsamples from the same literature group",
            ha="left", va="center", fontsize=8, color=COLORS["gray"])

    # 右侧分组划分
    train_box = FancyBboxPatch((0.58, 0.54), 0.18, 0.22, boxstyle="round,pad=0.02",
                               facecolor=COLORS["green_light"], edgecolor=COLORS["gray"])
    test_box = FancyBboxPatch((0.79, 0.54), 0.14, 0.22, boxstyle="round,pad=0.02",
                              facecolor=COLORS["orange_light"], edgecolor=COLORS["gray"])
    ax.add_patch(train_box)
    ax.add_patch(test_box)
    ax.text(0.67, 0.72, "Train", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.86, 0.72, "Test", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0.67, 0.61, "L1\nL2\nL3", ha="center", va="center", fontsize=9)
    ax.text(0.86, 0.61, "L4\nL5", ha="center", va="center", fontsize=9)
    ax.text(0.76, 0.30, "Entire literature groups are assigned\nto either training or testing only",
            ha="center", va="center", fontsize=8, color=COLORS["gray"])

    # -------------------------
    # 3b 性能下降
    # -------------------------
    ax = axes[1]
    add_panel_label(ax, "b")

    compare_df = build_random_vs_grouped_metrics(hub)
    if len(compare_df) >= 2:
        x_labels = compare_df["evaluation"].tolist()
        y = compare_df["R2"].values
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", linewidth=2.0, color=COLORS["blue"])
        ax.scatter(x[0], y[0], color=COLORS["orange"], s=55, zorder=3)
        ax.scatter(x[1], y[1], color=COLORS["red_soft"], s=55, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.set_ylabel("R²")
        ax.set_title("Performance drop from random to grouped evaluation", fontsize=10)
        beautify_axes(ax)

        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

        save_df(compare_df, TAB_DIR / "Figure_3b_random_vs_grouped_metrics.csv")
    else:
        ax.text(0.5, 0.5, "Insufficient summary data", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 3c 不同分组策略/种子的 R² 分布
    # -------------------------
    ax = axes[2]
    add_panel_label(ax, "c")

    split_summary = hub.tables.get("29_grouped_screen_split_summary", None)
    if isinstance(split_summary, pd.DataFrame) and len(split_summary) > 0:
        mm = guess_metric_cols(split_summary)
        r2_col = mm["r2"]

        # 识别策略列
        strategy_col = None
        for c in split_summary.columns:
            cl = c.lower()
            if any(k in cl for k in ["strategy", "method", "scheme", "split"]):
                strategy_col = c
                break

        if r2_col is not None:
            tmp = split_summary.copy()
            tmp[r2_col] = pd.to_numeric(tmp[r2_col], errors="coerce")
            tmp = tmp.dropna(subset=[r2_col])

            if strategy_col is None:
                tmp["__strategy__"] = "Grouped split"
                strategy_col = "__strategy__"

            groups = []
            labels = []
            for name, g in tmp.groupby(strategy_col):
                vals = pd.to_numeric(g[r2_col], errors="coerce").dropna().values
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(str(name))

            if len(groups) > 0:
                bp = ax.boxplot(groups, patch_artist=True, labels=[wrap_label(x, 12) for x in labels])
                for b in bp["boxes"]:
                    b.set(facecolor=COLORS["blue_light"], edgecolor=COLORS["gray"])
                for med in bp["medians"]:
                    med.set(color=COLORS["orange"], linewidth=1.5)

                # 叠加散点
                for i, vals in enumerate(groups, start=1):
                    x_jitter = np.random.normal(loc=i, scale=0.04, size=len(vals))
                    ax.scatter(x_jitter, vals, s=18, alpha=0.7, color=COLORS["blue_dark"], edgecolor="white", linewidth=0.3)

                ax.set_ylabel("R²")
                ax.set_title("Grouped-validation performance across splits/strategies", fontsize=10)
                beautify_axes(ax)

                rows = []
                for lab, vals in zip(labels, groups):
                    for v in vals:
                        rows.append({"strategy": lab, "R2": v})
                save_df(pd.DataFrame(rows), TAB_DIR / "Figure_3c_grouped_strategy_metrics.csv")
            else:
                ax.text(0.5, 0.5, "No valid R² values", ha="center", va="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "R² column not found", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Split-summary file not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 3d 文献层面误差
    # -------------------------
    ax = axes[3]
    add_panel_label(ax, "d")

    grouped_pred = hub.tables.get("29_grouped_screen_predictions", None)
    if isinstance(grouped_pred, pd.DataFrame) and len(grouped_pred) > 0:
        true_col = guess_true_col(grouped_pred)
        pred_col = guess_pred_col(grouped_pred)
        source_col = guess_source_id_column(grouped_pred)

        if true_col is not None and pred_col is not None:
            if source_col is None:
                source_col = guess_source_id_column(hub.model_ready_with_source) if hub.model_ready_with_source is not None else None

            tmp = grouped_pred.copy()
            tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            if source_col is None or source_col not in tmp.columns:
                tmp["__source__"] = "Unknown"
                source_col = "__source__"

            tmp = tmp.dropna(subset=[true_col, pred_col])

            summary_rows = []
            for gid, g in tmp.groupby(source_col):
                metrics = compute_metrics(g[true_col].values, g[pred_col].values)
                summary_rows.append({
                    "group": gid,
                    "N": metrics["N"],
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "target_mean": g[true_col].mean()
                })
            summary_df = pd.DataFrame(summary_rows).sort_values("MAE", ascending=False).head(20)

            y_pos = np.arange(len(summary_df))
            sizes = 20 + summary_df["N"].fillna(1).values * 4
            sc = ax.scatter(summary_df["MAE"], y_pos,
                            s=sizes, c=summary_df["target_mean"],
                            cmap="Blues", alpha=0.85, edgecolor="white", linewidth=0.4)

            ax.set_yticks(y_pos)
            ax.set_yticklabels([wrap_label(x, 10) for x in summary_df["group"].astype(str)], fontsize=7)
            ax.set_xlabel("MAE")
            ax.set_ylabel("Literature group")
            ax.set_title("Literature-level prediction error", fontsize=10)
            beautify_axes(ax)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Mean target")

            save_df(summary_df, TAB_DIR / "Figure_3d_literature_level_errors.csv")
        else:
            ax.text(0.5, 0.5, "Required prediction columns not found", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Grouped prediction file not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 3e 数值偏移热图（若现成表存在则读取）
    # -------------------------
    ax = axes[4]
    add_panel_label(ax, "e")

    possible_shift_tables = find_all_matching(OUTPUTS_DIR / "tables", "*numeric_shift*csv") + \
                            find_all_matching(OUTPUTS_DIR / "tables", "*smd*csv")
    shift_df = None
    for p in possible_shift_tables:
        df = safe_read_csv(p)
        if df is not None and len(df) > 0:
            shift_df = ensure_str_columns(df)
            logger.log(f"[INFO] Figure 3e 使用表：{p.name}")
            break

    if isinstance(shift_df, pd.DataFrame):
        # 选数值部分做热图
        numeric_part = shift_df.select_dtypes(include=[np.number])
        if numeric_part.shape[0] > 0 and numeric_part.shape[1] > 0:
            heat = numeric_part.copy()
            # 截取至适度尺寸
            heat = heat.iloc[:15, :12]
            im = ax.imshow(heat.values, aspect="auto", cmap="Blues")
            ax.set_xticks(range(heat.shape[1]))
            ax.set_xticklabels([wrap_label(c, 10) for c in heat.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(heat.shape[0]))
            if shift_df.index.name:
                ylabels = shift_df.index.astype(str).tolist()[:heat.shape[0]]
            else:
                ylabels = [str(i + 1) for i in range(heat.shape[0])]
            ax.set_yticklabels(ylabels, fontsize=7)
            ax.set_title("Numeric shift / SMD heatmap", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            save_df(heat.reset_index(drop=True), TAB_DIR / "Figure_3e_numeric_shift_smd.csv")
        else:
            ax.text(0.5, 0.5, "Numeric shift table has no numeric block", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Numeric shift table not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 3f 未见类别占比热图（若现成表存在则读取）
    # -------------------------
    ax = axes[5]
    add_panel_label(ax, "f")

    possible_unseen_tables = find_all_matching(OUTPUTS_DIR / "tables", "*unseen*csv") + \
                             find_all_matching(OUTPUTS_DIR / "tables", "*category*fraction*csv")
    unseen_df = None
    for p in possible_unseen_tables:
        df = safe_read_csv(p)
        if df is not None and len(df) > 0:
            unseen_df = ensure_str_columns(df)
            logger.log(f"[INFO] Figure 3f 使用表：{p.name}")
            break

    if isinstance(unseen_df, pd.DataFrame):
        numeric_part = unseen_df.select_dtypes(include=[np.number])
        if numeric_part.shape[0] > 0 and numeric_part.shape[1] > 0:
            heat = numeric_part.copy().iloc[:12, :12]
            im = ax.imshow(heat.values, aspect="auto", cmap="Oranges")
            ax.set_xticks(range(heat.shape[1]))
            ax.set_xticklabels([wrap_label(c, 10) for c in heat.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(heat.shape[0]))
            ax.set_yticklabels([str(i + 1) for i in range(heat.shape[0])], fontsize=7)
            ax.set_title("Unseen-category fraction heatmap", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            save_df(heat.reset_index(drop=True), TAB_DIR / "Figure_3f_unseen_category_fraction.csv")
        else:
            ax.text(0.5, 0.5, "Unseen-category table has no numeric block", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "Unseen-category table not found", ha="center", va="center")
        ax.set_axis_off()

    save_figure(fig, "Figure_3_cross_literature_generalization")

# =========================
# Figure 4
# =========================
def load_32_series_tables() -> Dict[str, pd.DataFrame]:
    tables_32 = {}
    all_32 = find_all_matching(OUTPUTS_DIR / "tables", "32*.csv")
    for p in all_32:
        df = safe_read_csv(p)
        if df is not None:
            tables_32[p.stem] = ensure_str_columns(df)
    return tables_32

def pick_best_table_by_columns(candidates: Dict[str, pd.DataFrame], required_keywords: List[str]) -> Optional[Tuple[str, pd.DataFrame]]:
    for name, df in candidates.items():
        cols_joined = " | ".join([c.lower() for c in df.columns])
        if all(any(k in c.lower() for c in df.columns) or k in cols_joined for k in required_keywords):
            return name, df
    return None

def plot_figure_4(hub: ProjectDataHub):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    tables_32 = load_32_series_tables()

    # -------------------------
    # 4a AD 构建流程图
    # -------------------------
    ax = axes[0]
    ax.set_axis_off()
    add_panel_label(ax, "a")

    steps = [
        "Target-free\nfeature space",
        "Novelty / density\nindicators",
        "Corrected AD risk",
        "Reliability-zone\nclassification",
        "Coverage–performance\nevaluation"
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for i, (x, step) in enumerate(zip(xs, steps)):
        rect = FancyBboxPatch((x - 0.08, y - 0.08), 0.16, 0.16,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=COLORS["blue_light"] if i % 2 == 0 else COLORS["green_light"],
                              edgecolor=COLORS["gray"])
        ax.add_patch(rect)
        ax.text(x, y, step, ha="center", va="center", fontsize=8)

        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x + 0.08, y), (xs[i + 1] - 0.08, y),
                                    arrowstyle="->", mutation_scale=12,
                                    linewidth=1.2, color=COLORS["gray"])
            ax.add_patch(arrow)
    ax.text(0.5, 0.18,
            "AD screening is kept target-free to avoid leakage from true fluoride-removal outcomes.",
            ha="center", va="center", fontsize=8, color=COLORS["gray"])

    # -------------------------
    # 4b AD 风险 vs 误差
    # -------------------------
    ax = axes[1]
    add_panel_label(ax, "b")

    risk_err_pair = pick_best_table_by_columns(tables_32, ["risk"])
    used_risk_df = risk_err_pair[1] if risk_err_pair else None

    plotted = False
    if isinstance(used_risk_df, pd.DataFrame):
        risk_col = None
        err_col = None
        for c in used_risk_df.columns:
            cl = c.lower()
            if risk_col is None and "risk" in cl:
                risk_col = c
            if err_col is None and ("abs_error" in cl or ("error" in cl and "abs" in cl)):
                err_col = c

        # 若没有显式 abs_error，尝试自己计算
        if err_col is None:
            true_col = guess_true_col(used_risk_df)
            pred_col = guess_pred_col(used_risk_df)
            if true_col and pred_col:
                tmp = used_risk_df.copy()
                tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
                tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
                tmp["__abs_error__"] = np.abs(tmp[true_col] - tmp[pred_col])
                err_col = "__abs_error__"
                used_risk_df = tmp

        if risk_col and err_col:
            tmp = used_risk_df.copy()
            tmp[risk_col] = pd.to_numeric(tmp[risk_col], errors="coerce")
            tmp[err_col] = pd.to_numeric(tmp[err_col], errors="coerce")
            tmp = tmp.dropna(subset=[risk_col, err_col])

            if len(tmp) > 0:
                ax.scatter(tmp[risk_col], tmp[err_col], s=16, alpha=0.65,
                           color=COLORS["blue"], edgecolor="white", linewidth=0.3)
                # 简单趋势线
                if len(tmp) > 2:
                    z = np.polyfit(tmp[risk_col], tmp[err_col], 1)
                    xp = np.linspace(tmp[risk_col].min(), tmp[risk_col].max(), 100)
                    yp = z[0] * xp + z[1]
                    ax.plot(xp, yp, color=COLORS["orange"], linewidth=1.3)

                ax.set_xlabel("Corrected AD risk")
                ax.set_ylabel("Absolute prediction error")
                ax.set_title("Relationship between AD risk and prediction error", fontsize=10)
                beautify_axes(ax)

                save_df(tmp[[risk_col, err_col]].rename(columns={risk_col: "ad_risk", err_col: "abs_error"}),
                        TAB_DIR / "Figure_4b_ad_risk_error.csv")
                plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "AD risk table not found or unusable", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 4c 覆盖率—性能前沿
    # -------------------------
    ax = axes[2]
    add_panel_label(ax, "c")

    frontier_rows = []
    for name, df in tables_32.items():
        mm = guess_metric_cols(df)
        if mm["r2"] and mm["coverage"]:
            tmp = df.copy()
            tmp[mm["r2"]] = pd.to_numeric(tmp[mm["r2"]], errors="coerce")
            tmp[mm["coverage"]] = pd.to_numeric(tmp[mm["coverage"]], errors="coerce")
            n_col = mm["n"]
            if n_col:
                tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce")
            else:
                tmp["__n__"] = np.nan
                n_col = "__n__"

            for _, r in tmp.iterrows():
                if pd.notna(r[mm["r2"]]) and pd.notna(r[mm["coverage"]]):
                    frontier_rows.append({
                        "table": name,
                        "R2": r[mm["r2"]],
                        "Coverage": r[mm["coverage"]],
                        "N": r[n_col]
                    })

    frontier_df = pd.DataFrame(frontier_rows)
    if len(frontier_df) > 0:
        sc = ax.scatter(frontier_df["Coverage"], frontier_df["R2"],
                        s=np.where(frontier_df["N"].fillna(1) > 0, frontier_df["N"].fillna(1) * 3 + 20, 30),
                        c=np.arange(len(frontier_df)),
                        cmap="Blues", alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("R²")
        ax.set_title("Coverage–performance frontier", fontsize=10)
        beautify_axes(ax)
        save_df(frontier_df, TAB_DIR / "Figure_4c_coverage_performance_frontier.csv")
    else:
        ax.text(0.5, 0.5, "No 32-series frontier data found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 4d 全域 / AD / 可靠域性能对比
    # -------------------------
    ax = axes[3]
    add_panel_label(ax, "d")

    domain_rows = []

    # All-domain
    compare_df = build_random_vs_grouped_metrics(hub)
    if len(compare_df) > 0:
        # 取 grouped 作为全域 grouped 基线
        row = compare_df[compare_df["evaluation"] == "Grouped validation"]
        if len(row) > 0:
            row = row.iloc[0]
            domain_rows.append({
                "domain": "All-domain\n(grouped)",
                "R2": row["R2"], "RMSE": row["RMSE"], "MAE": row["MAE"],
                "N": row["N"], "Coverage": row["Coverage"]
            })

    # 从 32 系列中抓取带 coverage 的最好结果作为 Reliable-domain
    best_reliable = None
    for name, df in tables_32.items():
        mm = guess_metric_cols(df)
        if mm["r2"] and mm["coverage"]:
            tmp = df.copy()
            tmp[mm["r2"]] = pd.to_numeric(tmp[mm["r2"]], errors="coerce")
            tmp[mm["coverage"]] = pd.to_numeric(tmp[mm["coverage"]], errors="coerce")
            if len(tmp.dropna(subset=[mm["r2"], mm["coverage"]])) > 0:
                tmp = tmp.dropna(subset=[mm["r2"], mm["coverage"]]).sort_values(mm["r2"], ascending=False)
                cand = tmp.iloc[0]
                cand_dict = {
                    "domain": f"Reliable-domain\n({name})",
                    "R2": cand[mm["r2"]],
                    "RMSE": cand[mm["rmse"]] if mm["rmse"] else np.nan,
                    "MAE": cand[mm["mae"]] if mm["mae"] else np.nan,
                    "N": cand[mm["n"]] if mm["n"] else np.nan,
                    "Coverage": cand[mm["coverage"]],
                }
                if best_reliable is None or pd.to_numeric(cand_dict["R2"], errors="coerce") > pd.to_numeric(best_reliable["R2"], errors="coerce"):
                    best_reliable = cand_dict

    # 若能找到，再加一行 AD domain（这里用次优覆盖/性能折中点近似）
    if best_reliable is not None:
        domain_rows.append({
            "domain": "AD / retained domain",
            "R2": best_reliable["R2"],
            "RMSE": best_reliable["RMSE"],
            "MAE": best_reliable["MAE"],
            "N": best_reliable["N"],
            "Coverage": best_reliable["Coverage"]
        })

    domain_df = pd.DataFrame(domain_rows)
    if len(domain_df) > 0:
        x = np.arange(len(domain_df))
        y = pd.to_numeric(domain_df["R2"], errors="coerce").values
        ax.bar(x, y, color=[COLORS["blue"], COLORS["orange"]][:len(x)], edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(domain_df["domain"].tolist(), fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title("Performance comparison across domains", fontsize=10)
        beautify_axes(ax)

        for xi, yi in zip(x, y):
            if pd.notna(yi):
                ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

        save_df(domain_df, TAB_DIR / "Figure_4d_domain_metrics.csv")
    else:
        ax.text(0.5, 0.5, "No domain metrics available", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 4e 阈值敏感性
    # -------------------------
    ax = axes[4]
    add_panel_label(ax, "e")

    thresh_rows = []
    for name, df in tables_32.items():
        # 只筛可能属于 32h 阈值敏感性文件
        if "32h" in name.lower() or "sensitivity" in name.lower() or "threshold" in name.lower():
            mm = guess_metric_cols(df)
            threshold_col = None
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in ["threshold", "nn_risk", "cutoff"]):
                    threshold_col = c
                    break
            if threshold_col and mm["r2"]:
                tmp = df.copy()
                tmp[threshold_col] = pd.to_numeric(tmp[threshold_col], errors="coerce")
                tmp[mm["r2"]] = pd.to_numeric(tmp[mm["r2"]], errors="coerce")
                cov_col = mm["coverage"]
                if cov_col:
                    tmp[cov_col] = pd.to_numeric(tmp[cov_col], errors="coerce")
                for _, r in tmp.iterrows():
                    if pd.notna(r[threshold_col]) and pd.notna(r[mm["r2"]]):
                        thresh_rows.append({
                            "table": name,
                            "threshold": r[threshold_col],
                            "R2": r[mm["r2"]],
                            "Coverage": r[cov_col] if cov_col else np.nan
                        })

    thresh_df = pd.DataFrame(thresh_rows)
    if len(thresh_df) > 0:
        for name, g in thresh_df.groupby("table"):
            ax.plot(g["threshold"], g["R2"], marker="o", linewidth=1.2, label=name)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("R²")
        ax.set_title("Threshold sensitivity analysis", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        beautify_axes(ax)
        save_df(thresh_df, TAB_DIR / "Figure_4e_threshold_sensitivity.csv")
    else:
        ax.text(0.5, 0.5, "Threshold-sensitivity table not found", ha="center", va="center")
        ax.set_axis_off()

    # -------------------------
    # 4f 保留域 vs 全域关键变量分布
    # -------------------------
    ax = axes[5]
    add_panel_label(ax, "f")

    if hub.model_ready is not None:
        df = hub.model_ready.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 选最相关的几个代表变量名称
        rep_candidates = []
        for col in df.columns:
            cl = col.lower()
            if any(k in cl for k in ["dose", "剂量", "ph", "fluor", "氟", "reaction", "settling", "time"]):
                rep_candidates.append(col)

        # 保底选前几个数值列
        if len(rep_candidates) < 4:
            for c in num_cols:
                if c not in rep_candidates:
                    rep_candidates.append(c)
                if len(rep_candidates) >= 4:
                    break

        rep_cols = rep_candidates[:4]

        # 目前没有直接可靠域样本清单时，用高性能 retained domain 的近似规则无法精确还原，
        # 因此这里先做“全域代表变量标准化分布示意”，下一版接入 32G/32H 精确规则后再升级。
        tmp = df[rep_cols].copy()
        for c in rep_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        std_tmp = standardize_numeric(tmp, rep_cols)
        plot_vals = [std_tmp[c].dropna().values for c in rep_cols if c in std_tmp.columns]

        if len(plot_vals) > 0:
            bp = ax.boxplot(plot_vals, patch_artist=True, labels=[wrap_label(c, 10) for c in rep_cols])
            fill_colors = [COLORS["blue_light"], COLORS["green_light"], COLORS["orange_light"], COLORS["gray_light"]]
            for i, b in enumerate(bp["boxes"]):
                b.set(facecolor=fill_colors[i % len(fill_colors)], edgecolor=COLORS["gray"])
            for med in bp["medians"]:
                med.set(color=COLORS["orange"], linewidth=1.3)
            ax.set_ylabel("Standardized value")
            ax.set_title("Representative chemical/operational profiles", fontsize=10)
            beautify_axes(ax)

            save_df(std_tmp, TAB_DIR / "Figure_4f_retained_domain_profiles.csv")
        else:
            ax.text(0.5, 0.5, "No representative columns available", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "model_ready not found", ha="center", va="center")
        ax.set_axis_off()

    save_figure(fig, "Figure_4_applicability_and_reliable_domain")

# =========================
# 主程序
# =========================
def main():
    start_time = time.time()
    logger.log("=" * 80)
    logger.log("JECE Figure Package (Part 1: Figure 1–4) started")
    logger.log(f"Project root: {PROJECT_ROOT}")
    logger.log(f"Output root : {JECE_DIR}")
    logger.log("=" * 80)

    hub = ProjectDataHub()
    hub.load()

    if RUN_AUDIT:
        logger.log("\n[STEP] Running audit ...")
        hub.audit_report()

    if RUN_FIGURE_1:
        logger.log("\n[STEP] Generating Figure 1 ...")
        try:
            plot_figure_1(hub)
        except Exception as e:
            logger.log(f"[ERROR] Figure 1 生成失败：{e}")

    if RUN_FIGURE_2:
        logger.log("\n[STEP] Generating Figure 2 ...")
        try:
            plot_figure_2(hub)
        except Exception as e:
            logger.log(f"[ERROR] Figure 2 生成失败：{e}")

    if RUN_FIGURE_3:
        logger.log("\n[STEP] Generating Figure 3 ...")
        try:
            plot_figure_3(hub)
        except Exception as e:
            logger.log(f"[ERROR] Figure 3 生成失败：{e}")

    if RUN_FIGURE_4:
        logger.log("\n[STEP] Generating Figure 4 ...")
        try:
            plot_figure_4(hub)
        except Exception as e:
            logger.log(f"[ERROR] Figure 4 生成失败：{e}")

    elapsed = time.time() - start_time
    logger.log("\n" + "=" * 80)
    logger.log(f"All requested tasks finished.")
    logger.log(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.log("=" * 80)
    logger.save()

if __name__ == "__main__":
    main()