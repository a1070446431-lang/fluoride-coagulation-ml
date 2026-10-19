# -*- coding: utf-8 -*-
"""
28_grouped_split_feasibility.py

Step 28: Feasibility audit for grouped split / leave-one-literature-out validation.

Purpose
-------
Check whether the current fluoride coagulation ML dataset contains a reliable
grouping variable for literature-level validation.

This script does NOT retrain the ML model.
It only answers:
    1. Is there a DOI / paper / source / study / reference column?
    2. Is the grouping variable present in model_ready_v1.csv?
    3. If not, is it present in cleaned_full_with_flags.csv or raw Excel?
    4. Can it be safely aligned to model_ready_v1.csv by row order?
    5. Is grouped split / leave-one-literature-out feasible?

Inputs
------
data/processed/model_ready_v1.csv
data/processed/cleaned_full_with_flags.csv
data/raw/*.xlsx

Outputs
-------
outputs/tables/28_group_column_candidates.csv
outputs/tables/28_best_group_distribution.csv
outputs/tables/28_group_split_feasibility.csv
outputs/tables/28_recommended_group_labels_for_model_ready.csv
outputs/tables/28_grouped_split_decision.txt

outputs/figures/80_group_size_distribution.png
outputs/figures/80_group_size_distribution.pdf
"""

from pathlib import Path
import re
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================
# 1. Path detection
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    candidates = [
        current,
        current.parent,
        current.parent.parent,
    ]

    for p in candidates:
        if (p / "data").exists() and (p / "outputs").exists():
            return p

    if current.name.lower() == "src":
        return current.parent

    raise FileNotFoundError("Cannot locate project root.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

COLOR_BAR = "#63B7B2"
COLOR_GRID = "#E6E6E6"
COLOR_RED = "#FF0000"
COLOR_GRAY = "#666666"


# ============================================================
# 3. Data reading
# ============================================================

def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read CSV file: {path}")


def find_model_ready_path() -> Path:
    candidates = [
        PROCESSED_DIR / "model_ready_v1.csv",
        PROCESSED_DIR / "model_ready-v1.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find model_ready_v1.csv or model_ready-v1.csv under data/processed."
    )


def load_available_tables():
    """
    Load model_ready, cleaned table, and raw Excel sheets if present.
    """
    tables = []

    model_ready_path = find_model_ready_path()
    model_ready = read_csv_smart(model_ready_path)

    tables.append({
        "table_name": "model_ready",
        "path": str(model_ready_path),
        "df": model_ready,
        "is_model_ready": True,
        "row_aligned_with_model_ready": True,
    })

    cleaned_path = PROCESSED_DIR / "cleaned_full_with_flags.csv"
    if cleaned_path.exists():
        cleaned = read_csv_smart(cleaned_path)
        tables.append({
            "table_name": "cleaned_full_with_flags",
            "path": str(cleaned_path),
            "df": cleaned,
            "is_model_ready": False,
            "row_aligned_with_model_ready": len(cleaned) == len(model_ready),
        })

    if RAW_DIR.exists():
        for xlsx_path in sorted(RAW_DIR.glob("*.xlsx")):
            try:
                xls = pd.ExcelFile(xlsx_path)
                for sheet in xls.sheet_names:
                    try:
                        df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet)
                        if len(df_sheet) == 0:
                            continue

                        tables.append({
                            "table_name": f"raw_excel::{xlsx_path.name}::{sheet}",
                            "path": str(xlsx_path),
                            "df": df_sheet,
                            "is_model_ready": False,
                            "row_aligned_with_model_ready": len(df_sheet) == len(model_ready),
                        })
                    except Exception as e:
                        log(f"[WARN] Failed to read sheet {sheet} in {xlsx_path.name}: {e}")
            except Exception as e:
                log(f"[WARN] Failed to open raw Excel file {xlsx_path}: {e}")

    return tables, model_ready


# ============================================================
# 4. Candidate column detection
# ============================================================

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("％", "%")
    s = s.lower().strip()
    s = re.sub(r"\s+", "", s)
    return s


STRONG_GROUP_KEYWORDS = [
    "doi",
    "paper_id",
    "paperid",
    "source_id",
    "sourceid",
    "study_id",
    "studyid",
    "reference",
    "ref",
    "citation",
    "publication",
    "article",
    "title",
    "journal",
    "literature",
    "文献",
    "论文",
    "文章",
    "来源",
    "出处",
    "题目",
    "标题",
    "期刊",
    "作者",
    "参考",
    "编号",
    "来源编号",
    "文献编号",
]

WEAK_GROUP_KEYWORDS = [
    "year",
    "年份",
    "年代",
]

TARGET_LEAKAGE_KEYWORDS = [
    "去除率",
    "removal",
    "removalrate",
    "出水",
    "effluent",
    "剩余",
    "残余",
    "remaining",
    "residual",
    "target",
    "label",
    "预测",
    "pred",
    "达标",
    "效果",
    "最终",
    "after",
    "post",
    "response",
    "y值",
]


def classify_candidate_column(col: str) -> tuple[bool, str, str]:
    """
    Return:
        is_candidate, strength, reason
    """
    n = normalize_text(col)

    for kw in TARGET_LEAKAGE_KEYWORDS:
        if normalize_text(kw) in n:
            return False, "excluded", "target_or_post-treatment_like_column"

    for kw in STRONG_GROUP_KEYWORDS:
        if normalize_text(kw) in n:
            return True, "strong", f"matched_keyword={kw}"

    for kw in WEAK_GROUP_KEYWORDS:
        if normalize_text(kw) in n:
            return True, "weak", f"matched_keyword={kw}"

    return False, "none", ""


def is_probably_row_id(series: pd.Series) -> bool:
    s = series.dropna()

    if len(s) < 5:
        return False

    # Numeric consecutive row ID.
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.95:
        vals = numeric.dropna().astype(float).values
        if len(np.unique(vals)) / len(vals) > 0.95:
            sorted_vals = np.sort(np.unique(vals))
            if len(sorted_vals) > 5:
                diffs = np.diff(sorted_vals)
                if np.nanmedian(diffs) == 1:
                    return True

    # Nearly every row has a unique string.
    unique_ratio = s.astype(str).nunique() / len(s)
    if unique_ratio > 0.95:
        return True

    return False


def audit_candidate_column(table_name: str, path: str, df: pd.DataFrame,
                           col: str, strength: str, reason: str,
                           model_ready_n: int,
                           row_aligned: bool,
                           is_model_ready: bool) -> dict:
    s = df[col]
    n_rows = len(df)
    missing_n = int(s.isna().sum())
    missing_ratio = missing_n / max(n_rows, 1)

    s_text = s.astype("string")
    s_nonmissing = s_text.dropna().str.strip()
    s_nonmissing = s_nonmissing[s_nonmissing != ""]

    n_nonmissing = len(s_nonmissing)
    n_unique = int(s_nonmissing.nunique()) if n_nonmissing > 0 else 0

    if n_nonmissing > 0:
        counts = s_nonmissing.value_counts()
        largest_group_size = int(counts.iloc[0])
        median_group_size = float(counts.median())
        mean_group_size = float(counts.mean())
        n_groups_ge_2 = int((counts >= 2).sum())
        n_groups_ge_5 = int((counts >= 5).sum())
        n_groups_ge_10 = int((counts >= 10).sum())
        largest_group_ratio = largest_group_size / n_nonmissing
        unique_ratio = n_unique / n_nonmissing
        top_values = "; ".join([f"{idx}:{val}" for idx, val in counts.head(5).items()])
    else:
        largest_group_size = 0
        median_group_size = np.nan
        mean_group_size = np.nan
        n_groups_ge_2 = 0
        n_groups_ge_5 = 0
        n_groups_ge_10 = 0
        largest_group_ratio = np.nan
        unique_ratio = np.nan
        top_values = ""

    row_id_like = is_probably_row_id(s)

    # Suitability scoring.
    score = 0.0

    if is_model_ready:
        score += 30
    elif row_aligned:
        score += 15
    else:
        score -= 40

    if strength == "strong":
        score += 45
    elif strength == "weak":
        score += 10

    if n_unique >= 5:
        score += 20
    elif n_unique >= 3:
        score += 8
    else:
        score -= 30

    if n_groups_ge_5 >= 5:
        score += 20
    elif n_groups_ge_2 >= 5:
        score += 10
    else:
        score -= 15

    if missing_ratio > 0.5:
        score -= 30
    elif missing_ratio > 0.2:
        score -= 10

    if not np.isnan(largest_group_ratio):
        if largest_group_ratio > 0.7:
            score -= 30
        elif largest_group_ratio > 0.5:
            score -= 15

    if row_id_like:
        score -= 50

    if strength == "weak":
        score -= 15

    if unique_ratio > 0.90:
        score -= 35

    if n_rows != model_ready_n:
        score -= 15

    if score >= 70:
        decision = "highly_suitable"
    elif score >= 45:
        decision = "usable_with_caution"
    elif score >= 20:
        decision = "weak_candidate"
    else:
        decision = "not_suitable"

    return {
        "table_name": table_name,
        "path": path,
        "column": col,
        "strength": strength,
        "reason": reason,
        "is_model_ready_table": is_model_ready,
        "row_aligned_with_model_ready": row_aligned,
        "n_rows": n_rows,
        "model_ready_n": model_ready_n,
        "missing_n": missing_n,
        "missing_ratio": missing_ratio,
        "n_nonmissing": n_nonmissing,
        "n_unique_groups": n_unique,
        "unique_ratio": unique_ratio,
        "largest_group_size": largest_group_size,
        "largest_group_ratio": largest_group_ratio,
        "median_group_size": median_group_size,
        "mean_group_size": mean_group_size,
        "n_groups_ge_2": n_groups_ge_2,
        "n_groups_ge_5": n_groups_ge_5,
        "n_groups_ge_10": n_groups_ge_10,
        "row_id_like": row_id_like,
        "score": score,
        "decision": decision,
        "top_values": top_values,
    }


def audit_all_tables(tables, model_ready_n: int) -> pd.DataFrame:
    rows = []

    for item in tables:
        table_name = item["table_name"]
        path = item["path"]
        df = item["df"]
        is_model_ready = item["is_model_ready"]
        row_aligned = item["row_aligned_with_model_ready"]

        log(f"[SCAN] {table_name}: shape={df.shape}")

        for col in df.columns:
            is_candidate, strength, reason = classify_candidate_column(str(col))

            if not is_candidate:
                continue

            row = audit_candidate_column(
                table_name=table_name,
                path=path,
                df=df,
                col=col,
                strength=strength,
                reason=reason,
                model_ready_n=model_ready_n,
                row_aligned=row_aligned,
                is_model_ready=is_model_ready,
            )

            rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(
        ["score", "is_model_ready_table", "row_aligned_with_model_ready"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return out


# ============================================================
# 5. Best group extraction and split feasibility
# ============================================================

def get_best_candidate(candidates: pd.DataFrame):
    if candidates.empty:
        return None

    usable = candidates[candidates["decision"].isin(["highly_suitable", "usable_with_caution", "weak_candidate"])].copy()

    if usable.empty:
        return None

    # Prefer model_ready, then row-aligned cleaned table.
    usable = usable.sort_values(
        ["score", "is_model_ready_table", "row_aligned_with_model_ready"],
        ascending=[False, False, False],
    )

    return usable.iloc[0].to_dict()


def find_table_item(tables, table_name: str):
    for item in tables:
        if item["table_name"] == table_name:
            return item
    return None


def extract_group_series(tables, best_candidate: dict, model_ready_n: int) -> pd.Series:
    item = find_table_item(tables, best_candidate["table_name"])

    if item is None:
        raise RuntimeError("Best candidate table cannot be found.")

    df = item["df"]
    col = best_candidate["column"]

    if len(df) != model_ready_n:
        raise RuntimeError(
            "Best candidate table is not row-aligned with model_ready_v1.csv. "
            "Do not use it for grouped validation without explicit row mapping."
        )

    group = df[col].astype("string").fillna("MISSING_GROUP").str.strip()
    group = group.replace("", "MISSING_GROUP")

    return group.reset_index(drop=True)


def group_distribution(group: pd.Series) -> pd.DataFrame:
    counts = group.value_counts(dropna=False)

    out = pd.DataFrame({
        "Group": counts.index.astype(str),
        "n_rows": counts.values,
    })

    out["rank"] = np.arange(1, len(out) + 1)
    out["row_fraction"] = out["n_rows"] / out["n_rows"].sum()

    return out


def simulate_group_splits(group: pd.Series, test_size: float = 0.2, n_splits: int = 5) -> pd.DataFrame:
    unique_groups = group.nunique(dropna=False)

    if unique_groups < 2:
        return pd.DataFrame()

    X_dummy = np.zeros((len(group), 1))
    y_dummy = np.zeros(len(group))

    gss = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=42,
    )

    rows = []

    for split_id, (train_idx, test_idx) in enumerate(gss.split(X_dummy, y_dummy, groups=group), start=1):
        train_groups = set(group.iloc[train_idx].astype(str))
        test_groups = set(group.iloc[test_idx].astype(str))
        overlap = train_groups.intersection(test_groups)

        rows.append({
            "split_id": split_id,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_row_fraction": len(train_idx) / len(group),
            "test_row_fraction": len(test_idx) / len(group),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap_n": len(overlap),
            "no_group_overlap": len(overlap) == 0,
            "largest_test_group_rows": int(group.iloc[test_idx].value_counts().max()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Figure
# ============================================================

def plot_group_size_distribution(dist_df: pd.DataFrame, best_col: str):
    if dist_df.empty:
        return

    top = dist_df.head(30).copy()
    top = top.sort_values("n_rows", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.22 * len(top))))

    y = np.arange(len(top))

    bars = ax.barh(
        y,
        top["n_rows"],
        color=COLOR_BAR,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.88,
    )

    labels = top["Group"].astype(str).str.slice(0, 55)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of rows")
    ax.set_title(f"Top group-size distribution for candidate column: {best_col}")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6)

    for bar, val in zip(bars, top["n_rows"]):
        ax.text(
            val + max(top["n_rows"].max() * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            str(int(val)),
            va="center",
            ha="left",
            fontsize=7,
        )

    fig.tight_layout()

    png_path = FIG_DIR / "80_group_size_distribution.png"
    pdf_path = FIG_DIR / "80_group_size_distribution.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    log(f"[FIG] {png_path}")
    log(f"[FIG] {pdf_path}")


# ============================================================
# 7. Decision report
# ============================================================

def write_decision_report(
    candidates: pd.DataFrame,
    best_candidate: dict | None,
    dist_df: pd.DataFrame,
    split_df: pd.DataFrame,
    model_ready_shape,
):
    path = TABLE_DIR / "28_grouped_split_decision.txt"

    lines = []
    lines.append("=" * 100)
    lines.append("Step 28 grouped split / leave-one-literature-out feasibility report")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"model_ready shape: {model_ready_shape}")
    lines.append("")

    if candidates.empty:
        lines.append("No candidate literature/source grouping column was found.")
        lines.append("")
        lines.append("Conclusion:")
        lines.append("Grouped split / leave-one-literature-out validation is currently NOT feasible.")
        lines.append("You need to add a source column, preferably DOI, paper_id, reference_id, or literature title,")
        lines.append("to model_ready_v1.csv or provide a reliable row mapping from cleaned/raw data to model_ready_v1.csv.")
        path.write_text("\n".join(lines), encoding="utf-8")
        log(f"[TEXT] {path}")
        return

    lines.append("Candidate grouping columns, sorted by suitability:")
    display_cols = [
        "table_name",
        "column",
        "strength",
        "n_rows",
        "missing_ratio",
        "n_unique_groups",
        "largest_group_size",
        "largest_group_ratio",
        "median_group_size",
        "n_groups_ge_5",
        "row_id_like",
        "score",
        "decision",
    ]

    lines.append(candidates[display_cols].head(20).to_string(index=False))
    lines.append("")

    if best_candidate is None:
        lines.append("No usable grouping column was found.")
        lines.append("")
        lines.append("Conclusion:")
        lines.append("Grouped validation is not defensible yet.")
        lines.append("Reason: candidate columns are missing, too weak, row-id-like, or not alignable with model_ready_v1.csv.")
        path.write_text("\n".join(lines), encoding="utf-8")
        log(f"[TEXT] {path}")
        return

    lines.append("Recommended candidate:")
    for k in [
        "table_name",
        "column",
        "strength",
        "reason",
        "is_model_ready_table",
        "row_aligned_with_model_ready",
        "n_rows",
        "n_unique_groups",
        "largest_group_size",
        "largest_group_ratio",
        "median_group_size",
        "n_groups_ge_5",
        "score",
        "decision",
    ]:
        lines.append(f"{k}: {best_candidate.get(k)}")

    lines.append("")

    if dist_df is not None and not dist_df.empty:
        n_groups = len(dist_df)
        n_rows = int(dist_df["n_rows"].sum())
        largest = int(dist_df["n_rows"].max())
        largest_ratio = largest / n_rows
        median_size = float(dist_df["n_rows"].median())
        n_ge_5 = int((dist_df["n_rows"] >= 5).sum())
        n_ge_10 = int((dist_df["n_rows"] >= 10).sum())

        lines.append("Best-group distribution summary:")
        lines.append(f"Total rows: {n_rows}")
        lines.append(f"Total groups: {n_groups}")
        lines.append(f"Largest group rows: {largest}")
        lines.append(f"Largest group ratio: {largest_ratio:.3f}")
        lines.append(f"Median group size: {median_size:.2f}")
        lines.append(f"Groups with >=5 rows: {n_ge_5}")
        lines.append(f"Groups with >=10 rows: {n_ge_10}")
        lines.append("")

        lines.append("Top 20 groups:")
        lines.append(dist_df.head(20).to_string(index=False))
        lines.append("")

    if split_df is not None and not split_df.empty:
        lines.append("Simulated GroupShuffleSplit feasibility:")
        lines.append(split_df.to_string(index=False))
        lines.append("")

    lines.append("Decision:")
    if best_candidate["decision"] == "highly_suitable" and best_candidate["row_aligned_with_model_ready"]:
        lines.append("Grouped split is feasible.")
        lines.append("Next step: run grouped holdout validation using this group column.")
        lines.append("Leave-one-literature-out may also be feasible if the number of groups and group sizes are not too imbalanced.")
    elif best_candidate["row_aligned_with_model_ready"]:
        lines.append("Grouped split is possible but should be treated cautiously.")
        lines.append("Next step: inspect whether the recommended group column truly represents publication/study source.")
        lines.append("Do NOT claim external generalization until grouped validation is completed.")
    else:
        lines.append("A possible group column exists, but it is not safely aligned with model_ready_v1.csv.")
        lines.append("Next step: propagate this source column into model_ready_v1.csv with verified row mapping before grouped validation.")

    lines.append("")
    lines.append("Important interpretation:")
    lines.append("Random train/test split performance can remain valid as random-split performance,")
    lines.append("but it cannot prove generalization to unseen literature sources.")
    lines.append("Grouped validation is needed to address possible literature-level leakage.")
    lines.append("=" * 100)

    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[TEXT] {path}")


# ============================================================
# 8. Main
# ============================================================

def main():
    log("\n" + "=" * 100)
    log("Step 28: grouped split feasibility audit started")
    log("=" * 100)
    log(f"PROJECT_ROOT  = {PROJECT_ROOT}")
    log(f"PROCESSED_DIR = {PROCESSED_DIR}")
    log(f"RAW_DIR       = {RAW_DIR}")
    log(f"TABLE_DIR     = {TABLE_DIR}")
    log(f"FIG_DIR       = {FIG_DIR}")
    log("=" * 100)

    tables, model_ready = load_available_tables()
    model_ready_n = len(model_ready)

    log(f"\n[OK] model_ready shape = {model_ready.shape}")
    log("\nLoaded tables:")
    for item in tables:
        log(
            f"  - {item['table_name']} | shape={item['df'].shape} | "
            f"row_aligned={item['row_aligned_with_model_ready']} | path={item['path']}"
        )

    log("\nScanning candidate group/source columns...")
    candidates = audit_all_tables(tables, model_ready_n=model_ready_n)

    candidates_path = TABLE_DIR / "28_group_column_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {candidates_path}")

    if candidates.empty:
        log("[RESULT] No candidate group columns found.")
        write_decision_report(
            candidates=candidates,
            best_candidate=None,
            dist_df=pd.DataFrame(),
            split_df=pd.DataFrame(),
            model_ready_shape=model_ready.shape,
        )
        return

    log("\n=== Candidate group columns ===")
    display_cols = [
        "table_name",
        "column",
        "strength",
        "n_rows",
        "missing_ratio",
        "n_unique_groups",
        "largest_group_size",
        "largest_group_ratio",
        "median_group_size",
        "n_groups_ge_5",
        "row_id_like",
        "score",
        "decision",
    ]
    print(candidates[display_cols].head(30).to_string(index=False))

    best_candidate = get_best_candidate(candidates)

    if best_candidate is None:
        log("[RESULT] No usable group candidate.")
        write_decision_report(
            candidates=candidates,
            best_candidate=None,
            dist_df=pd.DataFrame(),
            split_df=pd.DataFrame(),
            model_ready_shape=model_ready.shape,
        )
        return

    log("\n=== Recommended group candidate ===")
    for k, v in best_candidate.items():
        if k in [
            "table_name",
            "column",
            "strength",
            "reason",
            "is_model_ready_table",
            "row_aligned_with_model_ready",
            "n_rows",
            "n_unique_groups",
            "largest_group_size",
            "largest_group_ratio",
            "median_group_size",
            "n_groups_ge_5",
            "score",
            "decision",
        ]:
            log(f"{k}: {v}")

    # Extract group labels only if row-aligned.
    if not bool(best_candidate["row_aligned_with_model_ready"]):
        log("[WARN] Best group column is not row-aligned with model_ready_v1.csv.")
        log("[WARN] Grouped validation cannot be run safely until row mapping is fixed.")

        write_decision_report(
            candidates=candidates,
            best_candidate=best_candidate,
            dist_df=pd.DataFrame(),
            split_df=pd.DataFrame(),
            model_ready_shape=model_ready.shape,
        )
        return

    group = extract_group_series(tables, best_candidate, model_ready_n=model_ready_n)

    group_labels_path = TABLE_DIR / "28_recommended_group_labels_for_model_ready.csv"
    pd.DataFrame({
        "row_index": np.arange(len(group)),
        "recommended_group": group,
    }).to_csv(group_labels_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {group_labels_path}")

    dist_df = group_distribution(group)

    dist_path = TABLE_DIR / "28_best_group_distribution.csv"
    dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {dist_path}")

    split_df = simulate_group_splits(group, test_size=0.2, n_splits=5)

    split_path = TABLE_DIR / "28_group_split_feasibility.csv"
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {split_path}")

    plot_group_size_distribution(dist_df, str(best_candidate["column"]))

    write_decision_report(
        candidates=candidates,
        best_candidate=best_candidate,
        dist_df=dist_df,
        split_df=split_df,
        model_ready_shape=model_ready.shape,
    )

    log("\n" + "=" * 100)
    log("[DONE] Step 28 grouped split feasibility audit completed.")
    log("=" * 100)


if __name__ == "__main__":
    main()