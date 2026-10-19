# -*- coding: utf-8 -*-
"""
32f_reconstruct_and_audit_selective_ad_policy.py

Purpose
-------
Fix and re-audit Step 32D/32E selective applicability-domain (AD) policy results.

Why this script is necessary
----------------------------
Step 32E showed that the high-R2 policy summary was NOT reproduced:
- the policy summary reported high retained R2;
- the recomputed result used an EMPTY_RULE and therefore kept all rows in the
  filtered method/strategy subset.

This script performs a stricter reconstruction:

1. Searches all relevant Step 32B/32C/32D policy-summary / policy-mapping files.
2. Combines information across files if needed:
   - e.g., Strategy from 32d_high_r2_policy_summary.csv
   - Rule from 32d_est_replot_policy_id_mapping.csv or report text
3. Recomputes every candidate policy from the prediction table.
4. Audits whether the selected policy is:
   - target-free,
   - correctly parsed,
   - internally reproducible,
   - sufficiently stable across grouped splits,
   - not dominated by one literature group.
5. Exports EST-like figures and tables.

Important strictness
--------------------
- This script DOES NOT retrain models.
- This script DOES NOT select policies by test residuals.
- This script only reconstructs and audits target-free policy rules produced
  by previous 32C/32D outputs.
- If the policy cannot be reproduced, it says so directly.

Put this file under:
    fluoride_coagulation_ml_project/src/32f_reconstruct_and_audit_selective_ad_policy.py

Run in PyCharm:
    right click -> Run '32f_reconstruct_and_audit_selective_ad_policy'
"""

from __future__ import annotations

import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. Settings
# =============================================================================

SCRIPT_NAME = "32f_reconstruct_and_audit_selective_ad_policy.py"
START_TIME = time.time()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.linewidth": 1.05,
    "axes.edgecolor": "#222222",
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

COLOR_BLUE = "#4E79A7"
COLOR_GREEN = "#6FA77B"
COLOR_ORANGE = "#D9A051"
COLOR_RED = "#C44E52"
COLOR_GRAY = "#BDBDBD"
COLOR_LIGHT = "#D8D8D8"
GRID_COLOR = "#E6E6E6"


# =============================================================================
# 1. Path utilities
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def find_project_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve().parent
    start = Path(start).resolve()

    for p in [start] + list(start.parents):
        if (p / "outputs").exists() and ((p / "src").exists() or (p / "data").exists()):
            return p

    if start.name.lower() == "src":
        return start.parent
    return start


PROJECT_ROOT = find_project_root()
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def existing_paths(patterns: List[str], folders: Optional[List[Path]] = None) -> List[Path]:
    if folders is None:
        folders = [TABLE_DIR, REPORT_DIR, PROJECT_ROOT]
    out: List[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        for pat in patterns:
            out.extend(sorted(folder.glob(pat)))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen and p.exists():
            uniq.append(p)
            seen.add(rp)
    return uniq


def first_existing(patterns: List[str], required: bool = True) -> Optional[Path]:
    paths = existing_paths(patterns)
    if paths:
        return paths[0]
    if required:
        raise FileNotFoundError("Missing required file. Tried patterns:\n" + "\n".join(patterns))
    return None


# =============================================================================
# 2. General utilities
# =============================================================================

def norm_text(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def clean_string(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and not np.isfinite(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s


def read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    log(f"[TABLE] Loaded {path} | shape={df.shape}")
    return df


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[TABLE] {path} | shape={df.shape}")
    return path


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {norm_text(c): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        nc = norm_text(cand)
        if nc in norm_map:
            return norm_map[nc]

    for cand in candidates:
        nc = norm_text(cand)
        if not nc:
            continue
        for c in cols:
            if nc in norm_text(c):
                return c

    if required:
        raise KeyError(f"Cannot find required column from candidates={candidates}\nAvailable columns={cols}")
    return None


def find_metric_col(df: pd.DataFrame, metric: str, required: bool = False) -> Optional[str]:
    metric_n = norm_text(metric)
    preferred = [
        f"Retained_{metric}", f"retained_{metric}", f"Retained {metric}",
        f"{metric}_retained", metric,
        f"All_{metric}", f"Mean_{metric}",
    ]
    col = find_col(df, preferred, required=False)
    if col:
        return col

    # Avoid choosing unrelated columns like policy_id.
    for c in df.columns:
        nc = norm_text(c)
        if metric_n == nc or nc.endswith(metric_n) or (metric_n in nc and "delta" not in nc and "rank" not in nc):
            return c

    if required:
        raise KeyError(f"Cannot find metric column: {metric}")
    return None


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_float_array(x: object) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    return arr


def finite_mask(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(p)


def r2_np(y: np.ndarray, p: np.ndarray) -> float:
    m = finite_mask(y, p)
    if m.sum() < 2:
        return np.nan
    yy = y[m]
    pp = p[m]
    denom = np.sum((yy - yy.mean()) ** 2)
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


def rmse_np(y: np.ndarray, p: np.ndarray) -> float:
    m = finite_mask(y, p)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def mae_np(y: np.ndarray, p: np.ndarray) -> float:
    m = finite_mask(y, p)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y[m] - p[m])))


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = finite_mask(y, p)
    return {
        "R2": r2_np(y, p),
        "RMSE": rmse_np(y, p),
        "MAE": mae_np(y, p),
        "N": int(m.sum()),
    }


def nearly_equal(a: object, b: object, tol: float) -> bool:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return False
    if not np.isfinite(aa) or not np.isfinite(bb):
        return False
    return abs(aa - bb) <= tol


# =============================================================================
# 3. Plot utilities
# =============================================================================

def est_ax(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis in {"both", "y"}:
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    if grid_axis in {"both", "x"}:
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def annotate_bars(ax: plt.Axes, fmt: str, fontsize: float = 9.5) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        x = patch.get_x() + patch.get_width() / 2
        if h >= 0:
            y = h + 0.015 * yr
            va = "bottom"
        else:
            y = h - 0.025 * yr
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


def save_fig(fig: plt.Figure, filename: str) -> None:
    png = FIG_DIR / filename
    pdf = FIG_DIR / filename.replace(".png", ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[FIG] {png}")
    log(f"[FIG] {pdf}")


# =============================================================================
# 4. Policy candidate collection
# =============================================================================

RULE_LIKE_COLUMNS = [
    "Rule", "Policy_Rule", "Policy Rule", "Rule_Text", "Selection_Rule",
    "Policy_Text", "Subdomain_Rule", "Retained_Rule", "Condition", "Mask_Rule",
]
POLICY_ID_COLUMNS = [
    "Display_Policy_ID", "Policy_ID", "Policy ID", "ID", "PolicyID",
]
METHOD_COLUMNS = [
    "Method", "Model", "Model_Name", "Estimator", "Base_Model", "Selected_Method",
]
STRATEGY_COLUMNS = [
    "Strategy", "Validation_Strategy", "Split_Strategy", "Grouped_Strategy",
]
COVERAGE_COLUMNS = [
    "Coverage", "Retained_Coverage", "Coverage_Retained", "coverage",
]


def choose_rule_col(df: pd.DataFrame) -> Optional[str]:
    col = find_col(df, RULE_LIKE_COLUMNS, required=False)
    if col is not None:
        return col

    for c in df.columns:
        nc = norm_text(c)
        if ("rule" in nc or "condition" in nc or "mask" in nc) and "id" not in nc and "rank" not in nc:
            return c

    # Last resort: a column named Policy may contain the actual rule, but avoid policy ID.
    for c in df.columns:
        nc = norm_text(c)
        if nc in {"policy", "policytext"}:
            return c
    return None


def row_to_candidate(row: pd.Series, source_path: Path, source_type: str) -> Dict[str, object]:
    idx = row.name

    def val(candidates: List[str]) -> str:
        for c in candidates:
            if c in row.index:
                return clean_string(row[c])
        nmap = {norm_text(k): k for k in row.index}
        for c in candidates:
            nc = norm_text(c)
            if nc in nmap:
                return clean_string(row[nmap[nc]])
        return ""

    policy_id = val(POLICY_ID_COLUMNS)
    method = val(METHOD_COLUMNS)
    strategy = val(STRATEGY_COLUMNS)

    # Rule column by direct scan.
    rule = ""
    for c in row.index:
        if c in RULE_LIKE_COLUMNS or (("rule" in norm_text(c) or "condition" in norm_text(c)) and "id" not in norm_text(c)):
            candidate_rule = clean_string(row[c])
            if candidate_rule:
                rule = candidate_rule
                break

    # Avoid accidentally treating pure ID as rule.
    if rule and norm_text(rule).startswith("p") and len(rule) <= 6:
        rule = ""

    def metric(metric_name: str) -> float:
        for c in row.index:
            nc = norm_text(c)
            mn = norm_text(metric_name)
            if nc in {mn, "retained" + mn, mn + "retained", "mean" + mn}:
                try:
                    return float(row[c])
                except Exception:
                    return np.nan
        # substring fallback
        for c in row.index:
            nc = norm_text(c)
            mn = norm_text(metric_name)
            if mn in nc and "rank" not in nc and "delta" not in nc:
                try:
                    return float(row[c])
                except Exception:
                    continue
        return np.nan

    return {
        "Source_File": str(source_path),
        "Source_Type": source_type,
        "Source_Row": idx,
        "Policy_ID": policy_id,
        "Method": method,
        "Strategy": strategy,
        "Rule": rule,
        "Summary_R2": metric("R2"),
        "Summary_RMSE": metric("RMSE"),
        "Summary_MAE": metric("MAE"),
        "Summary_Coverage": metric("Coverage"),
    }


def collect_candidates_from_csvs() -> pd.DataFrame:
    patterns = [
        "32d*.csv",
        "32c2*.csv",
        "32c*.csv",
        "32b*rule*.csv",
        "32b*policy*.csv",
        "32b*recommendation*.csv",
    ]
    paths = existing_paths(patterns, folders=[TABLE_DIR])

    # Skip huge prediction/detail files.
    skip_words = [
        "prediction", "predictions", "retained_predictions", "retained_points",
        "best_policy_points", "audit", "sample_scores", "paper_summary",
        "performance_by", "feature_audit", "error_source",
    ]

    candidates: List[Dict[str, object]] = []
    for path in paths:
        name_n = norm_text(path.name)
        if any(w in name_n for w in skip_words):
            continue

        try:
            df = read_csv_safely(path)
        except Exception as exc:
            log(f"[WARN] Cannot read {path}: {exc}")
            continue

        if df.empty:
            continue

        # Keep only tables that look policy-like or metric-like.
        has_metric = any(find_metric_col(df, m, required=False) is not None for m in ["R2", "RMSE", "MAE"])
        has_rule = choose_rule_col(df) is not None
        has_method = find_col(df, METHOD_COLUMNS, required=False) is not None
        if not (has_metric or has_rule or has_method):
            continue

        # Limit rows only for very large non-summary files.
        if len(df) > 20000:
            log(f"[WARN] Skip very large policy-like file {path} | shape={df.shape}")
            continue

        for _, row in df.iterrows():
            cand = row_to_candidate(row, path, "csv")
            if (
                clean_string(cand["Method"])
                or clean_string(cand["Rule"])
                or np.isfinite(cand["Summary_R2"])
                or np.isfinite(cand["Summary_RMSE"])
            ):
                candidates.append(cand)

    out = pd.DataFrame(candidates)
    if out.empty:
        return pd.DataFrame(columns=[
            "Source_File", "Source_Type", "Source_Row", "Policy_ID", "Method", "Strategy",
            "Rule", "Summary_R2", "Summary_RMSE", "Summary_MAE", "Summary_Coverage"
        ])

    # Remove obvious duplicate rows.
    out = out.drop_duplicates(subset=["Policy_ID", "Method", "Strategy", "Rule", "Summary_R2", "Summary_RMSE", "Summary_MAE"], keep="first")
    return out.reset_index(drop=True)


def parse_policy_reports() -> pd.DataFrame:
    report_paths = existing_paths([
        "32d*report*.txt",
        "32c*report*.txt",
        "32e*report*.txt",
    ], folders=[TABLE_DIR, REPORT_DIR, PROJECT_ROOT])

    rows: List[Dict[str, object]] = []
    for path in report_paths:
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Split into candidate blocks when possible.
        # Most important fallback: parse "Best policy detected" block.
        policy_id = ""
        method = ""
        strategy = ""
        rule = ""
        r2 = np.nan
        rmse = np.nan
        mae = np.nan
        coverage = np.nan

        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue

            m = re.match(r"[-*]?\s*(Display\s+Policy\s+ID|Policy\s+ID)\s*:\s*(.+)", s, flags=re.I)
            if m:
                policy_id = clean_string(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Method|Model)\s*:\s*(.+)", s, flags=re.I)
            if m:
                method = clean_string(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Strategy|Validation\s+Strategy)\s*:\s*(.+)", s, flags=re.I)
            if m:
                strategy = clean_string(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Rule|Policy rule|Policy)\s*:\s*(.*)", s, flags=re.I)
            if m:
                val = clean_string(m.group(2))
                if val and not val.lower().startswith("summary"):
                    rule = val
                continue

            m = re.match(r"[-*]?\s*(Retained\s+R2|Retained\s+R²|R2)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s, flags=re.I)
            if m:
                r2 = float(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Retained\s+RMSE|RMSE)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s, flags=re.I)
            if m:
                rmse = float(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Retained\s+MAE|MAE)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s, flags=re.I)
            if m:
                mae = float(m.group(2))
                continue

            m = re.match(r"[-*]?\s*(Coverage)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s, flags=re.I)
            if m:
                coverage = float(m.group(2))
                continue

        if method or rule or np.isfinite(r2):
            rows.append({
                "Source_File": str(path),
                "Source_Type": "txt_report",
                "Source_Row": 0,
                "Policy_ID": policy_id,
                "Method": method,
                "Strategy": strategy,
                "Rule": rule,
                "Summary_R2": r2,
                "Summary_RMSE": rmse,
                "Summary_MAE": mae,
                "Summary_Coverage": coverage,
            })

    return pd.DataFrame(rows)


def enrich_missing_rules(cands: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows:
    - one row may have Strategy + metrics but no Rule;
    - another row may have Rule + metrics but no Strategy.

    This is exactly the failure mode suggested by Step 32E.
    """
    if cands.empty:
        return cands

    rows = cands.to_dict("records")
    enriched = list(rows)

    with_rule = [r for r in rows if clean_string(r.get("Rule"))]
    without_rule = [r for r in rows if not clean_string(r.get("Rule"))]

    for a in without_rule:
        for b in with_rule:
            same_method = norm_text(a.get("Method")) and norm_text(a.get("Method")) == norm_text(b.get("Method"))
            same_policy = norm_text(a.get("Policy_ID")) and norm_text(a.get("Policy_ID")) == norm_text(b.get("Policy_ID"))
            close_r2 = nearly_equal(a.get("Summary_R2"), b.get("Summary_R2"), 0.02)
            close_rmse = nearly_equal(a.get("Summary_RMSE"), b.get("Summary_RMSE"), 1.0)
            close_mae = nearly_equal(a.get("Summary_MAE"), b.get("Summary_MAE"), 1.0)

            if (same_policy or same_method) and (close_r2 or (close_rmse and close_mae)):
                nr = dict(a)
                nr["Rule"] = b.get("Rule", "")
                nr["Source_Type"] = "hybrid_csv_report"
                nr["Source_File"] = f"{a.get('Source_File')} + {b.get('Source_File')}"
                # Use coverage from whichever row has it.
                if not np.isfinite(float(nr.get("Summary_Coverage", np.nan))) and np.isfinite(float(b.get("Summary_Coverage", np.nan))):
                    nr["Summary_Coverage"] = b.get("Summary_Coverage")
                enriched.append(nr)

    out = pd.DataFrame(enriched)
    out = out.drop_duplicates(subset=["Policy_ID", "Method", "Strategy", "Rule", "Summary_R2", "Summary_RMSE", "Summary_MAE"], keep="first")
    return out.reset_index(drop=True)


# =============================================================================
# 5. Rule parsing and target-free checks
# =============================================================================

def normalize_zone(x: object) -> str:
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def find_zone_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(
        df,
        ["Reliability_Zone", "Reliability Zone", "AD_Zone", "Domain_Zone", "Zone", "RD_Zone"],
        required=False,
    )


def zone_mask(series: pd.Series, zone_value: str) -> pd.Series:
    z = normalize_zone(zone_value)
    z = re.sub(r"^eq\s+", "", z).strip()
    vals = series.astype(str).map(normalize_zone)

    if z in {"reliable", "reliable domain", "rd"}:
        keys = ["reliable"]
    elif z in {"warning", "warning domain", "warn"}:
        keys = ["warning"]
    elif z in {"outside", "outside domain", "out", "ood"}:
        keys = ["outside"]
    else:
        keys = [z]

    mask = pd.Series(False, index=series.index)
    for k in keys:
        mask |= vals.str.contains(re.escape(k), na=False)
    return mask.fillna(False)


def parse_condition_token(token: str) -> Optional[Tuple[str, str, str]]:
    token = str(token).strip()
    token = token.replace("≤", "<=").replace("≥", ">=")

    patterns = [
        r"^(?P<col>.+)_(?P<op>le|lt|ge|gt|eq|ne)_(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
        r"^(?P<col>.+)\s*(?P<op><=|<|>=|>|==|!=)\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$",
    ]
    op_map = {"<=": "le", "<": "lt", ">=": "ge", ">": "gt", "==": "eq", "!=": "ne"}
    for pat in patterns:
        m = re.match(pat, token)
        if m:
            col = m.group("col").strip()
            op = op_map.get(m.group("op"), m.group("op"))
            val = m.group("val")
            return col, op, val
    return None


def condition_mask(df: pd.DataFrame, col_hint: str, op: str, value: str) -> Tuple[pd.Series, str]:
    col = find_col(df, [col_hint], required=True)
    x = pd.to_numeric(df[col], errors="coerce")
    v = float(value)

    if op == "le":
        m = x <= v
    elif op == "lt":
        m = x < v
    elif op == "ge":
        m = x >= v
    elif op == "gt":
        m = x > v
    elif op == "eq":
        m = np.isclose(x.to_numpy(float), v, equal_nan=False)
        m = pd.Series(m, index=df.index)
    elif op == "ne":
        m = ~np.isclose(x.to_numpy(float), v, equal_nan=False)
        m = pd.Series(m, index=df.index)
    else:
        raise ValueError(f"Unsupported operator: {op}")

    return pd.Series(m, index=df.index).fillna(False), col


def split_rule_tokens(rule: str) -> List[str]:
    r = clean_string(rule)
    if not r:
        return []

    # Normalize separators while preserving underscores inside column names.
    tokens = re.split(r"__|&&|\bAND\b|\band\b|;", r)
    out = []
    for t in tokens:
        t = t.strip()
        if t:
            out.append(t)
    return out


def apply_rule(df: pd.DataFrame, rule: str) -> Tuple[pd.Series, List[str], List[str]]:
    tokens = split_rule_tokens(rule)
    if not tokens:
        return pd.Series(False, index=df.index), [], ["EMPTY_RULE"]

    used_cols: List[str] = []
    unresolved: List[str] = []
    mask = pd.Series(True, index=df.index)

    zcol = find_zone_col(df)

    for token in tokens:
        t = token.strip()

        # Examples: Zone_Outside, Zone_eq_Warning
        if norm_text(t).startswith("zone"):
            zone_value = re.sub(r"^Zone[_\s=]*", "", t, flags=re.I).strip()
            if zone_value.lower().startswith("eq_"):
                zone_value = zone_value[3:]
            if not zone_value:
                unresolved.append(f"{token}: empty zone value")
                continue
            if zcol is None:
                unresolved.append(f"{token}: no Reliability_Zone column")
                continue
            mask &= zone_mask(df[zcol], zone_value)
            used_cols.append(zcol)
            continue

        parsed = parse_condition_token(t)
        if parsed is None:
            unresolved.append(f"{token}: cannot parse")
            continue

        col_hint, op, val = parsed
        try:
            cm, actual_col = condition_mask(df, col_hint, op, val)
            mask &= cm
            used_cols.append(actual_col)
        except Exception as exc:
            unresolved.append(f"{token}: {exc}")

    used_cols = list(dict.fromkeys(used_cols))
    return mask.fillna(False), used_cols, unresolved


def target_free_check(used_cols: List[str], rule: str) -> Tuple[bool, str]:
    text = " ".join([rule] + used_cols).lower()
    bad_terms = [
        "observed", "actual", "target", "removal_rate", "f removal", "氟去除率",
        "predicted", "prediction", "residual", "error", "absolute_error",
        "rmse", "mae", "r2", "mse", "test_error", "heldout_error",
    ]
    hits = [t for t in bad_terms if t in text]
    return len(hits) == 0, ";".join(hits)


# =============================================================================
# 6. Prediction table and candidate evaluation
# =============================================================================

def load_prediction_table() -> Tuple[pd.DataFrame, Path]:
    pred_path = first_existing([
        "32c_fast_selected_holdout_predictions.csv",
        "32c_selected_holdout_predictions.csv",
        "32b_predictions_with_reliability_zone.csv",
        "32a2_ad_predictions_with_scores.csv",
        "32a_ad_predictions_with_scores.csv",
    ], required=True)
    return read_csv_safely(pred_path), pred_path


def filter_method_strategy(df: pd.DataFrame, cand: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    method_col = find_col(df, METHOD_COLUMNS, required=False)
    strategy_col = find_col(df, STRATEGY_COLUMNS, required=False)

    method = clean_string(cand.get("Method"))
    strategy = clean_string(cand.get("Strategy"))

    out = df
    info = {
        "method_col": method_col,
        "strategy_col": strategy_col,
        "method": method,
        "strategy": strategy,
        "method_filter": False,
        "strategy_filter": False,
    }

    if method_col and method:
        m = out[method_col].astype(str).map(norm_text) == norm_text(method)
        if m.sum() > 0:
            out = out.loc[m].copy()
            info["method_filter"] = True

    if strategy_col and strategy:
        s = out[strategy_col].astype(str).map(norm_text) == norm_text(strategy)
        if s.sum() > 0:
            out = out.loc[s].copy()
            info["strategy_filter"] = True

    return out, info


def evaluate_candidate(pred_df: pd.DataFrame, cand: Dict[str, object], obs_col: str, pred_col: str) -> Dict[str, object]:
    base_df, filter_info = filter_method_strategy(pred_df, cand)
    rule = clean_string(cand.get("Rule"))

    mask, used_cols, unresolved = apply_rule(base_df, rule)
    kept = base_df.loc[mask].copy()

    y = safe_float_array(kept[obs_col]) if len(kept) else np.array([])
    p = safe_float_array(kept[pred_col]) if len(kept) else np.array([])
    md = metric_dict(y, p) if len(kept) else {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "N": 0}

    tf_pass, tf_hits = target_free_check(used_cols, rule)

    summary_r2 = cand.get("Summary_R2", np.nan)
    summary_rmse = cand.get("Summary_RMSE", np.nan)
    summary_mae = cand.get("Summary_MAE", np.nan)
    summary_cov = cand.get("Summary_Coverage", np.nan)

    r2_diff = abs(float(summary_r2) - md["R2"]) if np.isfinite(float(summary_r2)) and np.isfinite(md["R2"]) else np.nan
    rmse_diff = abs(float(summary_rmse) - md["RMSE"]) if np.isfinite(float(summary_rmse)) and np.isfinite(md["RMSE"]) else np.nan
    mae_diff = abs(float(summary_mae) - md["MAE"]) if np.isfinite(float(summary_mae)) and np.isfinite(md["MAE"]) else np.nan

    coverage_vs_filtered = float(len(kept) / len(base_df)) if len(base_df) else np.nan
    coverage_vs_all = float(len(kept) / len(pred_df)) if len(pred_df) else np.nan
    coverage_diff = abs(float(summary_cov) - coverage_vs_filtered) if np.isfinite(float(summary_cov)) else np.nan

    # Consistency is strict but not impossible.
    consistency_pass = (
        np.isfinite(r2_diff) and r2_diff <= 0.01
        and (not np.isfinite(rmse_diff) or rmse_diff <= 0.25)
        and (not np.isfinite(mae_diff) or mae_diff <= 0.25)
    )

    return {
        "Policy_ID": clean_string(cand.get("Policy_ID")) or f"cand_{cand.get('Source_Row')}",
        "Method": clean_string(cand.get("Method")),
        "Strategy": clean_string(cand.get("Strategy")),
        "Rule": rule,
        "Source_File": clean_string(cand.get("Source_File")),
        "Source_Type": clean_string(cand.get("Source_Type")),
        "Summary_R2": summary_r2,
        "Summary_RMSE": summary_rmse,
        "Summary_MAE": summary_mae,
        "Summary_Coverage": summary_cov,
        "Recomputed_R2": md["R2"],
        "Recomputed_RMSE": md["RMSE"],
        "Recomputed_MAE": md["MAE"],
        "N_Retained": md["N"],
        "N_Filtered_MethodStrategy": len(base_df),
        "Coverage_vs_Filtered": coverage_vs_filtered,
        "Coverage_vs_AllRows": coverage_vs_all,
        "R2_Abs_Diff": r2_diff,
        "RMSE_Abs_Diff": rmse_diff,
        "MAE_Abs_Diff": mae_diff,
        "Coverage_Abs_Diff": coverage_diff,
        "Consistency_Pass": bool(consistency_pass),
        "Target_Free_Pass": bool(tf_pass),
        "Target_Free_Hits": tf_hits,
        "Used_Rule_Columns": ";".join(used_cols),
        "Unresolved_Rule_Tokens": ";".join(unresolved),
        "Rule_Parse_Pass": len(unresolved) == 0 and bool(rule),
        "Method_Filter_Applied": bool(filter_info["method_filter"]),
        "Strategy_Filter_Applied": bool(filter_info["strategy_filter"]),
    }


def evaluate_all_candidates(pred_df: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    obs_col = find_col(pred_df, ["Observed", "Observed_F_removal", "Observed F removal (%)", "y_true", "Actual"], required=True)
    pred_col = find_col(pred_df, ["Predicted", "Predicted_F_removal", "Predicted F removal (%)", "y_pred", "Prediction"], required=True)

    rows: List[Dict[str, object]] = []
    if cands.empty:
        return pd.DataFrame()

    for i, cand in cands.iterrows():
        rule = clean_string(cand.get("Rule"))
        # Main audit focuses on rows with actual rules.
        if not rule:
            continue
        try:
            rows.append(evaluate_candidate(pred_df, cand.to_dict(), obs_col, pred_col))
        except Exception as exc:
            rows.append({
                "Policy_ID": clean_string(cand.get("Policy_ID")) or f"cand_{i}",
                "Method": clean_string(cand.get("Method")),
                "Strategy": clean_string(cand.get("Strategy")),
                "Rule": rule,
                "Source_File": clean_string(cand.get("Source_File")),
                "Source_Type": clean_string(cand.get("Source_Type")),
                "Summary_R2": cand.get("Summary_R2", np.nan),
                "Summary_RMSE": cand.get("Summary_RMSE", np.nan),
                "Summary_MAE": cand.get("Summary_MAE", np.nan),
                "Summary_Coverage": cand.get("Summary_Coverage", np.nan),
                "Recomputed_R2": np.nan,
                "Recomputed_RMSE": np.nan,
                "Recomputed_MAE": np.nan,
                "N_Retained": 0,
                "N_Filtered_MethodStrategy": 0,
                "Coverage_vs_Filtered": np.nan,
                "Coverage_vs_AllRows": np.nan,
                "Consistency_Pass": False,
                "Target_Free_Pass": False,
                "Target_Free_Hits": "",
                "Used_Rule_Columns": "",
                "Unresolved_Rule_Tokens": str(exc),
                "Rule_Parse_Pass": False,
                "Method_Filter_Applied": False,
                "Strategy_Filter_Applied": False,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Remove exact duplicates.
    out = out.drop_duplicates(subset=["Method", "Strategy", "Rule", "Recomputed_R2", "N_Retained"], keep="first")
    out = out.sort_values(["Consistency_Pass", "Recomputed_R2", "N_Retained"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def choose_best_audited_policy(eval_df: pd.DataFrame) -> Optional[pd.Series]:
    if eval_df.empty:
        return None

    df = eval_df.copy()
    for c in ["Recomputed_R2", "Recomputed_RMSE", "Recomputed_MAE", "N_Retained", "Coverage_vs_Filtered"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Tier 1: strict reproduction and target-free.
    strict = df[
        (df["Consistency_Pass"] == True)
        & (df["Target_Free_Pass"] == True)
        & (df["Rule_Parse_Pass"] == True)
        & (df["N_Retained"] >= 50)
    ].copy()

    if not strict.empty:
        return strict.sort_values(["Recomputed_R2", "N_Retained"], ascending=[False, False]).iloc[0]

    # Tier 2: target-free and parsed, even if summary consistency fails.
    parsed = df[
        (df["Target_Free_Pass"] == True)
        & (df["Rule_Parse_Pass"] == True)
        & (df["N_Retained"] >= 50)
    ].copy()
    if not parsed.empty:
        return parsed.sort_values(["Recomputed_R2", "N_Retained"], ascending=[False, False]).iloc[0]

    # Tier 3: highest recomputed R2 with any parsed rule.
    parsed_any = df[(df["Rule_Parse_Pass"] == True) & (df["N_Retained"] >= 10)].copy()
    if not parsed_any.empty:
        return parsed_any.sort_values(["Recomputed_R2", "N_Retained"], ascending=[False, False]).iloc[0]

    return df.sort_values(["Recomputed_R2", "N_Retained"], ascending=[False, False]).iloc[0]


def retained_predictions_for_policy(pred_df: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    obs_col = find_col(pred_df, ["Observed", "Observed_F_removal", "Observed F removal (%)", "y_true", "Actual"], required=True)
    pred_col = find_col(pred_df, ["Predicted", "Predicted_F_removal", "Predicted F removal (%)", "y_pred", "Prediction"], required=True)

    cand = policy.to_dict()
    base_df, _ = filter_method_strategy(pred_df, cand)
    mask, used_cols, unresolved = apply_rule(base_df, clean_string(policy.get("Rule")))
    out = base_df.loc[mask].copy()
    out["_32F_Audited_Policy_ID"] = clean_string(policy.get("Policy_ID"))
    out["_32F_Audited_Method"] = clean_string(policy.get("Method"))
    out["_32F_Audited_Strategy"] = clean_string(policy.get("Strategy"))
    out["_32F_Audited_Rule"] = clean_string(policy.get("Rule"))
    out["_32F_Observed_Column"] = obs_col
    out["_32F_Predicted_Column"] = pred_col
    out["_32F_Used_Rule_Columns"] = ";".join(used_cols)
    out["_32F_Unresolved_Rule_Tokens"] = ";".join(unresolved)
    return out


def split_stability(pred_df: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    retained = retained_predictions_for_policy(pred_df, policy)
    if retained.empty:
        return pd.DataFrame()

    obs_col = find_col(retained, ["Observed", "Observed_F_removal", "Observed F removal (%)", "y_true", "Actual"], required=True)
    pred_col = find_col(retained, ["Predicted", "Predicted_F_removal", "Predicted F removal (%)", "y_pred", "Prediction"], required=True)
    split_col = find_col(retained, ["Split_ID", "Seed_or_Fold", "Seed", "Fold", "Random_Seed"], required=False)

    if split_col is None:
        return pd.DataFrame()

    rows = []
    for sid, g in retained.groupby(split_col, dropna=False):
        y = safe_float_array(g[obs_col])
        p = safe_float_array(g[pred_col])
        md = metric_dict(y, p)
        rows.append({
            "Split_ID": sid,
            "R2": md["R2"],
            "RMSE": md["RMSE"],
            "MAE": md["MAE"],
            "N_Retained": md["N"],
        })
    return pd.DataFrame(rows)


def group_dominance(retained: pd.DataFrame) -> pd.DataFrame:
    if retained.empty:
        return pd.DataFrame()
    group_col = find_col(retained, ["文献编号", "Literature_ID", "Paper_ID", "Source_Group", "Group", "Paper"], required=False)
    if group_col is None:
        return pd.DataFrame()

    out = (
        retained[group_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Literature_Group")
        .reset_index(name="N_Retained")
    )
    out["Fraction_of_Retained"] = out["N_Retained"] / out["N_Retained"].sum()
    return out


# =============================================================================
# 7. Figures
# =============================================================================

def plot_top_policy_metrics(eval_df: pd.DataFrame) -> None:
    if eval_df.empty:
        return

    df = eval_df.copy()
    df = df[(df["Rule_Parse_Pass"] == True) & (df["N_Retained"] > 0)].copy()
    if df.empty:
        df = eval_df.copy()

    df = df.sort_values("Recomputed_R2", ascending=False).head(12).copy()
    df["Display"] = [f"P{i+1:02d}" for i in range(len(df))]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
    axes = axes.ravel()

    panels = [
        ("Recomputed_R2", r"Recomputed retained $R^2$", "{:.3f}"),
        ("Recomputed_RMSE", "Recomputed retained RMSE", "{:.2f}"),
        ("Recomputed_MAE", "Recomputed retained MAE", "{:.2f}"),
        ("Coverage_vs_Filtered", "Coverage within method-strategy subset", "{:.3f}"),
    ]

    for ax, (col, title, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        colors = [COLOR_GREEN if (col == "Recomputed_R2" and np.isfinite(v) and v >= 0.5) else COLOR_BLUE for v in vals]
        ax.bar(df["Display"], vals, color=colors, edgecolor="#333333", linewidth=0.8, width=0.72)
        ax.set_title(title)
        ax.set_xlabel("Policy ID")
        ax.set_ylabel(title.split()[-1] if "Coverage" not in title else "Coverage")
        est_ax(ax, grid_axis="y")
        annotate_bars(ax, fmt, fontsize=9)

        if col == "Recomputed_R2":
            ax.axhline(0.5, color=COLOR_RED, linestyle="--", linewidth=1.1, label=r"$R^2=0.5$")
            ax.legend(frameon=True, loc="best")

    fig.suptitle("32F reconstructed selective AD policy audit", fontsize=17, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "174_32f_reconstructed_policy_metrics.png")


def plot_r2_coverage_frontier(eval_df: pd.DataFrame) -> None:
    if eval_df.empty:
        return

    df = eval_df.copy()
    df["Recomputed_R2"] = pd.to_numeric(df["Recomputed_R2"], errors="coerce")
    df["Coverage_vs_Filtered"] = pd.to_numeric(df["Coverage_vs_Filtered"], errors="coerce")
    df = df[np.isfinite(df["Recomputed_R2"]) & np.isfinite(df["Coverage_vs_Filtered"])].copy()
    if df.empty:
        return

    top = df.sort_values("Recomputed_R2", ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(9.8, 6.5))
    est_ax(ax, grid_axis="both")

    ax.scatter(
        df["Coverage_vs_Filtered"], df["Recomputed_R2"],
        s=42, color=COLOR_LIGHT, alpha=0.65,
        edgecolor="white", linewidth=0.25,
        label="Audited candidate policies",
    )

    ax.scatter(
        top["Coverage_vs_Filtered"], top["Recomputed_R2"],
        s=82, color=COLOR_BLUE, alpha=0.9,
        edgecolor="white", linewidth=0.45,
        label="Top retained $R^2$ policies",
    )

    ax.axhline(0.5, color=COLOR_RED, linestyle="--", linewidth=1.2, label=r"$R^2=0.5$")
    ax.axvline(0.01, color="#666666", linestyle=":", linewidth=1.2, label="1% coverage")

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        label = f"P{i:02d}\n{row.get('Method', '')}"
        ax.text(
            row["Coverage_vs_Filtered"],
            row["Recomputed_R2"] + 0.02,
            label,
            fontsize=8.8,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Coverage within method-strategy subset")
    ax.set_ylabel(r"Recomputed retained $R^2$")
    ax.set_title(r"$R^2$–coverage frontier after policy reconstruction")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    save_fig(fig, "175_32f_reconstructed_r2_coverage_frontier.png")


def plot_best_prediction(retained: pd.DataFrame, policy: pd.Series) -> None:
    if retained.empty:
        return

    obs_col = find_col(retained, ["Observed", "Observed_F_removal", "Observed F removal (%)", "y_true", "Actual"], required=True)
    pred_col = find_col(retained, ["Predicted", "Predicted_F_removal", "Predicted F removal (%)", "y_pred", "Prediction"], required=True)

    y = safe_float_array(retained[obs_col])
    p = safe_float_array(retained[pred_col])
    m = finite_mask(y, p)
    y = y[m]
    p = p[m]
    md = metric_dict(y, p)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    est_ax(ax, grid_axis="both")

    ax.scatter(y, p, s=48, alpha=0.82, color=COLOR_BLUE, edgecolor="white", linewidth=0.45)
    ax.plot([0, 100], [0, 100], "--", color=COLOR_RED, linewidth=1.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed F removal (%)")
    ax.set_ylabel("Predicted F removal (%)")
    ax.set_title(f"32F audited retained subdomain: {policy.get('Method', '')}")

    box = (
        rf"$R^2$ = {md['R2']:.3f}" + "\n"
        f"RMSE = {md['RMSE']:.2f}\n"
        f"MAE = {md['MAE']:.2f}\n"
        f"n = {md['N']}\n"
        f"coverage = {float(policy.get('Coverage_vs_Filtered', np.nan)):.3f}"
    )
    ax.text(
        0.06, 0.94, box,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    fig.tight_layout()
    save_fig(fig, "176_32f_audited_best_prediction_panel.png")


def plot_split_stability(split_df: pd.DataFrame) -> None:
    if split_df.empty:
        return

    df = split_df.copy()
    df["Split_Label"] = df["Split_ID"].astype(str)
    df = df.sort_values("Split_Label").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    panels = [
        ("R2", r"$R^2$", "{:.2f}"),
        ("RMSE", "RMSE", "{:.2f}"),
        ("N_Retained", "N retained", "{:.0f}"),
    ]

    for ax, (col, ylabel, fmt) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if col == "R2":
            colors = [COLOR_GREEN if np.isfinite(v) and v >= 0.5 else COLOR_ORANGE for v in vals]
        else:
            colors = [COLOR_BLUE for _ in vals]
        ax.bar(df["Split_Label"], vals, color=colors, edgecolor="#333333", linewidth=0.8)
        ax.set_title(ylabel)
        ax.set_xlabel("Split")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        est_ax(ax, grid_axis="y")
        annotate_bars(ax, fmt, fontsize=9)
        if col == "R2":
            ax.axhline(0.5, color=COLOR_RED, linestyle="--", linewidth=1.1)

    fig.suptitle("32F split-level stability of audited retained subdomain", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "177_32f_split_level_stability.png")


def plot_group_dominance(group_df: pd.DataFrame) -> None:
    if group_df.empty:
        return

    df = group_df.head(15).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.barh(df["Literature_Group"], df["N_Retained"], color=COLOR_BLUE, edgecolor="#333333", linewidth=0.8)
    ax.set_xlabel("Retained sample count")
    ax.set_ylabel("Literature group")
    ax.set_title("Literature-group dominance in audited retained subdomain")
    est_ax(ax, grid_axis="x")

    xmax = max(float(df["N_Retained"].max()), 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["N_Retained"] + 0.02 * xmax,
            i,
            f"{int(row['N_Retained'])} ({row['Fraction_of_Retained']:.1%})",
            va="center", ha="left", fontsize=9.5,
        )

    fig.tight_layout()
    save_fig(fig, "178_32f_retained_literature_group_dominance.png")


# =============================================================================
# 8. Report
# =============================================================================

def make_report(eval_df: pd.DataFrame, best: Optional[pd.Series], split_df: pd.DataFrame, group_df: pd.DataFrame, pred_path: Path, cand_count: int) -> str:
    lines = []
    lines.append("32F reconstructed selective AD policy audit report")
    lines.append("=" * 88)
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Prediction table used: {pred_path}")
    lines.append(f"Raw candidate rows collected: {cand_count}")
    lines.append(f"Evaluated policy rows with non-empty rules: {len(eval_df)}")
    lines.append("")

    if best is None:
        lines.append("Decision: NO_VALID_POLICY_FOUND")
        lines.append("No non-empty target-free policy rule could be reconstructed.")
        return "\n".join(lines)

    lines.append("Best audited policy")
    lines.append("-" * 88)
    lines.append(f"Policy ID: {best.get('Policy_ID')}")
    lines.append(f"Method: {best.get('Method')}")
    lines.append(f"Strategy: {best.get('Strategy')}")
    lines.append(f"Rule: {best.get('Rule')}")
    lines.append(f"Source type: {best.get('Source_Type')}")
    lines.append(f"Source file: {best.get('Source_File')}")
    lines.append("")
    lines.append("Summary vs recomputed metrics")
    lines.append("-" * 88)
    lines.append(f"Summary R2/RMSE/MAE/Coverage: {best.get('Summary_R2')}, {best.get('Summary_RMSE')}, {best.get('Summary_MAE')}, {best.get('Summary_Coverage')}")
    lines.append(f"Recomputed R2/RMSE/MAE/N: {best.get('Recomputed_R2'):.6f}, {best.get('Recomputed_RMSE'):.6f}, {best.get('Recomputed_MAE'):.6f}, {int(best.get('N_Retained'))}")
    lines.append(f"Coverage vs filtered method-strategy rows: {best.get('Coverage_vs_Filtered'):.6f}")
    lines.append(f"Coverage vs all prediction rows: {best.get('Coverage_vs_AllRows'):.6f}")
    lines.append("")
    lines.append("Audit flags")
    lines.append("-" * 88)
    lines.append(f"Consistency pass: {bool(best.get('Consistency_Pass'))}")
    lines.append(f"Target-free pass: {bool(best.get('Target_Free_Pass'))} | hits: {best.get('Target_Free_Hits')}")
    lines.append(f"Rule parse pass: {bool(best.get('Rule_Parse_Pass'))} | unresolved: {best.get('Unresolved_Rule_Tokens')}")
    lines.append(f"Used rule columns: {best.get('Used_Rule_Columns')}")
    lines.append(f"Method filter applied: {bool(best.get('Method_Filter_Applied'))}")
    lines.append(f"Strategy filter applied: {bool(best.get('Strategy_Filter_Applied'))}")

    if not split_df.empty:
        sr2 = pd.to_numeric(split_df["R2"], errors="coerce")
        good = int((sr2 >= 0.5).sum())
        total = int(sr2.notna().sum())
        min_n = int(pd.to_numeric(split_df["N_Retained"], errors="coerce").min())
        lines.append(f"Split-level R2 >= 0.5: {good}/{total}")
        lines.append(f"Minimum retained samples per split: {min_n}")
    else:
        lines.append("Split-level audit: unavailable")

    if not group_df.empty:
        lines.append(f"Largest retained literature-group fraction: {group_df['Fraction_of_Retained'].max():.4f}")
    else:
        lines.append("Literature-group dominance audit: unavailable")

    lines.append("")
    lines.append("Decision")
    lines.append("-" * 88)
    r2_ok = np.isfinite(float(best.get("Recomputed_R2"))) and float(best.get("Recomputed_R2")) >= 0.5
    n_ok = int(best.get("N_Retained")) >= 50
    target_free = bool(best.get("Target_Free_Pass"))
    parsed = bool(best.get("Rule_Parse_Pass"))
    consistent = bool(best.get("Consistency_Pass"))

    if consistent and target_free and parsed and r2_ok and n_ok:
        lines.append("STRICT_ACCEPT: The high-R2 selective AD result is reproducible from the prediction table.")
        lines.append("It can be reported as a target-free selective reliable-subdomain result, not as full-domain generalization.")
    elif target_free and parsed and r2_ok and n_ok:
        lines.append("EXPLORATORY_ACCEPT_ONLY: The recomputed retained R2 is high and the rule is target-free, but summary consistency is not perfect.")
        lines.append("Use only as exploratory selective-subdomain evidence unless the metric mismatch is explained.")
    else:
        lines.append("DO_NOT_ACCEPT: The high-R2 policy is not manuscript-ready.")
        lines.append("Do not use the previous 32D high-R2 number as a final conclusion.")
    lines.append("")
    lines.append("Generated outputs")
    lines.append("-" * 88)
    lines.append("Tables:")
    lines.append("  - 32f_collected_policy_candidates.csv")
    lines.append("  - 32f_reconstructed_policy_audit.csv")
    lines.append("  - 32f_best_audited_retained_predictions.csv")
    lines.append("  - 32f_best_policy_split_stability.csv")
    lines.append("  - 32f_best_policy_literature_group_dominance.csv")
    lines.append("  - 32f_reconstructed_policy_audit_report.txt")
    lines.append("Figures:")
    lines.append("  - 174_32f_reconstructed_policy_metrics.png/pdf")
    lines.append("  - 175_32f_reconstructed_r2_coverage_frontier.png/pdf")
    lines.append("  - 176_32f_audited_best_prediction_panel.png/pdf")
    lines.append("  - 177_32f_split_level_stability.png/pdf")
    lines.append("  - 178_32f_retained_literature_group_dominance.png/pdf")
    return "\n".join(lines)


# =============================================================================
# 9. Main
# =============================================================================

def main() -> None:
    log("=" * 88)
    log(f"Running {SCRIPT_NAME}")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 88)

    pred_df, pred_path = load_prediction_table()

    csv_cands = collect_candidates_from_csvs()
    txt_cands = parse_policy_reports()

    cands = pd.concat([csv_cands, txt_cands], ignore_index=True)
    cands = enrich_missing_rules(cands)

    if cands.empty:
        log("[ERROR] No policy candidates collected.")
        save_csv(cands, "32f_collected_policy_candidates.csv")
        return

    # Keep useful candidates and save.
    cands = cands.drop_duplicates(subset=["Policy_ID", "Method", "Strategy", "Rule", "Summary_R2", "Summary_RMSE", "Summary_MAE"], keep="first").reset_index(drop=True)
    save_csv(cands, "32f_collected_policy_candidates.csv")

    log("[STEP] Evaluating reconstructed policy rules...")
    eval_df = evaluate_all_candidates(pred_df, cands)
    save_csv(eval_df, "32f_reconstructed_policy_audit.csv")

    best = choose_best_audited_policy(eval_df)
    retained = pd.DataFrame()
    split_df = pd.DataFrame()
    group_df = pd.DataFrame()

    if best is not None:
        retained = retained_predictions_for_policy(pred_df, best)
        split_df = split_stability(pred_df, best)
        group_df = group_dominance(retained)

    save_csv(retained, "32f_best_audited_retained_predictions.csv")
    save_csv(split_df, "32f_best_policy_split_stability.csv")
    save_csv(group_df, "32f_best_policy_literature_group_dominance.csv")

    plot_top_policy_metrics(eval_df)
    plot_r2_coverage_frontier(eval_df)
    if best is not None and not retained.empty:
        plot_best_prediction(retained, best)
    if not split_df.empty:
        plot_split_stability(split_df)
    if not group_df.empty:
        plot_group_dominance(group_df)

    report = make_report(eval_df, best, split_df, group_df, pred_path, cand_count=len(cands))
    report_path = TABLE_DIR / "32f_reconstructed_policy_audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log(f"[REPORT] {report_path}")
    log("")
    log(report)

    elapsed = time.time() - START_TIME
    log("=" * 88)
    log("32F reconstructed selective AD policy audit finished.")
    log(f"Total runtime / elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    log("=" * 88)


if __name__ == "__main__":
    main()
