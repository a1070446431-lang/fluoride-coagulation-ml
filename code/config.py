from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "数据集修改最终.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

for p in [PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RENAME_MAP = {
    "氟离子分布形态": "氟形态",
    "游离氟占比": "游离氟占比",
    "初始 F⁻(mg/L)": "初始氟_mg_L",
    "干扰离子": "干扰离子",
    "干扰离子浓度 (mmol/L)": "干扰离子浓度_mmol_L",
    "水温 (℃)": "水温_C",
    "混凝剂": "混凝剂",
    "剂量 (mM)": "剂量_mM",
    "进水 pH": "进水pH",
    "停留时间（min）": "停留时间_min",
    "出水 pH": "出水pH",
    "氟去除率 (%)": "氟去除率_pct",
    "出水 F⁻(mg/L)": "出水氟_mg_L",
}

NUMERIC_COLS = [
    "游离氟占比", "初始氟_mg_L", "干扰离子浓度_mmol_L", "水温_C", "剂量_mM",
    "进水pH", "停留时间_min", "出水pH", "氟去除率_pct", "出水氟_mg_L"
]

INPUT_FEATURES = [
    "氟形态", "游离氟占比", "初始氟_mg_L", "干扰离子", "干扰离子浓度_mmol_L",
    "水温_C", "混凝剂", "剂量_mM", "进水pH", "停留时间_min"
]
TARGET = "氟去除率_pct"
