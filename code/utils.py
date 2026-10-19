import re
import datetime as dt
import numpy as np
import pandas as pd

_BASE_EXCEL_DATE = dt.datetime(1899, 12, 31)


def excel_datetime_to_number(value):
    """把被 Excel 错误识别成日期的数值还原成近似数字，例如 1900-01-08 12:00 -> 8.5。"""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        return (value - _BASE_EXCEL_DATE).total_seconds() / 86400
    if isinstance(value, dt.time):
        return value.hour + value.minute / 60 + value.second / 3600
    return np.nan


def parse_numeric(value):
    """从混杂单元格中提取数值：范围取均值，斜杠分隔多个值取均值，普通文本取第一个数字。"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.time)):
        return float(excel_datetime_to_number(value))

    s = str(value).strip()
    s = s.replace("−", "-").replace("～", "-").replace("—", "-").replace("–", "-")
    s = s.replace("，", ",").replace("、", ",").replace("；", ";")
    s = s.replace("约", "").replace("最高", "").replace("低于", "").replace("以下", "")

    # 处理 20-40 这类范围；避免把负数识别成范围。
    range_match = re.search(r"(?<![\d.])-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?", s)
    if range_match:
        nums = re.findall(r"-?\d+(?:\.\d+)?", range_match.group(0))
        if len(nums) >= 2:
            return float(np.mean([float(nums[0]), float(nums[1])]))

    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return np.nan
    if "/" in s and len(nums) >= 2:
        return float(np.mean([float(x) for x in nums]))
    return float(nums[0])


def count_numbers(value):
    if pd.isna(value) or isinstance(value, (int, float, np.integer, np.floating)):
        return 0
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.time)):
        return 1
    return len(re.findall(r"-?\d+(?:\.\d+)?", str(value)))


def save_current_figure(path, dpi=300):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
