import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "GCMS_单个样本数据"
REPORT_DIR = ROOT / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "preflight_report.json"

SCRIPTS = [
    ROOT / "CNN.py",
    ROOT / "CNN加偏移.py",
    ROOT / "CNN+Transformer的自注意力机制.py",
]


def check_data_dir(path: Path):
    result = {
        "path": str(path),
        "exists": path.exists(),
        "xlsx_count": 0,
        "unique_columns": 0,
        "row_count_min": None,
        "row_count_max": None,
        "rt_min": None,
        "rt_max": None,
    }
    if not path.exists():
        return result

    files = sorted(path.glob("*.xlsx")) + sorted(path.glob("*.xls"))
    result["xlsx_count"] = len(files)
    if not files:
        return result

    cols = []
    rows = []
    rt_mins = []
    rt_maxs = []

    for f in files:
        df = pd.read_excel(f)
        cols.append(tuple(df.columns))

        rt_col = [c for c in df.columns if ("rt" in str(c).lower() or "保留时间" in str(c))]
        int_col = [c for c in df.columns if ("intensity" in str(c).lower() or "归一化强度" in str(c))]
        if not rt_col or not int_col:
            continue

        sub = df[[rt_col[0], int_col[0]]].dropna().sort_values(by=rt_col[0])
        rows.append(len(sub))
        rt_mins.append(float(sub[rt_col[0]].min()))
        rt_maxs.append(float(sub[rt_col[0]].max()))

    result["unique_columns"] = len(set(cols))
    if rows:
        result["row_count_min"] = min(rows)
        result["row_count_max"] = max(rows)
    if rt_mins:
        result["rt_min"] = min(rt_mins)
        result["rt_max"] = max(rt_maxs)

    return result


def check_scripts(paths):
    out = []
    hardcoded_pattern = re.compile(r"[A-Za-z]:\\")
    for p in paths:
        item = {
            "path": str(p),
            "exists": p.exists(),
            "hardcoded_windows_path_matches": 0,
        }
        if p.exists():
            text = p.read_text(encoding="utf-8")
            item["hardcoded_windows_path_matches"] = len(hardcoded_pattern.findall(text))
        out.append(item)
    return out


report = {
    "data": check_data_dir(DATA_DIR),
    "scripts": check_scripts(SCRIPTS),
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Report saved: {REPORT_PATH}")
