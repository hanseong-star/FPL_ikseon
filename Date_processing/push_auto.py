import pandas as pd

CSV_IN  = "/home/hanseong/gdrive/ML_FPL_test_data/test_labels.csv"
CSV_OUT = "/home/hanseong/gdrive/ML_FPL_test_data/test_labels_plus.csv"

# 이미 값이 있으면 유지하고, 비어있을 때만 채울지 여부
FILL_ONLY_IF_EMPTY = True

# (road_name, detail, x, y)
# detail 없는 경우: "", None, NaN 모두 대비해서 아래에서 처리함
RULES = [
    ("donhwamunro_11_da", "A", 5, 3),
    ("donhwamunro_11_da", "B", 6, 2),
    ("donhwamunro_11_da", "C", 6, 0),
    ("donhwamunro_11_ga", "A", 6, 6),
    ("donhwamunro_11_ga", "B", 6, 3),
    ("donhwamunro_11_ga", "C", 7, 1),
    ("donhwamunro_11_na", "A", 4, 6),
    ("donhwamunro_11_na", "B", 4, 3),
    ("donhwamunro_11_na", "C", 6, 2),
    ("donhwamunro_11_na", "D", 6, 1),
    ("donhwamunro_11", "0", 4, 0),
    ("donhwamunro", "0", 7, 3),
    ("samildaero_26", "0", 2, 1),
    ("samildaero_28", "0", 2, 2),
    ("samildaero_30", "0", 4, 4),
    ("samildaero_32_ga", "0", 2, 5),
    ("samildaero_32", "0", 3, 7),
    ("samildaero", "0", 0, 4),
    ("suporo_28", "A", 3, 0),
    ("suporo_28", "B", 3, 2),
    ("suporo_28", "C", 4, 2),
    ("suporo_28", "D", 5, 2),
    ("suporo_28", "E", 6, 1),
]

df = pd.read_csv(CSV_IN)

# gps_lat/gps_lon -> x/y 로 컬럼명 변경
df = df.rename(columns={"gps_lat": "x", "gps_lon": "y"})

# 혹시 x,y가 없으면 생성
if "x" not in df.columns:
    df["x"] = pd.NA
if "y" not in df.columns:
    df["y"] = pd.NA

# 문자열 비교 준비 (NaN 안전)
rn = df["road_name"].astype(str)
dt = df["detail"].fillna("").astype(str)   # NaN이면 ""로 취급

for road, detail, xv, yv in RULES:
    mask = (rn == str(road)) & (dt == str(detail))
    if FILL_ONLY_IF_EMPTY:
        mask = mask & (df["x"].isna() | df["y"].isna())

    df.loc[mask, "x"] = xv
    df.loc[mask, "y"] = yv

df.to_csv(CSV_OUT, index=False)
print("Saved:", CSV_OUT)
