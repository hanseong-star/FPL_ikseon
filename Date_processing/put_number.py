import numpy as np
import pandas as pd


TRAINING_LABEL_CSV = "/home/hanseong/gdrive/ML_FPL_training_data/traing_labels.csv"
TEST_LABEL_CSV  = "/home/hanseong/gdrive/ML_FPL_test_data/test_labels.csv"

# 1) CSV 다시 로드 (이미 df_train/df_test 있으면 이 줄은 생략 가능)
df_train = pd.read_csv(TRAINING_LABEL_CSV)
df_test  = pd.read_csv(TEST_LABEL_CSV)

# 2) road_key 만들기: detail이 0이면 road_name만, 아니면 road_name_detail
def make_road_key(row):
    road = str(row["road_name"])
    detail = row["detail"]
    if pd.isna(detail) or str(detail) == "0":
        return road
    return f"{road}_{detail}"

df_train["road_key"] = df_train.apply(make_road_key, axis=1)
df_test["road_key"]  = df_test.apply(make_road_key, axis=1)

# 3) 네가 준 상대 위치값 매핑(현재까지)
#    - 아직 안 준 건 NaN으로 남겨두고, 나중에 dict에 계속 추가하면 됨
rel_pos_map = {
    # 0
    "suporo_28_C": 0,

    # 1
    "suporo_28_B": 1,
    "donhwamunro_11_da_A": 1,

    # 2
    "suporo_28_A": 2,
    "suporo_28_D": 2,
    "samildaero_26": 2,
    "samildaero_28": 2,
    "donhwamunro_11_na_B": 2,
    "donhwamunro_11_da_B": 2,
    "suporo_A": 2,

    # 오타/변형도 방어적으로 같이 넣기(있어도 손해 없음)
    "samildearo_26": 2,     # 사용자가 적은 철자
    "samildearo_28": 2,
    "donhwmunro_11_da_B": 2,  # 사용자가 적은 철자

    
}

# 4) rel_pos 컬럼 생성 (매핑 없으면 NaN)
df_train["rel_pos"] = df_train["road_key"].map(rel_pos_map).astype("float")
df_test["rel_pos"]  = df_test["road_key"].map(rel_pos_map).astype("float")

# 5) 확인 출력
print("Train rel_pos filled:", df_train["rel_pos"].notna().sum(), "/", len(df_train))
print("Test  rel_pos filled:", df_test["rel_pos"].notna().sum(), "/", len(df_test))

# 6) 아직 매핑 안 된 road_key 목록(나중에 네가 숫자 계속 추가할 때 필요)
unmapped_train = sorted(df_train.loc[df_train["rel_pos"].isna(), "road_key"].unique())
unmapped_test  = sorted(df_test.loc[df_test["rel_pos"].isna(), "road_key"].unique())

print("\nUnmapped road_key in TRAIN (sample up to 30):")
print(unmapped_train[:30], " ... total:", len(unmapped_train))

print("\nUnmapped road_key in TEST (sample up to 30):")
print(unmapped_test[:30], " ... total:", len(unmapped_test))

# 7) photo_id 옆에 보고 싶으면 이렇게
print("\nTrain preview:")
print(df_train[["road_name", "detail", "photo_id", "road_key", "rel_pos"]].head(10))

print("\nTest preview:")
print(df_test[["road_name", "detail", "photo_id", "road_key", "rel_pos"]].head(10))
