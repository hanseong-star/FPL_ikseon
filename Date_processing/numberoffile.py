from pathlib import Path

# 🔴 파일 개수를 확인할 폴더들 (순서대로)
FOLDERS = [
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_A"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_B"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_C"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_A"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_B"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_C"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_D"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_A"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_B"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_C"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero_26"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero_28"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero_30"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero_32_ga"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/samildaero_32"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_A"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_B"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_C"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_D"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_E")
]


RAW_EXTS = [".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw"]

total_count = 0

for folder in FOLDERS:
    if not folder.exists():
        print(f"❌ 폴더 없음: {folder}")
        continue

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in RAW_EXTS
    ]

    count = len(files)
    total_count += count

    print(f"📂 {folder.name} : {count}개")

print("-" * 30)
print(f"✅ 전체 RAW 파일 개수 합계: {total_count}개")
