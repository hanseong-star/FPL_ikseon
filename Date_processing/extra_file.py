from pathlib import Path
import shutil
import re

# ✅ 1) 폴더 23개를 "원하는 순서대로" 넣어줘 (중요)
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

# ✅ 2) 각 폴더에서 뽑을 개수 (길이 23이어야 함)
COUNTS = [
    62, 
    16, 
    27,
    29,
    43,
    23,
    42,
    18,
    17,
    30,
    16,
    12,
    66,
    26,
    38,
    59,
    43,
    79,
    35,
    20,
    12,
    20,
    10
]
COUNTS = [
    62, 16, 27, 29, 43, 23, 42, 18, 17, 30, 16, 12,
    66, 26, 38, 59, 43, 79, 35, 20, 12, 20, 10
]
# ==============================

# ✅ 결과 폴더(원하면 경로 바꿔도 됨)
TEST_DIR  = Path("/home/hanseong/gdrive/ML_FPL_test_data")
TRAIN_DIR = Path("/home/hanseong/gdrive/ML_FPL_training_data")
TEST_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# ✅ RAW 확장자(필요하면 추가)
RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw"}

# 파일명 끝 숫자 추출: raw_0123 -> 123, DSC_0001 -> 1
_num_pat = re.compile(r"(\d+)$")

def sort_key(p: Path):
    m = _num_pat.search(p.stem)
    if m:
        return (0, int(m.group(1)))
    return (1, p.name.lower())

def copy_files(file_list, dst_root: Path, folder_name: str):
    copied = 0
    for f in file_list:
        # 덮어쓰기 방지: 폴더명을 prefix로 붙여 저장
        dst = dst_root / f"{folder_name}_{f.name}"
        if dst.exists():
            dst = dst_root / f"{folder_name}_{f.stem}_dup{f.suffix.lower()}"
        shutil.copyfile(f, dst)
        copied += 1
    return copied

def main():
    if len(FOLDERS) != len(COUNTS):
        raise ValueError(f"FOLDERS({len(FOLDERS)})와 COUNTS({len(COUNTS)}) 길이가 달라요.")

    total_test = 0
    total_train = 0

    for i, (folder, n_test) in enumerate(zip(FOLDERS, COUNTS), start=1):
        if not folder.exists():
            print(f"[{i}] ❌ 폴더 없음: {folder} (스킵)")
            continue

        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in RAW_EXTS]
        files_sorted = sorted(files, key=sort_key)

        total = len(files_sorted)
        if total == 0:
            print(f"[{i}] ⚠ RAW 없음: {folder.name} (스킵)")
            continue

        # 뒤에서 n개 = test
        n_test = max(0, min(n_test, total))
        test_files  = files_sorted[-n_test:] if n_test > 0 else []
        train_files = files_sorted[:-n_test] if n_test > 0 else files_sorted

        c_test  = copy_files(test_files,  TEST_DIR,  folder.name)
        c_train = copy_files(train_files, TRAIN_DIR, folder.name)

        total_test  += c_test
        total_train += c_train

        print(f"[{i}] 📂 {folder.name}: 전체 {total}개 → test {c_test}개, train {c_train}개")

    print("-" * 60)
    print(f"✅ 최종 합계: test {total_test}개, train {total_train}개")
    print(f"📁 TEST_DIR : {TEST_DIR}")
    print(f"📁 TRAIN_DIR: {TRAIN_DIR}")

if __name__ == "__main__":
    main()