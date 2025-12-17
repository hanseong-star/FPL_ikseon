import random
from pathlib import Path

# =========================
# 설정
# =========================
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_A"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_B"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_C"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_A"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_B"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_C"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_na_D"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_A"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_B"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_da_C"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero_26"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero_28"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero_30"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero_32_ga"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/samlidaero_32"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_A"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_B"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_C"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_D"),
#     Path("/home/hanseong/gdrive/ML_FPL_raw_data/suporo_28_E")
# # 🔴 처리할 폴더들을 "순서대로" 나열
FOLDERS = [
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_A"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_B"),
    Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_C"),
]

PREFIX = "raw"
START_IDX = 2689          # raw_0000부터 시작
PAD = 4                # raw_0000 형식

RAW_EXTS = [".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw"]

# =========================
# 메인 로직
# =========================

def process_folders_sequentially():
    global_idx = START_IDX

    for folder in FOLDERS:
        print(f"\n📂 폴더 처리 중: {folder}")

        if not folder.exists():
            print(f"❌ 폴더 없음: {folder} → 스킵")
            continue

        # 1️⃣ RAW 파일 수집
        files = sorted([
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in RAW_EXTS
        ])

        count = len(files)
        if count == 0:
            print("⚠ RAW 파일 없음 → 스킵")
            continue

        print(f"  RAW 파일 개수: {count}")

        # 2️⃣ 임시 이름으로 변경 (충돌 완전 차단)
        temp_files = []
        for i, f in enumerate(files):
            tmp = folder / f"__temp__{i}{f.suffix.lower()}"
            f.rename(tmp)
            temp_files.append(tmp)

        print("  ✅ 임시 이름 변경 완료")

        # 3️⃣ 랜덤 섞기
        random.shuffle(temp_files)

        # 4️⃣ 최종 이름 부여 (번호 이어서)
        for tmp in temp_files:
            num = str(global_idx).zfill(PAD)
            new_name = f"{PREFIX}_{num}{tmp.suffix.lower()}"
            new_path = folder / new_name

            tmp.rename(new_path)
            print(f"    {tmp.name} → {new_name}")

            global_idx += 1

        print(f"  🎉 폴더 완료, 다음 시작 번호: {global_idx}")

    print("\n🔥 모든 폴더 처리 완료!")
    print(f"최종 마지막 번호: {global_idx - 1}")

# =========================
# 실행
# =========================

if __name__ == "__main__":
    process_folders_sequentially()
