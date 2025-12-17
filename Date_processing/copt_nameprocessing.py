from pathlib import Path

FOLDER = Path("/home/hanseong/gdrive/ML_FPL_raw_data/donhwamunro_11_ga_C")  # 🔴 실제 폴더로 수정

for p in FOLDER.iterdir():
    if not p.is_file():
        continue

    name = p.name

    # 케이스 1: "xxx.NEF의 사본"
    if name.endswith(".NEF의 사본"):
        new_name = name.replace(".NEF의 사본", ".NEF")
        new_path = FOLDER / new_name

        # 혹시 같은 이름이 이미 있으면 덮어쓰기 방지
        if new_path.exists():
            stem = p.stem.replace(".NEF의 사본", "")
            new_path = FOLDER / f"{stem}_copy.NEF"

        p.rename(new_path)
        print(f"FIX: {name} → {new_path.name}")

    # 케이스 2: "xxx.NEF의 사본 (1)" 같은 경우
    elif ".NEF의 사본" in name:
        stem = name.replace(".NEF의 사본", "")
        new_name = stem + ".NEF"
        new_path = FOLDER / new_name

        if new_path.exists():
            new_path = FOLDER / (stem + "_copy.NEF")

        p.rename(new_path)
        print(f"FIX: {name} → {new_path.name}")
