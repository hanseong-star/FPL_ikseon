from pathlib import Path
import pandas as pd
import subprocess

CSV_PATH = Path("/home/hanseong/gdrive/ML_FPL_test_data/labels.csv")
RAW_DIR  = Path("/home/hanseong/gdrive/ML_FPL_test_data/raw")
JPG_DIR  = Path("/home/hanseong/gdrive/ML_FPL_test_data/jpg")

RAW_SUFFIXES = {".nef", ".NEF"}  # ✅ 둘 다 허용

def get_gps(path: Path):
    try:
        out = subprocess.check_output(
            ["exiftool", "-n", "-GPSLatitude", "-GPSLongitude", str(path)],
            stderr=subprocess.DEVNULL
        ).decode(errors="ignore")

        lat = lon = ""
        for line in out.splitlines():
            if "GPS Latitude" in line:
                lat = line.split(":", 1)[1].strip()
            elif "GPS Longitude" in line:
                lon = line.split(":", 1)[1].strip()
        return lat, lon
    except Exception:
        return "", ""

def build_raw_index():
    """
    RAW_DIR 아래(하위폴더 포함) 모든 NEF를 찾아 stem -> path로 맵핑
    """
    raw_map = {}
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix in RAW_SUFFIXES:
            raw_map[p.stem] = p
    return raw_map

def main():
    # ✅ photo_id 앞 0 보존
    df = pd.read_csv(CSV_PATH, dtype={"photo_id": str})

    if "gps_lat" not in df.columns:
        df["gps_lat"] = ""
    if "gps_lon" not in df.columns:
        df["gps_lon"] = ""

    raw_map = build_raw_index()
    print(f"RAW 인덱스 개수: {len(raw_map)}")

    # 디버그: JPG 샘플
    sample_jpg = sorted([p.name for p in JPG_DIR.glob("*.jpg")])[:3]
    print("JPG 샘플:", sample_jpg)

    filled = 0
    miss_raw = 0

    for idx, row in df.iterrows():
        if str(row["gps_lat"]).strip() and str(row["gps_lon"]).strip():
            continue

        photo_id = str(row["photo_id"]).strip().zfill(4)

        jpgs = list(JPG_DIR.glob(f"*_{photo_id}.jpg"))
        if not jpgs:
            continue

        jpg = jpgs[0]
        stem = jpg.stem  # 예: donhwamunro_11_da_A_raw_0914

        # ✅ 1순위: JPG stem 그대로
        candidates = [stem]

        # ✅ 2순위: RAW 파일명엔 _raw_ 가 없을 수도 있음
        candidates.append(stem.replace("_raw_", "_"))

        # ✅ 3순위: RAW 파일명이 raw_0914 형태일 수도 있음
        candidates.append(f"raw_{photo_id}")

        nef_path = None
        for s in candidates:
            if s in raw_map:
                nef_path = raw_map[s]
                break

        if nef_path is None:
            miss_raw += 1
            # 디버그(너무 많이 찍히면 주석처리)
            # print(f"[MISS RAW] {jpg.name}  candidates={candidates[:3]}")
            continue

        lat, lon = get_gps(nef_path)
        if lat and lon:
            df.at[idx, "gps_lat"] = lat
            df.at[idx, "gps_lon"] = lon
            filled += 1

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"✅ GPS 보완 완료: {filled}개 행 업데이트")
    print(f"❗ RAW 매칭 실패: {miss_raw}개 행")
    print(f"📄 CSV 저장: {CSV_PATH}")

if __name__ == "__main__":
    main()
