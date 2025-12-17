from pathlib import Path
import subprocess
import csv
import re

# =========================
# 설정
# =========================
JPG_DIR = Path("/home/hanseong/gdrive/ML_FPL_training_data/jpg")
OUT_CSV = Path("/home/hanseong/gdrive/ML_FPL_training_data/traing_labels.csv")

# =========================
# GPS 추출
# =========================
def get_gps(path: Path):
    try:
        out = subprocess.check_output(
            ["exiftool", "-n", "-GPSLatitude", "-GPSLongitude", str(path)],
            stderr=subprocess.DEVNULL
        ).decode()

        lat = lon = ""
        for line in out.splitlines():
            if "GPS Latitude" in line:
                lat = line.split(":")[-1].strip()
            elif "GPS Longitude" in line:
                lon = line.split(":")[-1].strip()

        return lat, lon
    except Exception:
        return "", ""

# =========================
# 파일명 파싱
# =========================
def parse_filename(name: str):
    """
    return:
      photo_id (str),
      road_name (str),
      detail (str or "0")
    """

    stem = Path(name).stem
    parts = stem.split("_")

    # 마지막은 반드시 4자리 숫자
    photo_id = parts[-1]
    if not photo_id.isdigit() or len(photo_id) != 4:
        return None

    # 'raw' 위치 찾기
    if "raw" not in parts:
        return None

    raw_idx = parts.index("raw")

    # raw 앞부분만 도로명/세부도로명 후보
    core = parts[:raw_idx]

    detail = "0"
    if core and core[-1] in {"A", "B", "C", "D", "E"}:
        detail = core[-1]
        core = core[:-1]

    # 도로명 규칙
    if len(core) == 1:
        road = core[0]
    elif len(core) == 2:
        road = "_".join(core)
    else:
        road = "_".join(core[:3])

    return photo_id, road, detail


# =========================
# main
# =========================
def main():
    rows = []

    for img in sorted(JPG_DIR.iterdir()):
        if img.suffix.lower() != ".jpg":
            continue

        parsed = parse_filename(img.name)
        if parsed is None:
            continue

        photo_id, road, detail = parsed
        lat, lon = get_gps(img)

        rows.append([photo_id, road, detail, lat, lon])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_id", "road_name", "detail", "gps_lat", "gps_lon"])
        writer.writerows(rows)

    print(f"✅ CSV 생성 완료: {OUT_CSV}")
    print(f"총 {len(rows)}개 샘플")

if __name__ == "__main__":
    main()
