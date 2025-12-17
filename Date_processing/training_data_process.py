from pathlib import Path
import rawpy
import imageio

# =========================
# 경로 설정
# =========================

BASE_DIR = Path("/home/hanseong/gdrive/ML_FPL_training_data")

RAW_DIR = BASE_DIR / "raw"
JPG_DIR = BASE_DIR / "jpg"

RAW_DIR.mkdir(parents=True, exist_ok=True)
JPG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# JPEG 설정
# =========================

JPEG_QUALITY = 95
RAW_EXTS = [".nef", ".cr2", ".cr3", ".arw", ".raw"]

def is_raw_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in RAW_EXTS

# =========================
# 1️⃣ RAW 파일을 raw/ 폴더로 이동
# =========================
def collect_raw_files():
    moved = 0

    for p in BASE_DIR.iterdir():
        if is_raw_file(p):
            target = RAW_DIR / p.name

            # 이름 충돌 방지
            if target.exists():
                idx = 1
                while True:
                    target = RAW_DIR / f"{p.stem}_dup{idx}{p.suffix.lower()}"
                    if not target.exists():
                        break
                    idx += 1

            p.rename(target)
            moved += 1
            print(f"[MOVE] {p.name} → raw/{target.name}")

    print(f"RAW 정리 완료: {moved}개 이동")

# =========================
# 2️⃣ RAW → JPG 변환
# =========================
def convert_raw_to_jpeg(raw_path: Path, delete_raw: bool = False):
    jpg_path = JPG_DIR / (raw_path.stem + ".jpg")

    if jpg_path.exists():
        print(f"[SKIP] {jpg_path.name} 이미 존재")
        return

    print(f"[CONVERT] {raw_path.name} → {jpg_path.name}")

    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            bright=1.5,
            gamma=(2.2, 4.5),
            output_bps=8
        )

    imageio.imwrite(str(jpg_path), rgb, quality=JPEG_QUALITY)
    print(f"   → 저장 완료: jpg/{jpg_path.name}")

    if delete_raw:
        raw_path.unlink()
        print(f"   → RAW 삭제: {raw_path.name}")

# =========================
# main
# =========================
def main():
    # 1️⃣ RAW 정리
    collect_raw_files()

    # 2️⃣ 변환
    raw_files = sorted([f for f in RAW_DIR.iterdir() if is_raw_file(f)])
    print(f"총 RAW 파일 개수: {len(raw_files)}")

    for i, raw_path in enumerate(raw_files, start=1):
        print(f"[{i}/{len(raw_files)}]")
        convert_raw_to_jpeg(raw_path, delete_raw=False)

    print("✅ RAW → JPG 변환 완료")

if __name__ == "__main__":
    main()
