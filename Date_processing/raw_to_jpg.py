import os
from pathlib import Path
import rawpy
import imageio
# RAW 파일이 있는 폴더
RAW_DIR = Path("/home/hanseong/gdrive/ML_FPL_test_data/raw")

# JPG를 저장할 폴더 
JPG_DIR = Path("/home/hanseong/gdrive/ML_FPL_test_data/jpg")
JPG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# JPEG 설정
# =========================

JPEG_QUALITY = 95

RAW_EXTS = [".nef", ".cr2", ".cr3", ".arw", ".raw"]

def is_raw_file(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS

def convert_raw_to_jpeg(raw_path: Path, delete_raw: bool = False):
    # RAW 파일명 그대로, 저장 위치만 JPG_DIR로
    jpg_path = JPG_DIR / (raw_path.stem + ".jpg")

    # 이미 JPEG가 있으면 스킵
    if jpg_path.exists():
        print(f"[SKIP] {jpg_path.name} 이미 존재")
        if delete_raw:
            raw_path.unlink(missing_ok=True)
            print(f"       (원본 삭제) {raw_path.name}")
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

    # JPG 저장
    imageio.imwrite(str(jpg_path), rgb, quality=JPEG_QUALITY)
    print(f"   → 저장 완료: {jpg_path}")

    if delete_raw:
        raw_path.unlink()
        print(f"   → RAW 삭제: {raw_path.name}")

def main():
    raw_files = [
        f for f in sorted(RAW_DIR.iterdir())
        if f.is_file() and is_raw_file(f)
    ]

    print(f"총 RAW 파일 개수: {len(raw_files)}")

    for i, raw_path in enumerate(raw_files, start=1):
        print(f"[{i}/{len(raw_files)}]")
        convert_raw_to_jpeg(raw_path, delete_raw=False)

if __name__ == "__main__":
    main()