from pathlib import Path
import subprocess
from PIL import Image, ImageEnhance

RAW_PATH = Path("/home/hanseong/gdrive/raw_2296.nef")  
OUT_DIR  = Path("/home/hanseong/gdrive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_preview(nef: Path, out_jpg: Path):
    # NEF 내 PreviewImage 추출 → JPG 저장
    cmd = ["exiftool", "-b", "-PreviewImage", str(nef)]
    data = subprocess.check_output(cmd)
    out_jpg.write_bytes(data)

def make_variant(src_jpg: Path, out_jpg: Path, brightness=1.0, warmth=0):
    """
    brightness: 1.0 기본, >1 밝게, <1 어둡게
    warmth: -20~+20 권장. +면 따뜻(오렌지), -면 차가움(블루)
    """
    img = Image.open(src_jpg).convert("RGB")

    # 밝기
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 색온도(간단 버전): R/B 채널 가감
    if warmth != 0:
        r, g, b = img.split()
        r = r.point(lambda x: max(0, min(255, x + warmth)))
        b = b.point(lambda x: max(0, min(255, x - warmth)))
        img = Image.merge("RGB", (r, g, b))

    img.save(out_jpg, quality=95)

origin = OUT_DIR / "preview_origin.jpg"
bright = OUT_DIR / "preview_bright.jpg"
dark   = OUT_DIR / "preview_dark.jpg"

extract_preview(RAW_PATH, origin)

# 🔧 여기 숫자만 바꾸면서 “아침/밤” 느낌 맞추면 됨
make_variant(origin, bright, brightness=1.50, warmth=+20)  # 아침: 밝게 + 따뜻
make_variant(origin, dark,   brightness=0.50, warmth=+15)  # 밤: 어둡게 + 차갑게

print("완료:", OUT_DIR)
