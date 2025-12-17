import rawpy
import imageio

# 원본 RAW 파일 경로
raw_path = "/home/hanseong/gdrive/ML_FPL_test_data/donhwamunro_11_134.NEF"

# 저장할 JPG 경로
jpg_path = "/home/hanseong/gdrive/ML_FPL_test_data/donhwamunro_11_134.jpg"

with rawpy.imread(raw_path) as raw:
    rgb = raw.postprocess(
        use_camera_wb=True,     # 카메라 화이트밸런스 사용
        no_auto_bright=False,   # 자동 밝기 보정 ON (False면 켜짐)
        bright=1.5,             # 전체 밝기 조금 올리기 (1.0 기본)
        gamma=(2.2, 4.5),       # sRGB에 가까운 톤
        output_bps=8            # JPG용 8bit
    )

imageio.imwrite(jpg_path, rgb, quality=95)
print("변환 완료:", jpg_path)