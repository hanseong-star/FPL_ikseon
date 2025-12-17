import os
import glob
import cv2
import pandas as pd

# =========================
# 설정 (여기만 너 환경에 맞게 수정)
# =========================
CSV_PATH = "/home/hanseong/gdrive/ML_FPL_training_data/training_labels.csv"   # 예: training_labels.csv 경로로 바꿔도 됨
IMG_DIR  = "/home/hanseong/gdrive/ML_FPL_training_data/jpg"               # 이미지 폴더 (jpg/png 등)
ID_COL   = "photo_id"                      # CSV에서 이미지 id 컬럼명 (너 csv에 맞게 수정!)
OUT_CSV  = "/home/hanseong/gdrive/ML_FPL_training_data/training_labels_plus.csv"  # 결과 저장 경로

MAX_W, MAX_H = 2000, 1300


VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def list_images(img_dir):
    # raw_가 들어간 파일 전부(확장자 무관) 찾기
    img_list = glob.glob(os.path.join(img_dir, "*raw_*.*")) + glob.glob(os.path.join(img_dir, "*RAW_*.*"))
    img_list = [p for p in img_list if p.lower().endswith(VALID_EXT)]
    return sorted(img_list)

def main():
    cv2.setNumThreads(0)

    df = pd.read_csv(CSV_PATH)
    # gps_lat/gps_lon을 x/y로 컬럼명 변경
    df.rename(columns={"gps_lat":"x", "gps_lon":"y"}, inplace=True)
    if "x" not in df.columns: df["x"] = pd.NA
    if "y" not in df.columns: df["y"] = pd.NA

    img_list = list_images(IMG_DIR)
    print("CSV rows:", len(df))
    print("Images :", len(img_list))

    n = min(len(df), len(img_list))
    if n == 0:
        print("[ERR] No pairs to label. Check IMG_DIR or filename pattern.")
        return
    if len(df) != len(img_list):
        print(f"[WARN] counts differ -> using first {n} pairs (min length).")

    cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)

    try:
        for i in range(n):
            img_path = img_list[i]
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERR] cannot read: {img_path}")
                continue

            disp, _ = resize_for_display(img)
            fn = os.path.basename(img_path)

            # 이미지 표시
            show = disp.copy()
            msg = f"{i+1}/{n}  {fn}"
            cv2.putText(show, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(show, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("annotator", show)
            cv2.waitKey(1)  # 창 업데이트

            # 기존 값 보여주기
            prev_x, prev_y = df.loc[i, "x"], df.loc[i, "y"]
            print("\n" + "="*60)
            print(f"[{i+1}/{n}] {fn}")
            print(f"Current saved x,y = ({prev_x}, {prev_y})")
            print("Input format: x y   (e.g., 812 435)")
            print("Enter = skip | s = skip | q = quit")

            user = input("x y > ").strip().lower()

            if user in ("q", "quit", "exit"):
                df.to_csv(OUT_CSV, index=False)
                print(f"[SAVE&QUIT] {OUT_CSV}")
                return

            if user == "" or user in ("s", "skip"):
                # 스킵
                continue

            parts = user.split()
            if len(parts) != 2:
                print("[WARN] Please type exactly two numbers: x y (or Enter/s/q)")
                continue

            try:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
            except ValueError:
                print("[WARN] Invalid numbers. Try again.")
                continue

            df.loc[i, "x"] = x
            df.loc[i, "y"] = y
            df.to_csv(OUT_CSV, index=False)
            print(f"[SAVE] row={i} -> (x,y)=({x},{y})  saved to: {OUT_CSV}")

    finally:
        cv2.destroyAllWindows()
        df.to_csv(OUT_CSV, index=False)
        print(f"[DONE] {OUT_CSV}")

if __name__ == "__main__":
    main()