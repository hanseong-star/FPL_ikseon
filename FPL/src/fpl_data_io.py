# src/data_io.py
import os
import cv2
import numpy as np
import pandas as pd

def build_image_index(*root_dirs, exts=(".jpg", ".jpeg", ".JPG", ".JPEG")):
    """
    파일명 -> 전체 경로 dict 생성
    같은 파일명이 여러 곳에 있으면 마지막으로 발견한 경로로 덮어씀.
    """
    index = {}
    for root in root_dirs:
        for r, _, files in os.walk(root):
            for fn in files:
                if fn.endswith(exts):
                    index[fn] = os.path.join(r, fn)
    return index

def resize_keep_direction(img, landscape=(1024, 682), portrait=(682, 1024)):
    """
    가로(landscape)면 1024x682, 세로(portrait)면 682x1024로 리사이즈.
    """
    h, w = img.shape[:2]
    if w >= h:
        return cv2.resize(img, landscape, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, portrait, interpolation=cv2.INTER_AREA)

def make_filename(road: str, detail, photo_id: int) -> str:
    """
    너 파일명 규칙:
    - detail이 0이면: {road}_raw_{photo_id:04d}.jpg
    - detail이 0이 아니면: {road}_{detail}_raw_{photo_id:04d}.jpg
    """
    if pd.isna(detail) or str(detail).strip() == "0":
        return f"{road}_raw_{photo_id:04d}.jpg"
    return f"{road}_{str(detail).strip()}_raw_{photo_id:04d}.jpg"

def load_dataset(label_csv: str, image_index: dict, resize=True):
    """
    CSV를 읽어서 이미지와 메타데이터를 '같은 인덱스'로 묶어 반환.
    SVM 라벨은 road_name만(도로명만) 유지.
    x, y 좌표는 '사후 평가용 메타데이터'로만 함께 반환.
    """
    df = pd.read_csv(label_csv)

    images = []
    road_labels = []
    photo_ids = []
    details = []
    filenames = []

    xs = []   
    ys = []   

    missed = []

    for _, row in df.iterrows():
        road = str(row["road_name"])
        detail = row["detail"]
        photo_id = int(row["photo_id"])

        filename = make_filename(road, detail, photo_id)
        path = image_index.get(filename)

        if path is None:
            missed.append(filename)
            continue

        img = cv2.imread(path)
        if img is None:
            missed.append(filename)
            continue

        if resize:
            img = resize_keep_direction(img)

        img = img.astype(np.float32) / 255.0

        images.append(img)
        road_labels.append(road)
        photo_ids.append(photo_id)
        details.append(detail)
        filenames.append(filename)

        x = row["x"] if "x" in row and pd.notna(row["x"]) else None
        y = row["y"] if "y" in row and pd.notna(row["y"]) else None
        xs.append(x)
        ys.append(y)

    return {
        "images": images,
        "road_labels": np.array(road_labels, dtype=object),
        "photo_ids": np.array(photo_ids, dtype=np.int32),
        "details": np.array(details, dtype=object),
        "filenames": np.array(filenames, dtype=object),

        # ⭐ 추가된 반환값
        "xs": np.array(xs, dtype=np.float32),
        "ys": np.array(ys, dtype=np.float32),

        "missed": missed,
        "df": df,
    }

