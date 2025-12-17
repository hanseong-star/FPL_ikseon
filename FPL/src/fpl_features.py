# src/features.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


# -----------------------------
# Helpers
# -----------------------------
def _to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """
    Accepts either:
      - uint8 BGR image in [0,255]
      - float image in [0,1] (common in ML pipelines)
      - float image in [0,255]
    Returns uint8 BGR in [0,255].
    """
    if img is None:
        raise ValueError("Input image is None")

    if img.dtype == np.uint8:
        return img

    x = img.astype(np.float32, copy=False)
    mx = float(np.nanmax(x)) if x.size else 0.0

    # Heuristic: if it looks like [0,1], scale up
    if mx <= 1.5:
        x = x * 255.0

    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x


def _safe_resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid resize size: {size_wh}")
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _hist_norm(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32, copy=False)
    s = h.sum()
    return (h / (s + 1e-6)).astype(np.float32, copy=False)


# -----------------------------
# Existing 3x3 extractors (kept)
# -----------------------------
def extract_color_hs_3x3(images, h_bins=30, s_bins=32):
    """
    HSV에서 H, S만 사용.
    3x3 분할 후 각 patch의 H hist + S hist를 concat -> 전체 concat.
    반환: (N, D_color)
    """
    feats_all = []

    for img in images:
        img_u8 = _to_uint8_bgr(img)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)

        H, W = hsv.shape[:2]
        ph, pw = H // 3, W // 3

        feats = []
        for r in range(3):
            for c in range(3):
                patch = hsv[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]

                h_hist = np.histogram(patch[:, :, 0], bins=h_bins, range=(0, 180))[0]
                s_hist = np.histogram(patch[:, :, 1], bins=s_bins, range=(0, 256))[0]

                h_hist = _hist_norm(h_hist)
                s_hist = _hist_norm(s_hist)

                feats.append(np.concatenate([h_hist, s_hist], axis=0))

        feats_all.append(np.concatenate(feats, axis=0))

    return np.array(feats_all, dtype=np.float32)


def extract_hog_3x3(
    images,
    hog_size=(128, 128),
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
):
    """
    3x3 분할 후 각 patch -> gray -> hog_size로 resize -> HOG 추출 -> concat.
    반환: (N, D_hog)
    """
    feats_all = []

    for img in images:
        img_u8 = _to_uint8_bgr(img)
        H, W = img_u8.shape[:2]
        ph, pw = H // 3, W // 3

        feats = []
        for r in range(3):
            for c in range(3):
                patch = img_u8[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                gray = _safe_resize(gray, (hog_size[0], hog_size[1]))  # hog_size is (W,H)

                hf = hog(
                    gray,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    block_norm="L2-Hys",
                    transform_sqrt=True,
                    feature_vector=True,
                )
                feats.append(hf.astype(np.float32, copy=False))

        feats_all.append(np.concatenate(feats, axis=0))

    return np.array(feats_all, dtype=np.float32)


# -----------------------------
# NEW: Full-image (no 3x3) extractors
# -----------------------------
def extract_color_hs_full(
    images,
    h_bins=30,
    s_bins=32,
    sizes: tuple[tuple[int, int] | None, ...] | None = None,
):
    """
    HSV에서 H, S만 사용 (전체 이미지 기반, 3x3 분할 없음)

    - sizes=None 이면 "현재 이미지 크기"에서 1번만 추출
    - sizes=((256,256),(128,128))처럼 주면 멀티스케일로 각각 추출 후 concat
    - sizes에 None을 섞으면 (None,(256,256)) 처럼 '원본 + 리사이즈' 같이 가능

    반환: (N, D_color_full)
    """
    feats_all = []

    for img in images:
        img_u8 = _to_uint8_bgr(img)

        feats = []
        size_list = sizes if sizes is not None else (None,)

        for sz in size_list:
            im = _safe_resize(img_u8, sz) if (sz is not None) else img_u8
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            h_hist = np.histogram(hsv[:, :, 0], bins=h_bins, range=(0, 180))[0]
            s_hist = np.histogram(hsv[:, :, 1], bins=s_bins, range=(0, 256))[0]

            feats.append(np.concatenate([_hist_norm(h_hist), _hist_norm(s_hist)], axis=0))

        feats_all.append(np.concatenate(feats, axis=0))

    return np.array(feats_all, dtype=np.float32)


def extract_hog_full(
    images,
    hog_sizes: tuple[tuple[int, int] | None, ...] | None = ((256, 256), (128, 128)),  # (W,H) list/tuple or None
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
):
    """
    전체 이미지 기반 HOG (3x3 분할 없음)

    - hog_sizes=None 이면 "현재 이미지 크기"에서 HOG 1번만 추출 (리사이즈 없음)
    - hog_sizes=((258,258),(126,126)) 처럼 원하는 해상도들 지정 가능
    - hog_sizes에 None을 섞어서 (None,(256,256)) 처럼 '원본 + 리사이즈' 가능

    반환: (N, D_hog_full)
    """
    feats_all = []

    for img in images:
        img_u8 = _to_uint8_bgr(img)
        feats = []

        size_list = hog_sizes if hog_sizes is not None else (None,)

        for sz in size_list:
            im = _safe_resize(img_u8, sz) if (sz is not None) else img_u8
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            hf = hog(
                gray,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                transform_sqrt=True,
                feature_vector=True,
            )
            feats.append(hf.astype(np.float32, copy=False))

        feats_all.append(np.concatenate(feats, axis=0))

    return np.array(feats_all, dtype=np.float32)


def extract_lbp_full(
    images,
    resize: tuple[int, int] | None = None,  # (W,H) or None
    P=24,
    R=3,
    method="uniform",
):
    """
    전체 이미지 기반 LBP 히스토그램 (3x3 분할 없음)

    - resize=None 이면 리사이즈 없이 "현재 이미지 크기"에서 계산
    - resize=(W,H)를 주면 LBP 계산 전에 리사이즈

    반환: (N, D_lbp)
    - method='uniform'이면 보통 D = P + 2 (예: P=24 -> 26)
    """
    feats_all = []

    for img in images:
        img_u8 = _to_uint8_bgr(img)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

        if resize is not None:
            gray = _safe_resize(gray, resize)

        lbp = local_binary_pattern(gray, P=P, R=R, method=method)

        if method == "uniform":
            n_bins = int(P + 2)
        else:
            n_bins = int(lbp.max() + 1)

        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        feats_all.append(_hist_norm(hist))

    return np.array(feats_all, dtype=np.float32)


# -----------------------------
# NEW: Convenience concatenation (HOG + Color(HS) + LBP)
# -----------------------------
def extract_all_full_concat(
    images,
    hog_sizes: tuple[tuple[int, int] | None, ...] | None = ((256, 256), (128, 128)),
    hog_orientations=9,
    hog_pixels_per_cell=(8, 8),
    hog_cells_per_block=(2, 2),
    h_bins=60,
    s_bins=64,
    color_sizes: tuple[tuple[int, int] | None, ...] | None = None,  # None = no resize (use current)
    use_lbp=True,
    lbp_resize: tuple[int, int] | None = None,  # None = no resize (use current)
    lbp_P=24,
    lbp_R=3,
    lbp_method="uniform",
):
    """
    전체 이미지 기반으로
      X = [HOG(multi-scale) | HS hist(optional multi-scale) | LBP hist(optional)]

    - resize가 이미 되어있다면:
        hog_sizes=None (또는 (None,))
        color_sizes=None
        lbp_resize=None
      로 두면 "추가 리사이즈 없이" 현재 크기 그대로 피처를 뽑음.

    반환: (N, D_total)
    """
    X_hog = extract_hog_full(
        images,
        hog_sizes=hog_sizes,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
    )
    X_color = extract_color_hs_full(
        images,
        h_bins=h_bins,
        s_bins=s_bins,
        sizes=color_sizes,
    )

    if use_lbp:
        X_lbp = extract_lbp_full(
            images,
            resize=lbp_resize,
            P=lbp_P,
            R=lbp_R,
            method=lbp_method,
        )
        return np.concatenate([X_hog, X_color, X_lbp], axis=1).astype(np.float32, copy=False)

    return np.concatenate([X_hog, X_color], axis=1).astype(np.float32, copy=False)