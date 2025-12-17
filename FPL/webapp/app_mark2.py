#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import argparse
import numpy as np
import joblib

import cv2
from skimage.feature import hog


# ---------------------------
# Paths (same rule as trainers)
# ---------------------------
def _detail_root(out_dir: str, feature_tag: str):
    feature_tag = (feature_tag or "").strip()
    if feature_tag in ("", "legacy"):
        return os.path.join(out_dir, "detail_models")
    return os.path.join(out_dir, f"detail_models_{feature_tag}")

def _knn_root(out_dir: str, feature_tag: str):
    feature_tag = (feature_tag or "").strip()
    if feature_tag in ("", "legacy"):
        return os.path.join(out_dir, "knn_models")
    return os.path.join(out_dir, f"knn_models_{feature_tag}")


# ---------------------------
# Full feature extractors (NO 3x3 split)
# ---------------------------
def extract_color_hs_full_bgr(img_bgr, h_bins=60, s_bins=64):
    """HSV(H,S) 2D histogram -> flatten, L1 normalize"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV H range: [0,180), S range: [0,256)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist  # (h_bins*s_bins,)

def extract_hog_full_bgr(img_bgr, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """HOG on gray image (same params you used for full)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return feat.astype(np.float32)  # (D,)


# ---------------------------
# Model loader helpers
# ---------------------------
def _load_if_exists(path):
    return joblib.load(path) if os.path.exists(path) else None

def load_road_models(model_dir: str, best_dim: int):
    """
    기대 파일들(너가 저장하던 방식 기준):
      - hog_scaler_dim{best_dim}.pkl
      - hog_pca_dim{best_dim}.pkl
      - hog_svm_dim{best_dim}.pkl
      - color_svm.pkl
      - fusion_lr_dim{best_dim}.pkl  (있으면 사용)
      - (선택) color_scaler.pkl  (있으면 color feature transform)
    """
    md = model_dir
    hog_scaler = joblib.load(os.path.join(md, f"hog_scaler_dim{best_dim}.pkl"))
    hog_pca    = joblib.load(os.path.join(md, f"hog_pca_dim{best_dim}.pkl"))
    hog_svm    = joblib.load(os.path.join(md, f"hog_svm_dim{best_dim}.pkl"))

    color_svm  = joblib.load(os.path.join(md, "color_svm.pkl"))

    fusion_lr  = _load_if_exists(os.path.join(md, f"fusion_lr_dim{best_dim}.pkl"))
    color_scaler = _load_if_exists(os.path.join(md, "color_scaler.pkl"))  # optional

    return {
        "hog_scaler": hog_scaler,
        "hog_pca": hog_pca,
        "hog_svm": hog_svm,
        "color_svm": color_svm,
        "fusion_lr": fusion_lr,
        "color_scaler": color_scaler,
    }

def load_knn_global(model_dir: str, feature_tag: str):
    """
    knn_models_<tag>/ 안에 저장된 전역 HOG scaler/PCA 로드
      - knn_hog_scaler_pca{n_comp}.pkl
      - knn_hog_pca_pca{n_comp}.pkl
    n_comp는 파일명에서 자동 탐색.
    """
    root = _knn_root(model_dir, feature_tag)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"KNN root not found: {root}")

    scaler_files = sorted(glob.glob(os.path.join(root, "knn_hog_scaler_pca*.pkl")))
    pca_files    = sorted(glob.glob(os.path.join(root, "knn_hog_pca_pca*.pkl")))
    if not scaler_files or not pca_files:
        raise FileNotFoundError(f"KNN scaler/pca not found under: {root}")

    # 보통 1개만 존재. 여러 개면 가장 마지막(가장 큰 dim) 사용
    scaler_path = scaler_files[-1]
    pca_path    = pca_files[-1]

    scaler = joblib.load(scaler_path)
    pca    = joblib.load(pca_path)

    return {"knn_root": root, "scaler": scaler, "pca": pca, "scaler_path": scaler_path, "pca_path": pca_path}

def _try_fusion_predict(fusion_lr, hog_svm, color_svm, Xh_p, Xc):
    """
    fusion_lr가 있을 때, 학습 당시 입력 feature 형태가 뭔지 모를 수 있어서
    n_features_in_에 맞춰 후보를 자동 매칭.
    """
    if fusion_lr is None:
        return None

    n_in = getattr(fusion_lr, "n_features_in_", None)
    if n_in is None:
        return None

    # 후보들 생성
    hog_proba   = hog_svm.predict_proba(Xh_p)[0]
    color_proba = color_svm.predict_proba(Xc)[0]

    cands = []

    # 1) [hog_proba | color_proba]
    cands.append(np.hstack([hog_proba, color_proba]))

    # 2) decision_function (ovo면 차원이 애매할 수 있음)
    try:
        hog_dec = np.ravel(hog_svm.decision_function(Xh_p))
        col_dec = np.ravel(color_svm.decision_function(Xc))
        cands.append(np.hstack([hog_dec, col_dec]))
        cands.append(np.hstack([hog_proba, color_proba, hog_dec, col_dec]))
    except Exception:
        pass

    # 3) 단일 proba만
    cands.append(hog_proba)
    cands.append(color_proba)

    for v in cands:
        v = np.asarray(v, dtype=np.float32).reshape(1, -1)
        if v.shape[1] == n_in:
            proba = fusion_lr.predict_proba(v)[0]
            return proba

    return None


# ---------------------------
# Predict (road -> detail -> knn)
# ---------------------------
DETAIL_ROADS_DEFAULT = {
    "donhwamunro_11_ga",
    "donhwamunro_11_na",
    "donhwamunro_11_da",
    "suporo_28",
}

def predict_one_image(
    image_path: str,
    model_dir: str,
    feature_tag: str,
    best_dim: int,
    alpha_fallback: float = 0.5,
    h_bins: int = 60,
    s_bins: int = 64,
    hog_orient: int = 12,
    hog_ppc=(8, 8),
    hog_cpb=(2, 2),
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # ---- full features
    x_color = extract_color_hs_full_bgr(img, h_bins=h_bins, s_bins=s_bins)
    x_hog   = extract_hog_full_bgr(img, orientations=hog_orient, pixels_per_cell=hog_ppc, cells_per_block=hog_cpb)

    Xc = x_color.reshape(1, -1)
    Xh = x_hog.reshape(1, -1)

    # ---- road models
    road_models = load_road_models(model_dir, best_dim)
    hog_scaler = road_models["hog_scaler"]
    hog_pca    = road_models["hog_pca"]
    hog_svm    = road_models["hog_svm"]
    color_svm  = road_models["color_svm"]
    fusion_lr  = road_models["fusion_lr"]
    color_scaler = road_models["color_scaler"]

    if color_scaler is not None:
        Xc_road = color_scaler.transform(Xc)
    else:
        Xc_road = Xc

    Xh_s = hog_scaler.transform(Xh)
    Xh_p = hog_pca.transform(Xh_s)

    # ---- road predict
    hog_proba   = hog_svm.predict_proba(Xh_p)[0]
    color_proba = color_svm.predict_proba(Xc_road)[0]

    fusion_proba = _try_fusion_predict(fusion_lr, hog_svm, color_svm, Xh_p, Xc_road)
    if fusion_proba is None:
        fusion_proba = alpha_fallback * hog_proba + (1.0 - alpha_fallback) * color_proba

    road_classes = hog_svm.classes_
    road_idx = int(np.argmax(fusion_proba))
    road_pred = str(road_classes[road_idx])
    road_conf = float(fusion_proba[road_idx])

    # ---- detail predict (if road is detail road)
    detail_pred = "0"
    detail_conf = 0.0
    used_detail = False

    if road_pred in DETAIL_ROADS_DEFAULT:
        droot = _detail_root(model_dir, feature_tag)
        road_dir = os.path.join(droot, road_pred)

        # 디테일 모델 폴더 없으면 스킵
        if os.path.isdir(road_dir):
            detail_color_svm = joblib.load(os.path.join(road_dir, "detail_color_svm.pkl"))
            detail_hog_svm   = joblib.load(os.path.join(road_dir, "detail_hog_svm.pkl"))
            detail_scaler    = joblib.load(os.path.join(road_dir, "detail_hog_scaler.pkl"))
            detail_pca       = joblib.load(os.path.join(road_dir, "detail_hog_pca.pkl"))
            label_map        = joblib.load(os.path.join(road_dir, "detail_label_map.pkl"))

            meta_path = os.path.join(road_dir, "meta.json")
            alpha_detail = alpha_fallback
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                alpha_detail = float(meta.get("alpha_shape_for_runtime_fusion", alpha_fallback))

            # color은 detail 쪽도 "훈련 때 넣어준 그대로"를 써야 함
            # (훈련 때 Xc_tr를 넣었으면 여기서도 같은 전처리가 필요)
            Xc_detail = Xc  # 여기서 별도 scaler를 쓰려면 detail 학습과 동일하게 맞춰줘야 함

            Xhd_s = detail_scaler.transform(Xh)
            Xhd_p = detail_pca.transform(Xhd_s)

            dp_color = detail_color_svm.predict_proba(Xc_detail)[0]
            dp_hog   = detail_hog_svm.predict_proba(Xhd_p)[0]
            dp_fused = alpha_detail * dp_hog + (1.0 - alpha_detail) * dp_color

            det_idx = int(np.argmax(dp_fused))
            det_conf = float(dp_fused[det_idx])

            # reverse map: idx -> detail string
            rev = {v: k for k, v in label_map.items()}
            detail_pred = str(rev.get(det_idx, "0"))
            detail_conf = det_conf
            used_detail = True

    # ---- KNN (x,y)
    knn_global = load_knn_global(model_dir, feature_tag)
    knn_root = knn_global["knn_root"]
    knn_scaler = knn_global["scaler"]
    knn_pca    = knn_global["pca"]

    # KNN feature: [HOG(PCA with KNN scaler/pca) + Color]
    Xhk_s = knn_scaler.transform(Xh)
    Xhk_p = knn_pca.transform(Xhk_s)
    Z = np.hstack([Xhk_p, Xc])  # KNN 학습 때 X_color_train을 뭐로 넣었는지에 따라 여기 전처리가 동일해야 함!

    road_knn_dir = os.path.join(knn_root, road_pred)
    knn_level = "road"
    x_pred, y_pred = np.nan, np.nan

    if os.path.isdir(road_knn_dir):
        # detail 모델이 있으면 detail KNN 우선
        det_knn_path = os.path.join(road_knn_dir, f"knn_detail_{detail_pred}.pkl")
        road_knn_path = os.path.join(road_knn_dir, "knn_road.pkl")

        if detail_pred != "0" and os.path.exists(det_knn_path):
            knn = joblib.load(det_knn_path)
            knn_level = f"detail_{detail_pred}"
        elif os.path.exists(road_knn_path):
            knn = joblib.load(road_knn_path)
            knn_level = "road"
        else:
            knn = None

        if knn is not None:
            xy = knn.predict(Z)[0]
            x_pred, y_pred = float(xy[0]), float(xy[1])

    return {
        "image_path": image_path,
        "feature_tag": feature_tag,
        "best_dim": int(best_dim),

        "road_pred": road_pred,
        "road_conf": road_conf,

        "detail_pred": detail_pred,
        "detail_conf": detail_conf,
        "used_detail": bool(used_detail),

        "x_pred": x_pred,
        "y_pred": y_pred,
        "knn_level": knn_level,
        "knn_root": knn_root,
    }


# ---------------------------
# CLI
# ---------------------------
def is_image_file(p):
    ext = os.path.splitext(p)[1].lower()
    return ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="FPL_models")
    ap.add_argument("--feature_tag", type=str, default="full")  # ✅ full 모델 쓰려면 "full"
    ap.add_argument("--best_dim", type=int, default=128)

    ap.add_argument("--alpha", type=float, default=0.5)  # fusion_lr 없을 때만 사용

    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--image_dir", type=str, default="")

    ap.add_argument("--out_json", type=str, default="")   # 단일 이미지 결과 저장
    ap.add_argument("--out_csv", type=str, default="")    # 폴더 처리 결과 저장

    args = ap.parse_args()

    if not args.image and not args.image_dir:
        print("❗ --image 또는 --image_dir 중 하나는 필요합니다.")
        sys.exit(1)

    if args.image:
        pred = predict_one_image(
            args.image,
            model_dir=args.model_dir,
            feature_tag=args.feature_tag,
            best_dim=args.best_dim,
            alpha_fallback=args.alpha,
        )
        print(json.dumps(pred, ensure_ascii=False, indent=2))

        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(pred, f, ensure_ascii=False, indent=2)
            print("✅ saved:", args.out_json)
        return

    # dir mode
    paths = []
    for p in sorted(glob.glob(os.path.join(args.image_dir, "*"))):
        if is_image_file(p):
            paths.append(p)

    if not paths:
        print("❗ image_dir에 이미지가 없습니다:", args.image_dir)
        sys.exit(1)

    rows = []
    for p in paths:
        try:
            pred = predict_one_image(
                p,
                model_dir=args.model_dir,
                feature_tag=args.feature_tag,
                best_dim=args.best_dim,
                alpha_fallback=args.alpha,
            )
            rows.append(pred)
            print(f"[OK] {os.path.basename(p)} -> road={pred['road_pred']} detail={pred['detail_pred']} xy=({pred['x_pred']:.3f},{pred['y_pred']:.3f})")
        except Exception as e:
            print(f"[ERR] {p} : {e}")
            rows.append({"image_path": p, "error": str(e)})

    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print("✅ saved:", args.out_csv)
    else:
        print("✅ done. (원하면 --out_csv 결과 저장)")

if __name__ == "__main__":
    main()
