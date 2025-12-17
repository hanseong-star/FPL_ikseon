#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib
import sys

ROOT = Path.cwd().resolve()   # 보통 FPL 폴더에서 노트북 실행중이면 이게 루트
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 너 repo 함수 그대로 사용 (3x3)
from src.fpl_features import extract_color_hs_3x3, extract_hog_3x3


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pretty_probs(probs: np.ndarray, classes) -> dict:
    probs = np.asarray(probs).reshape(-1)
    return {str(c): float(p) for c, p in zip(list(classes), probs)}


def top1(prob_dict: dict | None):
    if not prob_dict:
        return None
    k = max(prob_dict, key=prob_dict.get)
    return {"pred": k, "prob": prob_dict[k]}


def load_label_map(label_map_path: Path):
    """
    road_label_map.pkl 이 어떤 형식인지 확실치 않아서,
    dict 형태면 그대로 쓰고,
    아니면 None 반환.
    """
    if not label_map_path.exists():
        return None
    lm = joblib.load(str(label_map_path))
    if isinstance(lm, dict):
        return lm
    return None


def apply_label_map(classes, label_map: dict | None):
    """
    classes_가 숫자 라벨일 때, {int: str} 형태 맵이 있으면 사람이 읽는 이름으로 변환.
    """
    if label_map is None:
        return [str(c) for c in list(classes)]
    out = []
    for c in list(classes):
        try:
            key = int(c)
        except Exception:
            out.append(str(c))
            continue
        out.append(str(label_map.get(key, c)))
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--out_dir", default="one_image_outputs", help="output directory")

    ap.add_argument("--model_dir", default="FPL_models", help="dir containing saved .pkl models")
    ap.add_argument("--best_dim", type=int, required=True, help="PCA dim used in saved hog models")

    # 기록용 리사이즈(학습과 무관, 그냥 눈으로 보기 좋게 저장)
    ap.add_argument("--save_resize_w", type=int, default=640)
    ap.add_argument("--save_resize_h", type=int, default=480)

    # === 3x3 COLOR params (학습과 동일) ===
    ap.add_argument("--h_bins", type=int, default=30)
    ap.add_argument("--s_bins", type=int, default=32)

    # === 3x3 HOG params (학습과 동일) ===
    ap.add_argument("--hog_w", type=int, default=128, help="hog_size width")
    ap.add_argument("--hog_h", type=int, default=128, help="hog_size height")
    ap.add_argument("--hog_ori", type=int, default=9)
    ap.add_argument("--hog_ppc", type=int, default=8)
    ap.add_argument("--hog_cpb", type=int, default=2)

    # LR fusion 입력 순서: 네 코드에서 X_fuse = [P_shape, P_color]
    ap.add_argument("--fusion_order", default="shape,color", choices=["shape,color", "color,shape"])

    args = ap.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    model_dir = Path(args.model_dir)
    best_dim = args.best_dim

    # -------------------------
    # Load image
    # -------------------------
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # -------------------------
    # Save resized image (record)
    # -------------------------
    resized = cv2.resize(bgr, (args.save_resize_w, args.save_resize_h), interpolation=cv2.INTER_AREA)
    resized_path = out_dir / f"{img_path.stem}_resized_{args.save_resize_w}x{args.save_resize_h}.jpg"
    cv2.imwrite(str(resized_path), resized)

    # -------------------------
    # 1) Extract 3x3 features (exactly as training)
    # -------------------------
    X_color = extract_color_hs_3x3([bgr], h_bins=args.h_bins, s_bins=args.s_bins)   # (1, D)
    X_hog   = extract_hog_3x3(
        [bgr],
        hog_size=(args.hog_w, args.hog_h),
        orientations=args.hog_ori,
        pixels_per_cell=(args.hog_ppc, args.hog_ppc),
        cells_per_block=(args.hog_cpb, args.hog_cpb),
    )  # (1, D)

    color_feat = X_color[0].astype(np.float32, copy=False)
    hog_feat   = X_hog[0].astype(np.float32, copy=False)

    # 저장
    color_feat_path = out_dir / f"{img_path.stem}_color3x3_HS_{args.h_bins}x{args.s_bins}.npy"
    hog_feat_path   = out_dir / f"{img_path.stem}_hog3x3_{args.hog_w}x{args.hog_h}_ori{args.hog_ori}.npy"
    np.save(str(color_feat_path), color_feat)
    np.save(str(hog_feat_path), hog_feat)

    # -------------------------
    # 2) Load models
    # -------------------------
    color_svm_path = model_dir / "color_svm.pkl"
    hog_svm_path   = model_dir / f"hog_svm_dim{best_dim}.pkl"
    hog_scaler_path= model_dir / f"hog_scaler_dim{best_dim}.pkl"
    hog_pca_path   = model_dir / f"hog_pca_dim{best_dim}.pkl"
    fusion_lr_path = model_dir / f"fusion_lr_dim{best_dim}.pkl"
    label_map_path = model_dir / "road_label_map.pkl"

    for p in [color_svm_path, hog_svm_path, hog_scaler_path, hog_pca_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")

    color_svm  = joblib.load(str(color_svm_path))
    hog_svm    = joblib.load(str(hog_svm_path))
    hog_scaler = joblib.load(str(hog_scaler_path))
    hog_pca    = joblib.load(str(hog_pca_path))

    if not hasattr(color_svm, "predict_proba"):
        raise ValueError("color_svm has no predict_proba(). Train with probability=True.")
    if not hasattr(hog_svm, "predict_proba"):
        raise ValueError("hog_svm has no predict_proba(). Train with probability=True.")

    label_map = load_label_map(label_map_path)

    # -------------------------
    # 3) Predict probabilities
    # -------------------------
    # Color
    P_color = color_svm.predict_proba(X_color)[0]  # (K,)
    color_classes_raw = getattr(color_svm, "classes_", list(range(len(P_color))))
    color_classes = apply_label_map(color_classes_raw, label_map)

    # HOG pipeline: scaler -> pca -> svm
    X_hog_scaled = hog_scaler.transform(X_hog)
    X_hog_pca    = hog_pca.transform(X_hog_scaled)
    hog_pca_path_out = out_dir / f"{img_path.stem}_hog3x3_pca_dim{best_dim}.npy"
    np.save(str(hog_pca_path_out), X_hog_pca[0].astype(np.float32, copy=False))

    P_shape = hog_svm.predict_proba(X_hog_pca)[0]  # (K,)
    shape_classes_raw = getattr(hog_svm, "classes_", list(range(len(P_shape))))
    shape_classes = apply_label_map(shape_classes_raw, label_map)

    # sanity: class order should match; if not, fusion이 깨질 수 있음
    same_order = (list(color_classes_raw) == list(shape_classes_raw))

    # -------------------------
    # 4) Fusion (LR) if available
    #    네 학습 코드: X_fuse = [P_shape, P_color]
    # -------------------------
    fusion_probs = None
    fusion_classes = None
    if fusion_lr_path.exists():
        fusion_lr = joblib.load(str(fusion_lr_path))
        if not hasattr(fusion_lr, "predict_proba"):
            raise ValueError("fusion_lr has no predict_proba().")

        if args.fusion_order == "shape,color":
            X_fuse = np.hstack([P_shape.reshape(1, -1), P_color.reshape(1, -1)])
        else:
            X_fuse = np.hstack([P_color.reshape(1, -1), P_shape.reshape(1, -1)])

        fusion_probs = fusion_lr.predict_proba(X_fuse)[0]
        fusion_classes_raw = getattr(fusion_lr, "classes_", list(range(len(fusion_probs))))
        fusion_classes = apply_label_map(fusion_classes_raw, label_map)

    # -------------------------
    # 5) Save result JSON
    # -------------------------
    result = {
        "image": str(img_path),
        "saved_resized_image": str(resized_path),
        "saved_color_feature": str(color_feat_path),
        "saved_hog_feature": str(hog_feat_path),
        "saved_hog_pca_feature": str(hog_pca_path_out),
        "models": {
            "color_svm": str(color_svm_path),
            "hog_svm": str(hog_svm_path),
            "hog_scaler": str(hog_scaler_path),
            "hog_pca": str(hog_pca_path),
            "fusion_lr": str(fusion_lr_path) if fusion_lr_path.exists() else None,
            "road_label_map": str(label_map_path) if label_map_path.exists() else None,
        },
        "notes": {
            "hog_params": {
                "hog_size_wh": [args.hog_w, args.hog_h],
                "orientations": args.hog_ori,
                "pixels_per_cell": [args.hog_ppc, args.hog_ppc],
                "cells_per_block": [args.hog_cpb, args.hog_cpb],
            },
            "color_params": {"h_bins": args.h_bins, "s_bins": args.s_bins},
            "fusion_order": args.fusion_order,
            "class_order_same_between_color_and_hog": bool(same_order),
            "warning_if_false": "If class order differs, reorder probabilities before fusion.",
        },
        "color_probs": pretty_probs(P_color, color_classes),
        "hog_shape_probs": pretty_probs(P_shape, shape_classes),
        "fusion_probs": pretty_probs(fusion_probs, fusion_classes) if fusion_probs is not None else None,
        "top1_color": top1(pretty_probs(P_color, color_classes)),
        "top1_hog": top1(pretty_probs(P_shape, shape_classes)),
        "top1_fusion": top1(pretty_probs(fusion_probs, fusion_classes)) if fusion_probs is not None else None,
    }

    json_path = out_dir / f"{img_path.stem}_inference_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved JSON: {json_path}")


if __name__ == "__main__":
    main()
