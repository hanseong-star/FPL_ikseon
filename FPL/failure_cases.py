# failure_cases.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib

from src.fpl_features import extract_color_hs_3x3, extract_hog_3x3


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="test images folder")
    ap.add_argument("--img_list", default="", help="optional txt with filenames (one per line)")
    ap.add_argument("--y_test_npy", required=True, help="npy file of y_test_road aligned with images order")

    ap.add_argument("--model_dir", default="FPL_models")
    ap.add_argument("--best_dim", type=int, required=True)

    # feature params (너 현재값)
    ap.add_argument("--hog_w", type=int, default=128)
    ap.add_argument("--hog_h", type=int, default=128)
    ap.add_argument("--hog_ori", type=int, default=9)
    ap.add_argument("--hog_ppc", type=int, default=8)
    ap.add_argument("--hog_cpb", type=int, default=2)
    ap.add_argument("--h_bins", type=int, default=30)
    ap.add_argument("--s_bins", type=int, default=32)

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out_dir", default="extra_results/failure_cases")

    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # image list
    if args.img_list:
        names = [x.strip() for x in Path(args.img_list).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        # 폴더 내 jpg/png 전부 (정렬)
        names = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    y_test = np.load(args.y_test_npy).astype(int)
    assert len(names) == len(y_test), f"image count {len(names)} != y_test {len(y_test)}"

    # load models
    md = Path(args.model_dir)
    color_svm = joblib.load(str(md / "color_svm.pkl"))
    hog_svm = joblib.load(str(md / f"hog_svm_dim{args.best_dim}.pkl"))
    hog_scaler = joblib.load(str(md / f"hog_scaler_dim{args.best_dim}.pkl"))
    hog_pca = joblib.load(str(md / f"hog_pca_dim{args.best_dim}.pkl"))
    fusion_path = md / f"fusion_lr_dim{args.best_dim}.pkl"
    fusion_lr = joblib.load(str(fusion_path)) if fusion_path.exists() else None

    # iterate and score margin (confident wrong = 좋은 발표 소재)
    wrong = []
    for i, fn in enumerate(names):
        bgr = cv2.imread(str(img_dir / fn), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        Xc = extract_color_hs_3x3([bgr], h_bins=args.h_bins, s_bins=args.s_bins)
        Xh = extract_hog_3x3(
            [bgr],
            hog_size=(args.hog_w, args.hog_h),
            orientations=args.hog_ori,
            pixels_per_cell=(args.hog_ppc, args.hog_ppc),
            cells_per_block=(args.hog_cpb, args.hog_cpb),
        )

        Pc = color_svm.predict_proba(Xc)[0]
        Xhp = hog_pca.transform(hog_scaler.transform(Xh))
        Ps = hog_svm.predict_proba(Xhp)[0]

        if fusion_lr is not None:
            Xf = np.hstack([Ps.reshape(1, -1), Pc.reshape(1, -1)])
            Pf = fusion_lr.predict_proba(Xf)[0]
            Puse = Pf
        else:
            # fusion 없으면 HOG를 기준으로
            Puse = Ps

        pred = int(np.argmax(Puse))
        gt = int(y_test[i])
        conf = float(np.max(Puse))

        if pred != gt:
            wrong.append((conf, fn, gt, pred))

    wrong.sort(reverse=True, key=lambda x: x[0])
    pick = wrong[: args.topk]

    manifest = []
    for rank, (conf, fn, gt, pred) in enumerate(pick, start=1):
        src = img_dir / fn
        dst = out_dir / f"{rank:02d}_conf{conf:.3f}_gt{gt}_pred{pred}_{fn}"
        img = cv2.imread(str(src))
        cv2.imwrite(str(dst), img)
        manifest.append({"rank": rank, "file": fn, "saved_as": dst.name, "gt": gt, "pred": pred, "conf": conf})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} failure cases to {out_dir}")
    print("Also saved manifest.json")

if __name__ == "__main__":
    main()
