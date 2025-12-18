# src/fpl_detail_models.py
import os, json
import numpy as np
import joblib

from src.fpl_models import fit_sigmoid_calibrator

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DETAIL_ROADS_DEFAULT = {
    "donhwamunro_11_ga",
    "donhwamunro_11_na",
    "donhwamunro_11_da",
    "suporo_28",
}

def _clean_detail(d):
    if d is None:
        return "0"
    s = str(d).strip()
    if s.lower() in ("nan", "none", ""):
        return "0"
    return s

def _detail_root(out_dir: str, feature_tag: str):
    """
    feature_tag 별로 detail_models 폴더를 분리해서 저장
      - feature_tag == "" or "legacy" -> detail_models (기존 호환)
      - else -> detail_models_<feature_tag>
    """
    feature_tag = (feature_tag or "").strip()
    if feature_tag in ("", "legacy"):
        return os.path.join(out_dir, "detail_models")
    return os.path.join(out_dir, f"detail_models_{feature_tag}")

def train_and_save_detail_models(
    X_hog_train: np.ndarray,
    X_color_train: np.ndarray,
    training_road_label,
    training_detail,
    out_dir="FPL_models",
    detail_roads=DETAIL_ROADS_DEFAULT,
    hog_pca_dim=64,
    C=10,
    gamma="scale",
    alpha_shape=0.389,
    min_samples_per_detail=8,
    min_total_samples=30,

    # ✅ 추가: feature 구분 태그
    feature_tag="full",   # 예: "full", "patch3x3"
):
    """
    각 'detail 있는 도로'마다 (feature_tag 버전으로):
      - detail Color SVM
      - detail HOG(Scaler+PCA) SVM
      - detail_label_map
    를 저장

    out_dir/detail_models_<feature_tag>/<road>/
      detail_color_svm.pkl
      detail_hog_svm.pkl
      detail_hog_scaler.pkl
      detail_hog_pca.pkl
      detail_label_map.pkl
      meta.json
    """
    out_dir = str(out_dir)
    detail_root = _detail_root(out_dir, feature_tag)
    os.makedirs(detail_root, exist_ok=True)

    road_arr = np.asarray(training_road_label, dtype=object)
    det_arr  = np.array([_clean_detail(d) for d in training_detail], dtype=object)

    saved = []
    skipped = []

    X_hog_train = np.asarray(X_hog_train)
    X_color_train = np.asarray(X_color_train)

    if X_hog_train.ndim != 2 or X_color_train.ndim != 2:
        raise ValueError("X_hog_train / X_color_train must be 2D arrays: (N, D).")

    if len(road_arr) != len(det_arr) or len(road_arr) != X_hog_train.shape[0] or len(road_arr) != X_color_train.shape[0]:
        raise ValueError("Input lengths mismatch among labels/details/features.")

    for road in sorted(set(road_arr)):
        if road not in detail_roads:
            continue

        idx_road = np.where(road_arr == road)[0]
        if len(idx_road) < min_total_samples:
            skipped.append((road, "too_few_total_samples", int(len(idx_road))))
            continue

        # detail=0 제외하고 detail 학습
        idx_use = [i for i in idx_road if det_arr[i] != "0"]
        if len(idx_use) < min_total_samples:
            skipped.append((road, "too_few_detail_samples", int(len(idx_use))))
            continue

        details = sorted(set(det_arr[idx_use]))
        ok_details = []
        for d in details:
            n = int(np.sum(det_arr[idx_use] == d))
            if n >= min_samples_per_detail:
                ok_details.append(d)

        if len(ok_details) < 2:
            skipped.append((road, "not_enough_detail_classes", len(ok_details)))
            continue

        detail_label_map = {d: j for j, d in enumerate(ok_details)}

        idx_final = [i for i in idx_use if det_arr[i] in detail_label_map]
        y_detail = np.array([detail_label_map[det_arr[i]] for i in idx_final], dtype=np.int64)

        Xh = X_hog_train[idx_final]
        Xc = X_color_train[idx_final]

        # 1) Color SVM
        color_svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            random_state=42,
        )
        color_svm.fit(Xc, y_detail)

        cal_color = fit_sigmoid_calibrator(color_svm, Xc, y_detail,
                                  q_lo=0.10, q_hi=0.90, p_lo=0.05, p_hi=0.95)

        # 2) HOG SVM: scaler + PCA + SVM
        scaler = StandardScaler()
        Xh_s = scaler.fit_transform(Xh)

        n_comp = int(min(hog_pca_dim, Xh_s.shape[0] - 1, Xh_s.shape[1]))
        if n_comp < 2:
            skipped.append((road, "pca_dim_too_small", n_comp))
            continue

        pca = PCA(n_components=n_comp, random_state=42)
        Xh_p = pca.fit_transform(Xh_s)

        hog_svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            random_state=42,
        )
        hog_svm.fit(Xh_p, y_detail)

        cal_hog = fit_sigmoid_calibrator(hog_svm, Xh_p, y_detail,
                                q_lo=0.10, q_hi=0.90, p_lo=0.05, p_hi=0.95)

        # 3) 저장
        road_dir = os.path.join(detail_root, road)
        os.makedirs(road_dir, exist_ok=True)

        joblib.dump(color_svm,        os.path.join(road_dir, "detail_color_svm.pkl"))
        joblib.dump(hog_svm,          os.path.join(road_dir, "detail_hog_svm.pkl"))
        joblib.dump(scaler,           os.path.join(road_dir, "detail_hog_scaler.pkl"))
        joblib.dump(pca,              os.path.join(road_dir, "detail_hog_pca.pkl"))
        joblib.dump(detail_label_map, os.path.join(road_dir, "detail_label_map.pkl"))
        joblib.dump(cal_color, os.path.join(road_dir, "detail_color_cal_sigmoid.pkl"))
        joblib.dump(cal_hog,   os.path.join(road_dir, "detail_hog_cal_sigmoid.pkl"))


        meta = {
            "road": road,
            "feature_tag": str((feature_tag or "").strip()),
            "details": ok_details,
            "n_total_road": int(len(idx_road)),
            "n_used_detail": int(len(idx_final)),
            "hog_pca_dim_actual": int(n_comp),
            "C": float(C),
            "gamma": str(gamma),
            "alpha_shape_for_runtime_fusion": float(alpha_shape),
            "min_samples_per_detail": int(min_samples_per_detail),
        }
        meta["prob_method"] = "sigmoid"
        with open(os.path.join(road_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        saved.append((road, ok_details, int(len(idx_final)), n_comp))
        print(f"[SAVED] detail({feature_tag}) road={road} | classes={ok_details} | n={len(idx_final)} | pca={n_comp}")

    return {"saved": saved, "skipped": skipped, "detail_root": detail_root, "feature_tag": feature_tag}
