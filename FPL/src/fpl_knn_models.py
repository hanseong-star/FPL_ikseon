# src/fpl_knn_models.py
import os, json
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

DETAIL_ROADS_DEFAULT = {"donhwamunro_11_ga", "donhwamunro_11_na", "donhwamunro_11_da", "suporo_28"}

def _is_valid_xy(x, y):
    return np.isfinite(x) and np.isfinite(y)

def _clean_detail(d):
    if d is None:
        return "0"
    s = str(d).strip()
    if s.lower() in ("nan", "none", ""):
        return "0"
    return s

def _knn_root(out_dir: str, feature_tag: str):
    feature_tag = (feature_tag or "").strip()
    if feature_tag in ("", "legacy"):
        return os.path.join(out_dir, "knn_models")
    return os.path.join(out_dir, f"knn_models_{feature_tag}")

def build_knn_features(X_hog, X_color, scaler, pca):
    Xh_s = scaler.transform(X_hog)
    Xh_p = pca.transform(Xh_s)
    return np.hstack([Xh_p, X_color])

def train_and_save_knn_models(
    X_hog_train, X_color_train,
    training_road_label, training_detail,
    training_x, training_y,
    out_dir,
    hog_pca_dim=128,
    n_neighbors=7,
    detail_roads=DETAIL_ROADS_DEFAULT,
    min_samples=10,
    training_paths=None,

    # ✅ 추가: feature 구분 태그
    feature_tag="full",  # 예: "full", "patch3x3"
):
    """
    KNN (위치 회귀):
      입력: [HOG(PCA) + Color]
      출력: (x, y)

    feature_tag 별로 knn_models 폴더를 분리해서 저장:
      out_dir/knn_models_<feature_tag>/<road>/
    """
    if training_paths is not None:
        training_paths = np.asarray(training_paths, dtype=object)

    training_road_label = np.asarray(training_road_label, dtype=object)
    dt_all = np.array([_clean_detail(d) for d in training_detail], dtype=object)

    training_x = np.asarray(training_x, dtype=np.float32)
    training_y = np.asarray(training_y, dtype=np.float32)

    X_hog_train = np.asarray(X_hog_train)
    X_color_train = np.asarray(X_color_train)

    if X_hog_train.ndim != 2 or X_color_train.ndim != 2:
        raise ValueError("X_hog_train / X_color_train must be 2D arrays: (N, D).")
    if len(training_road_label) != X_hog_train.shape[0] or len(training_road_label) != X_color_train.shape[0]:
        raise ValueError("Input lengths mismatch among labels/details/features.")

    # --- 전역 HOG scaler/PCA (KNN용 공통)
    scaler = StandardScaler()
    Xh_s = scaler.fit_transform(X_hog_train)

    n_comp = int(min(hog_pca_dim, Xh_s.shape[0] - 1, Xh_s.shape[1]))
    if n_comp < 2:
        raise ValueError("KNN PCA n_components too small. Need more samples.")

    pca = PCA(n_components=n_comp, random_state=42)
    Xh_p = pca.fit_transform(Xh_s)

    Z = np.hstack([Xh_p, X_color_train])

    knn_root = _knn_root(out_dir, feature_tag)
    os.makedirs(knn_root, exist_ok=True)

    joblib.dump(scaler, os.path.join(knn_root, f"knn_hog_scaler_pca{n_comp}.pkl"))
    joblib.dump(pca,    os.path.join(knn_root, f"knn_hog_pca_pca{n_comp}.pkl"))

    roads = sorted(list(set(training_road_label)))
    saved = []

    for road in roads:
        idx0 = np.where(training_road_label == road)[0]
        idx = [i for i in idx0 if _is_valid_xy(training_x[i], training_y[i])]
        if len(idx) < max(min_samples, n_neighbors):
            print(f"[SKIP] KNN({feature_tag}) road={road}: too few xy samples ({len(idx)})")
            continue

        Xk = Z[idx]
        Yk = np.stack([training_x[idx], training_y[idx]], axis=1)

        knn = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean"
        )
        knn.fit(Xk, Yk)

        road_dir = os.path.join(knn_root, road)
        os.makedirs(road_dir, exist_ok=True)

        joblib.dump(knn, os.path.join(road_dir, "knn_road.pkl"))
        with open(os.path.join(road_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "road": road,
                    "level": "road",
                    "feature_tag": str((feature_tag or "").strip()),
                    "n": int(len(idx)),
                    "n_neighbors": int(n_neighbors),
                    "pca_dim_actual": int(n_comp),
                },
                f, ensure_ascii=False, indent=2
            )

        saved.append((road, "road"))
        if training_paths is not None:
            np.save(os.path.join(road_dir, "train_paths_road.npy"), training_paths[idx])

        print(f"[SAVED] KNN({feature_tag}) road={road} | n={len(idx)}")

        # --- detail별 KNN (detail 있는 도로만)
        if road in detail_roads:
            details_in_road = sorted(list(set(dt_all[idx])))
            for det in details_in_road:
                if det == "0":
                    continue
                idx2 = [i for i in idx if dt_all[i] == det]
                if len(idx2) < max(min_samples, n_neighbors):
                    print(f"  [SKIP] KNN({feature_tag}) road={road} detail={det}: too few xy samples ({len(idx2)})")
                    continue

                Xk2 = Z[idx2]
                Yk2 = np.stack([training_x[idx2], training_y[idx2]], axis=1)

                knn2 = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights="distance",
                    metric="euclidean"
                )
                knn2.fit(Xk2, Yk2)

                joblib.dump(knn2, os.path.join(road_dir, f"knn_detail_{det}.pkl"))
                saved.append((road, det))
                if training_paths is not None:
                    np.save(os.path.join(road_dir, f"train_paths_detail_{det}.npy"), training_paths[idx2])

                print(f"  [SAVED] KNN({feature_tag}) road={road} detail={det} | n={len(idx2)}")

    print("✅ KNN saved under:", knn_root)
    return {"saved": saved, "knn_root": knn_root, "pca_dim_actual": int(n_comp), "feature_tag": feature_tag}
