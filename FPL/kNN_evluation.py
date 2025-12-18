import os, json, joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.metrics import confusion_matrix, classification_report

MODEL_DIR = "FPL_models"
DETAIL_ROOT = os.path.join(MODEL_DIR, "detail_models")  # feature_tag 쓰면 detail_models_<tag>로 바꿔줘

# =========================
# 유틸
# =========================
def _clean_detail(d):
    if d is None:
        return "0"
    s = str(d).strip()
    if s.lower() in ("nan", "none", ""):
        return "0"
    return s

def _safe_classification_report(y_true, y_pred, labels=None):
    try:
        return classification_report(y_true, y_pred, labels=labels, digits=4)
    except Exception as e:
        return f"(report failed) {e}"

def _mae_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse

# =========================
# 1) DETAIL SVM 평가
# =========================
def eval_detail_svms(
    X_hog_test, X_color_test,
    y_test_road_label, y_test_detail,
    detail_root=DETAIL_ROOT
):
    y_test_road_label = np.asarray(y_test_road_label, dtype=object)
    y_test_detail = np.asarray([_clean_detail(d) for d in y_test_detail], dtype=object)

    rows = []
    per_road_reports = {}

    if not os.path.isdir(detail_root):
        print(f"❗ detail_root not found: {detail_root}")
        return pd.DataFrame(), per_road_reports

    for road in sorted(os.listdir(detail_root)):
        road_dir = os.path.join(detail_root, road)
        if not os.path.isdir(road_dir):
            continue

        meta_path = os.path.join(road_dir, "meta.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label_map = joblib.load(os.path.join(road_dir, "detail_label_map.pkl"))
        inv_map = {v: k for k, v in label_map.items()}

        # 모델 로드
        color_svm = joblib.load(os.path.join(road_dir, "detail_color_svm.pkl"))
        hog_svm   = joblib.load(os.path.join(road_dir, "detail_hog_svm.pkl"))
        scaler    = joblib.load(os.path.join(road_dir, "detail_hog_scaler.pkl"))
        pca       = joblib.load(os.path.join(road_dir, "detail_hog_pca.pkl"))

        alpha = float(meta.get("alpha_shape_for_runtime_fusion", 0.5))

        # 테스트 중 해당 road만 + detail=0 제외 (detail 분류 평가니까)
        idx = np.where((y_test_road_label == road) & (y_test_detail != "0"))[0]
        if len(idx) == 0:
            continue

        Xc = X_color_test[idx]
        Xh = X_hog_test[idx]

        # 해당 road의 label_map에 포함되는 detail만 평가 (나머지는 "unknown" 취급)
        y_true_str = y_test_detail[idx]
        mask_known = np.array([d in label_map for d in y_true_str], dtype=bool)

        if mask_known.sum() < 5:
            rows.append({
                "road": road,
                "n_test_detail": int(len(idx)),
                "n_known_classes": int(mask_known.sum()),
                "acc": np.nan,
                "note": "too_few_known_samples"
            })
            continue

        idx2 = idx[mask_known]
        Xc2 = X_color_test[idx2]
        Xh2 = X_hog_test[idx2]
        y_true = np.array([label_map[_clean_detail(d)] for d in y_test_detail[idx2]], dtype=int)

        # 예측: color proba + hog proba -> weighted fusion
        P_color = color_svm.predict_proba(Xc2)

        Xh_s = scaler.transform(Xh2)
        Xh_p = pca.transform(Xh_s)
        P_hog = hog_svm.predict_proba(Xh_p)

        P_final = alpha * P_hog + (1.0 - alpha) * P_color
        y_pred = np.argmax(P_final, axis=1)

        acc = float(np.mean(y_pred == y_true))

        # report (문자 레이블로 보기 좋게)
        y_true_str2 = np.array([inv_map[i] for i in y_true], dtype=object)
        y_pred_str2 = np.array([inv_map[i] for i in y_pred], dtype=object)

        rows.append({
            "road": road,
            "alpha": alpha,
            "n_test_detail": int(len(idx)),
            "n_used": int(len(idx2)),
            "n_classes": int(len(label_map)),
            "acc": acc
        })

        per_road_reports[road] = {
            "acc": acc,
            "confusion_matrix": confusion_matrix(y_true_str2, y_pred_str2, labels=list(label_map.keys())),
            "report": _safe_classification_report(y_true_str2, y_pred_str2, labels=list(label_map.keys())),
            "meta": meta
        }

    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    return df, per_road_reports

# =========================
# 2) KNN (좌표 회귀) 평가
# =========================
def eval_knn_xy(
    X_hog_test, X_color_test,
    y_test_road_label, y_test_detail,
    test_x, test_y,
    out_dir=MODEL_DIR,
    feature_tag="full",  # 너가 저장할 때 feature_tag를 썼으면 맞춰줘
):
    """
    fpl_knn_models 저장 규칙이 프로젝트마다 다를 수 있어서,
    아래는 '가장 흔한' 패턴:
      - knn_models_<feature_tag>/<road>/knn_xy.pkl
      - detail이면 knn_models_<feature_tag>/<road>/<detail>/knn_xy.pkl
    네 저장 구조가 다르면 경로만 맞춰주면 됨.
    """
    y_test_road_label = np.asarray(y_test_road_label, dtype=object)
    y_test_detail = np.asarray([_clean_detail(d) for d in y_test_detail], dtype=object)
    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    knn_root = os.path.join(out_dir, f"knn_models_{feature_tag}")
    if not os.path.isdir(knn_root):
        knn_root = os.path.join(out_dir, "knn_models")  # fallback

    preds_x = np.full_like(test_x, np.nan, dtype=float)
    preds_y = np.full_like(test_y, np.nan, dtype=float)
    used = np.zeros_like(test_x, dtype=bool)

    missing = defaultdict(int)

    for i in range(len(y_test_road_label)):
        road = y_test_road_label[i]
        det  = y_test_detail[i]

        # detail 모델 우선
        cand_paths = []
        if det != "0":
            cand_paths.append(os.path.join(knn_root, road, det, "knn_xy.pkl"))
        cand_paths.append(os.path.join(knn_root, road, "knn_xy.pkl"))

        model_path = None
        for p in cand_paths:
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            missing[(road, det)] += 1
            continue

        knn = joblib.load(model_path)

        # 입력 feature: (1, D_hog + D_color) 형태로 가정
        x_in = np.hstack([X_hog_test[i], X_color_test[i]]).reshape(1, -1)

        xy_hat = knn.predict(x_in).reshape(-1)
        preds_x[i] = float(xy_hat[0])
        preds_y[i] = float(xy_hat[1])
        used[i] = True

    # 평가 (예측 가능한 샘플만)
    idx = np.where(used & np.isfinite(test_x) & np.isfinite(test_y))[0]
    if len(idx) == 0:
        print("❗ No valid KNN predictions to evaluate.")
        return None, {"missing": dict(missing)}

    mae_x, rmse_x = _mae_rmse(preds_x[idx], test_x[idx])
    mae_y, rmse_y = _mae_rmse(preds_y[idx], test_y[idx])

    # 유클리드 거리 오차(좌표 단위가 같다는 전제)
    dist = np.sqrt((preds_x[idx] - test_x[idx])**2 + (preds_y[idx] - test_y[idx])**2)
    mae_dist = float(np.mean(dist))
    rmse_dist = float(np.sqrt(np.mean(dist**2)))

    summary = {
        "n_eval": int(len(idx)),
        "mae_x": mae_x, "rmse_x": rmse_x,
        "mae_y": mae_y, "rmse_y": rmse_y,
        "mae_dist": mae_dist, "rmse_dist": rmse_dist,
    }
    return summary, {"missing": dict(missing), "used_mask": used}

# =========================
# 실행 예시
# =========================

# ---- (A) detail svm 평가 ----
# 전제: y_test_road_label, y_test_detail 존재해야 함
if all(v in globals() for v in ["y_test_road_label", "y_test_detail"]):
    df_detail, reports = eval_detail_svms(
        X_hog_test=X_hog_test,
        X_color_test=X_color_test,
        y_test_road_label=y_test_road_label,
        y_test_detail=y_test_detail,
        detail_root=DETAIL_ROOT
    )
    print("\n[DETAIL SVM] per-road summary")
    display(df_detail)

    # 특정 road 보고 싶으면:
    # road = df_detail.iloc[0]["road"]
    # print(reports[road]["report"])
else:
    print("⚠️ y_test_road_label / y_test_detail 이 없어서 detail svm 평가는 스킵했어.")

# ---- (B) knn xy 평가 ----
# 전제: test_x,test_y, y_test_road_label,y_test_detail 존재해야 함
need = ["test_x", "test_y", "y_test_road_label", "y_test_detail"]
if all(v in globals() for v in need):
    knn_summary, knn_debug = eval_knn_xy(
        X_hog_test=X_hog_test,
        X_color_test=X_color_test,
        y_test_road_label=y_test_road_label,
        y_test_detail=y_test_detail,
        test_x=test_x,
        test_y=test_y,
        out_dir=MODEL_DIR,
        feature_tag="full"  # 저장할 때 feature_tag 안 썼으면 "legacy" 또는 기본 폴더명으로 수정
    )
    print("\n[KNN XY] summary:", knn_summary)
    print("[KNN XY] missing models top-10:", sorted(knn_debug["missing"].items(), key=lambda x: -x[1])[:10])
else:
    print("⚠️ test_x/test_y 또는 y_test_* 정보가 없어서 KNN 평가는 스킵했어.")
