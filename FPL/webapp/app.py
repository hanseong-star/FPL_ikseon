# webapp/app.py
import os, sys, base64
from pathlib import Path
import re

import numpy as np
import cv2
import joblib
from flask import Flask, request, render_template

# =========================
# (A) src import 경로 추가
# =========================
ROOT = Path(__file__).resolve().parents[1]      # .../FPL
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))                   # src 폴더를 import 경로로



sys.path.insert(0, str(ROOT))
from src.fpl_data_io import resize_keep_direction
from src.fpl_features import extract_color_hs_3x3, extract_hog_3x3


# =========================
# (B) Flask 기본 설정 (로그인 없이)
# =========================
app = Flask(__name__)

DETAIL_ROADS = {"donhwamunro_11_ga", "donhwamunro_11_na", "donhwamunro_11_da", "suporo_28"}


# =========================
# (C) 유틸
# =========================
def bgr_to_data_uri(img_bgr, max_w=720, quality=85):
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if w > max_w:
        nh = int(h * (max_w / w))
        img_bgr = cv2.resize(img_bgr, (max_w, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def inv_map_from_label_map(road_label_map: dict):
    return {v: k for k, v in road_label_map.items()}


def proba_topk(P, classes, inv_map, k=3):
    """
    P: (1, K)
    classes: (K,)  -> P의 열 순서에 대응하는 클래스 라벨
    inv_map: class_id(int) -> road_name(str)
    """
    p = P[0]
    order = np.argsort(p)[::-1][:k]
    rows = []
    for rank, j in enumerate(order, start=1):
        c = int(classes[j])
        name = inv_map.get(c, str(c))
        rows.append({
            "rank": rank,
            "class_id": c,            # ✅ 매칭 검증용
            "road": name,
            "prob": float(p[j]),
        })
    return rows

def reorder_proba(P, from_classes, to_classes):
    """
    P의 열(from_classes 순서)을 to_classes 순서로 재정렬.
    """
    from_classes = np.asarray(from_classes)
    to_classes   = np.asarray(to_classes)

    idx = []
    for c in to_classes:
        pos = np.where(from_classes == c)[0]
        if len(pos) == 0:
            raise ValueError(f"Missing class {c} in model classes_")
        idx.append(int(pos[0]))
    return P[:, idx]


# =========================
# (D) 서비스 클래스 (모델 로드 + 예측)
# =========================
class FPLService:
    def __init__(self, model_dir, best_dim=128, alpha_shape=0.389):
        self.model_dir = Path(model_dir)
        self.best_dim = int(best_dim)
        self.alpha_shape = float(alpha_shape)

        # ---- (1) Road 분류 모델
        self.color_svm = joblib.load(self.model_dir / "color_svm.pkl")
        self.hog_svm = joblib.load(self.model_dir / f"hog_svm_dim{self.best_dim}.pkl")
        self.hog_pca = joblib.load(self.model_dir / f"hog_pca_dim{self.best_dim}.pkl")
        self.hog_scaler = joblib.load(self.model_dir / f"hog_scaler_dim{self.best_dim}.pkl")
        self.fusion_lr = joblib.load(self.model_dir / f"fusion_lr_dim{self.best_dim}.pkl")

        self.road_label_map = joblib.load(self.model_dir / "road_label_map.pkl")
        self.inv_road = inv_map_from_label_map(self.road_label_map)
                # ===== sanity check: classes_ alignment =====
        self.road_classes = np.asarray(self.fusion_lr.classes_)  # LR 출력 클래스 순서

        def _warn_classes(name, classes_arr):
            if not np.array_equal(np.asarray(classes_arr), self.road_classes):
                print(f"[WARN] {name}.classes_ != fusion_lr.classes_")
                print("  ", name, ":", list(classes_arr))
                print("  fusion_lr:", list(self.road_classes))

        _warn_classes("hog_svm", self.hog_svm.classes_)
        _warn_classes("color_svm", self.color_svm.classes_)

        # inv_road 커버리지 체크
        missing = [int(c) for c in self.road_classes if int(c) not in self.inv_road]
        if missing:
            print("[WARN] inv_road missing class ids:", missing)


        # ---- (2) KNN root + KNN용 scaler/pca 자동 탐색 로드
        self.knn_root = self.model_dir / "knn_models"
        if not self.knn_root.exists():
            raise FileNotFoundError(f"KNN root not found: {self.knn_root}")

        self.knn_hog_scaler = None
        self.knn_hog_pca = None
        for fn in os.listdir(self.knn_root):
            if fn.startswith("knn_hog_scaler_") and fn.endswith(".pkl"):
                self.knn_hog_scaler = joblib.load(self.knn_root / fn)
            if fn.startswith("knn_hog_pca_") and fn.endswith(".pkl"):
                self.knn_hog_pca = joblib.load(self.knn_root / fn)
        if self.knn_hog_scaler is None or self.knn_hog_pca is None:
            raise FileNotFoundError("KNN scaler/pca not found under knn_root")

        # ---- (3) detail SVM 모델 로드(있으면 사용)
        # 저장 규칙(유연하게 탐색):
        # FPL_models/detail_models/<road>/
        #   detail_color_svm.pkl or color_svm.pkl
        #   detail_hog_svm.pkl   or hog_svm.pkl
        #   detail_hog_scaler.pkl or hog_scaler.pkl
        #   detail_hog_pca.pkl    or hog_pca.pkl
        #   detail_label_map.pkl  or label_map.pkl
        self.detail_models = {}
        detail_root = self.model_dir / "detail_models"
        if detail_root.exists():
            for r in DETAIL_ROADS:
                rdir = detail_root / r
                if not rdir.exists():
                    continue

                def pick(*cands):
                    for name in cands:
                        p = rdir / name
                        if p.exists():
                            return p
                    return None

                p_color = pick("detail_color_svm.pkl", "color_svm.pkl")
                p_hog   = pick("detail_hog_svm.pkl", "hog_svm.pkl")
                p_scal  = pick("detail_hog_scaler.pkl", "hog_scaler.pkl")
                p_pca   = pick("detail_hog_pca.pkl", "hog_pca.pkl")
                p_map   = pick("detail_label_map.pkl", "label_map.pkl")

                if all([p_color, p_hog, p_scal, p_pca, p_map]):
                    self.detail_models[r] = {
                        "color": joblib.load(p_color),
                        "hog": joblib.load(p_hog),
                        "hog_scaler": joblib.load(p_scal),
                        "hog_pca": joblib.load(p_pca),
                        "label_map": joblib.load(p_map),
                    }

    # ---- 단일 이미지 전처리
    def preprocess_img(self, img_bgr):
        img_bgr = resize_keep_direction(img_bgr)          # training과 동일
        img = img_bgr.astype(np.float32) / 255.0
        return img_bgr, img

    # ---- 단일 이미지 feature (HOG raw, Color)
    def features_one(self, img_float01):
        X_hog = extract_hog_3x3([img_float01], hog_size=(128, 128),
                                orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X_col = extract_color_hs_3x3([img_float01], h_bins=30, s_bins=32)
        return X_hog, X_col

    # ---- road TOP3
    def predict_roads_top3(self, X_hog_raw, X_color):
        P_color = self.color_svm.predict_proba(X_color)  # (1,K)

        Xh_s = self.hog_scaler.transform(X_hog_raw)
        Xh_p = self.hog_pca.transform(Xh_s)
        P_shape_raw = self.hog_svm.predict_proba(Xh_p)
        P_color_raw = self.color_svm.predict_proba(X_color)

        # ===== (중요) LR이 기대하는 클래스 순서(self.road_classes)로 확률 열 정렬 =====
        P_shape = P_shape_raw
        P_color = P_color_raw

        if not np.array_equal(self.hog_svm.classes_, self.road_classes):
            P_shape = reorder_proba(P_shape_raw, self.hog_svm.classes_, self.road_classes)

        if not np.array_equal(self.color_svm.classes_, self.road_classes):
            P_color = reorder_proba(P_color_raw, self.color_svm.classes_, self.road_classes)

        X_fuse = np.hstack([P_shape, P_color])
        P_fuse = self.fusion_lr.predict_proba(X_fuse)

        classes = self.fusion_lr.classes_  # (== self.road_classes)
        top3 = proba_topk(P_fuse, classes, self.inv_road, k=3)

        # ===== 확률-라벨 매칭 디버그 체크 =====
        s = float(P_fuse[0].sum())
        best_j = int(np.argmax(P_fuse[0]))
        best_c = int(classes[best_j])
        best_road = self.inv_road.get(best_c, str(best_c))

        if len(top3) > 0 and top3[0]["road"] != best_road:
            print("[BUG?] top3[0] road mismatch with argmax mapping!")
            print("  top3[0] =", top3[0], "argmax_road =", best_road, "argmax_class_id =", best_c)

        if abs(s - 1.0) > 1e-3:
            print(f"[WARN] fusion prob sum != 1 : sum={s:.6f}")

        return top3, P_fuse

    # ---- detail 예측(있으면)
    def predict_detail_if_any(self, road, X_hog_raw, X_color):
        if road not in self.detail_models:
            return None, None

        m = self.detail_models[road]
        P_c = m["color"].predict_proba(X_color)

        Xh_s = m["hog_scaler"].transform(X_hog_raw)
        Xh_p = m["hog_pca"].transform(Xh_s)
        P_h = m["hog"].predict_proba(Xh_p)

        # 동일 가중치(형태 비중 alpha_shape)
        P = self.alpha_shape * P_h + (1.0 - self.alpha_shape) * P_c

        inv_det = {v: k for k, v in m["label_map"].items()}
        pairs = []
        for j, c in enumerate(m["hog"].classes_):
            pairs.append((inv_det.get(int(c), str(c)), float(P[0, j])))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[0][0], pairs  # top1 detail, ranked list

    # ---- 좌표 + 유사이미지 (✅ 저장된 train_paths_*.npy 사용)
    def locate_and_retrieve(self, road, detail, X_hog_raw, X_color):
        road_dir = self.knn_root / road
        if not road_dir.exists():
            return None

        # detail KNN을 쓸지 결정
        use_detail = False
        knn_path = road_dir / "knn_road.pkl"
        paths_npy = road_dir / "train_paths_road.npy"

        if detail is not None:
            cand_knn = road_dir / f"knn_detail_{detail}.pkl"
            cand_npy = road_dir / f"train_paths_detail_{detail}.npy"
            if cand_knn.exists() and cand_npy.exists():
                use_detail = True
                knn_path = cand_knn
                paths_npy = cand_npy

        if not knn_path.exists() or not paths_npy.exists():
            return None

        knn = joblib.load(knn_path)

        # KNN feature Z (knn용 scaler/pca)
        Xh_s = self.knn_hog_scaler.transform(X_hog_raw)
        Xh_p = self.knn_hog_pca.transform(Xh_s)
        Z = np.hstack([Xh_p, X_color])  # (1, D)

        # 좌표 예측
        pred_xy = knn.predict(Z)[0]  # (2,)

        # ✅ 유사 이미지: KNN이 학습한 subset 내부에서 1-NN index 얻기
        dist, nn_idx = knn.kneighbors(Z, n_neighbors=1, return_distance=True)
        j = int(nn_idx[0, 0])
        nn_dist = float(dist[0, 0])

        train_paths = np.load(paths_npy, allow_pickle=True)

        # (1) KNN fit 샘플 수와 경로 수 매칭 체크
        n_fit = getattr(knn, "n_samples_fit_", None)
        if n_fit is not None and len(train_paths) != int(n_fit):
            print(f"[WARN] train_paths length != knn.n_samples_fit_ (road={road}, detail={detail})",
                "len(train_paths)=", len(train_paths), "n_samples_fit_=", int(n_fit))

        # (2) 인덱스 범위 체크
        best_path = None
        if 0 <= j < len(train_paths):
            best_path = str(train_paths[j])
        else:
            print(f"[WARN] neighbor index out of range: j={j}, len(train_paths)={len(train_paths)}")

        # (3) road prefix 체크 (파일명 규칙 기준)
        if best_path:
            base = os.path.basename(best_path)
            if not base.startswith(str(road)):
                print(f"[WARN] NN image basename not starting with road '{road}': {base}")

        return {
            "xy": (float(pred_xy[0]), float(pred_xy[1])),
            "similar_path": best_path,
            "use_detail": bool(use_detail),
            "nn_index": j,          # ✅
            "nn_dist": nn_dist,     # ✅
        }



# =========================
# (E) 전역 서비스 1회 생성
# =========================
SERVICE = None

def get_service():
    global SERVICE
    if SERVICE is not None:
        return SERVICE

    MODEL_DIR = str(ROOT / "FPL_models")
    BEST_DIM = int(os.environ.get("FPL_BEST_DIM", "128"))
    ALPHA_SHAPE = float(os.environ.get("FPL_ALPHA_SHAPE", "0.389"))  # ✅ 3.89가 아니라 0.389

    SERVICE = FPLService(
        model_dir=MODEL_DIR,
        best_dim=BEST_DIM,
        alpha_shape=ALPHA_SHAPE,
    )
    return SERVICE


# =========================
# (F) 라우팅
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    svc = get_service()

    f = request.files.get("image")
    if f is None or f.filename.strip() == "":
        return render_template("index.html", result={"error": "이미지를 업로드해줘."})

    data = np.frombuffer(f.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return render_template("index.html", result={"error": "이미지 읽기 실패(파일 형식 확인)."})
    
    show_bgr, img = svc.preprocess_img(img_bgr)
    X_hog_raw, X_color = svc.features_one(img)

    # 1) road top3
    top3, _ = svc.predict_roads_top3(X_hog_raw, X_color)

    # 2) 각 road 후보에 대해: detail -> knn -> 유사이미지
    road_results = []

    for item in top3:
        # item: {"rank":1, "class_id":..., "road":..., "prob":...}
        road = item["road"]
        prob = item["prob"]
        class_id = item.get("class_id", None)  # 디버그용

        # --- (라벨 매칭 체크) ---
        mapped = svc.inv_road.get(int(class_id)) if class_id is not None else None
        if class_id is not None and mapped != road:
            print("[BUG] prob-label mismatch!",
                "class_id=", class_id,
                "inv_road[class_id]=", mapped,
                "top3_road=", road)

        detail_pred = None
        detail_rank = None

        if road in DETAIL_ROADS:
            detail_pred, detail_rank = svc.predict_detail_if_any(road, X_hog_raw, X_color)

        loc = svc.locate_and_retrieve(road, detail_pred, X_hog_raw, X_color)

        sim_img_uri = None
        if loc and loc.get("similar_path") and os.path.exists(loc["similar_path"]):
            sim_bgr = cv2.imread(loc["similar_path"])
            sim_img_uri = bgr_to_data_uri(sim_bgr, max_w=540)

        road_results.append({
            "rank": item.get("rank"),
            "class_id": class_id,
            "road": road,
            "prob": prob,
            "detail": detail_pred,
            "detail_rank": detail_rank,
            "xy": loc["xy"] if loc else None,
            "used_detail_knn": loc["use_detail"] if loc else False,
            "similar_image": sim_img_uri,
            "similar_path": loc.get("similar_path") if loc else None,   # ✅ 경로도 같이 넣어(검증용)
            "nn_index": loc.get("nn_index") if loc else None,           # ✅ 아래 2)에서 추가
            "nn_dist": loc.get("nn_dist") if loc else None,             # ✅ 아래 2)에서 추가
        })


    result = {
        "upload_image": bgr_to_data_uri(show_bgr, max_w=720),
        "top3": top3,
        "roads": road_results,
    }
    return render_template("index.html", result=result)


if __name__ == "__main__":
    # 로컬에서만 접속 가능(나만)
    app.run(host="127.0.0.1", port=5000, debug=False)
