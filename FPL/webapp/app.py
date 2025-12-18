# webapp/app.py
import os, sys, base64
from pathlib import Path
import re
import traceback

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


import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_proba_custom(svm, X, calibrator=None, power=1.0):
    """
    svm: sklearn SVC(probability=False 여도 OK)
    calibrator: dict with keys ['a','b','classes', ...]
    return: (N, K) probability
    """
    scores = svm.decision_function(X)

    # binary면 (N,) 로 나올 수 있어서 (N,2) 형태로 맞춰줌
    if scores.ndim == 1:
        # sklearn SVC binary decision_function은 class1에 대한 score 1개만 줌
        # -> 두 클래스 확률로 만들기 위해 [-score, +score]로 확장
        scores = np.vstack([-scores, scores]).T  # (N,2)

    N, K = scores.shape

    if calibrator is None:
        raise RuntimeError("calibrator dict is required for probability output (predict_proba_custom).")

    a = np.asarray(calibrator.get("a", 1.0), dtype=np.float64)
    b = np.asarray(calibrator.get("b", 0.0), dtype=np.float64)

    # a,b가 스칼라이면 broadcast, 벡터면 (K,)로 맞추기
    if a.ndim == 0: a = np.full((K,), float(a))
    if b.ndim == 0: b = np.full((K,), float(b))

    a = a.reshape(1, -1)  # (1,K)
    b = b.reshape(1, -1)  # (1,K)

    # Platt sigmoid: p = 1 / (1 + exp(a*score + b))
    # (너 dict 구조상 a,b를 이렇게 쓰는 형태가 가장 자연스러움)
    P = 1.0 / (1.0 + np.exp(a * scores + b))

    # optional clip (p_lo/p_hi 있으면)
    p_lo = calibrator.get("p_lo", None)
    p_hi = calibrator.get("p_hi", None)
    if p_lo is not None and p_hi is not None:
        P = np.clip(P, float(p_lo), float(p_hi))

    # optional sharpening
    if power is not None and float(power) != 1.0:
        P = np.power(P, float(power))

    # row normalize (K-class)
    row_sum = P.sum(axis=1, keepdims=True) + 1e-12
    P = P / row_sum
    return P



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
        self.hog_cal = None
        self.color_cal = None  
        # =========================
        # 0) 기본 멤버 먼저 세팅
        # =========================
        self.model_dir = Path(model_dir)
        self.best_dim = int(best_dim)
        self.alpha_shape = float(alpha_shape)
        # ---- Fusion LR 로드
        p_fusion = self.model_dir / f"fusion_lr_dim{self.best_dim}.pkl"
        if not p_fusion.exists():
            raise FileNotFoundError(f"fusion LR not found: {p_fusion}")

        self.fusion_lr = joblib.load(p_fusion)


        # (중요) attribute error 방지용 초기화
        self.hog_cal = None
        self.color_cal = None
        self.hog_svm = None
        self.color_svm = None
        self.fusion_lr = None
        self.road_classes = None

        # =========================
        # 1) label map 로드
        # =========================
        self.road_label_map = joblib.load(self.model_dir / "road_label_map.pkl")
        self.inv_road = inv_map_from_label_map(self.road_label_map)

        # =========================
        # 2) Road 모델 로드 (HOG/Color/Fusion)
        # =========================
        # HOG pipeline
        self.hog_pca    = joblib.load(self.model_dir / f"hog_pca_dim{self.best_dim}.pkl")
        self.hog_scaler = joblib.load(self.model_dir / f"hog_scaler_dim{self.best_dim}.pkl")

        # HOG calibrator (있으면 우선)
        hog_cal_path = self.model_dir / f"hog_cal_dim{self.best_dim}.pkl"
        if hog_cal_path.exists():
            self.hog_cal = joblib.load(hog_cal_path)

        # HOG SVM fallback (있으면 로드)
        hog_svm_path = self.model_dir / f"hog_svm_dim{self.best_dim}.pkl"
        if hog_svm_path.exists():
            self.hog_svm = joblib.load(hog_svm_path)

        # Color calibrator (있으면 우선)
        color_cal_path = self.model_dir / "color_cal.pkl"
        if color_cal_path.exists():
            self.color_cal = joblib.load(color_cal_path)

        # Color SVM fallback (항상 로드 시도)
        self.color_svm = joblib.load(self.model_dir / "color_svm.pkl")

        # Fusion LR (항상 필요)
        self.fusion_lr = joblib.load(self.model_dir / f"fusion_lr_dim{self.best_dim}.pkl")
        self.road_classes = np.asarray(self.fusion_lr.classes_)

        # =========================
        # 3) classes_ sanity check (로드 후에만!)
        # =========================
        def _warn_classes(name, classes_arr):
            if classes_arr is None:
                print(f"[WARN] {name} is None")
                return
            if self.road_classes is None:
                print("[WARN] road_classes is None (fusion_lr not loaded?)")
                return
            if not np.array_equal(np.asarray(classes_arr), self.road_classes):
                print(f"[WARN] {name}.classes_ != fusion_lr.classes_")
                print("  ", name, ":", list(classes_arr))
                print("  fusion_lr:", list(self.road_classes))

        # HOG classes source
        if self.hog_cal is not None and hasattr(self.hog_cal, "classes_"):
            _warn_classes("hog_cal", self.hog_cal.classes_)
        elif self.hog_svm is not None and hasattr(self.hog_svm, "classes_"):
            _warn_classes("hog_svm", self.hog_svm.classes_)
        else:
            print("[WARN] Neither hog_cal nor hog_svm is available.")

        # Color classes source
        if self.color_cal is not None and hasattr(self.color_cal, "classes_"):
            _warn_classes("color_cal", self.color_cal.classes_)
        elif self.color_svm is not None and hasattr(self.color_svm, "classes_"):
            _warn_classes("color_svm", self.color_svm.classes_)
        else:
            print("[WARN] color_svm is not available?")

        # inv_road 커버리지 체크
        missing = [int(c) for c in self.road_classes if int(c) not in self.inv_road]
        if missing:
            print("[WARN] inv_road missing class ids:", missing)

        # =========================
        # 4) KNN root + KNN용 scaler/pca 자동 탐색 로드
        # =========================*
# ---- (2) KNN root + KNN용 scaler/pca 로드 (best_dim 고정)
        self.knn_root = self.model_dir / "knn_models_patch3x3"  # 네가 쓰는 버전 우선
        if not self.knn_root.exists():
            self.knn_root = self.model_dir / "knn_models"
        if not self.knn_root.exists():
            raise FileNotFoundError(f"KNN root not found: {self.knn_root}")

        # ✅ best_dim에 맞는 파일명을 "정확히" 찍어서 로드
        p_scal = self.knn_root / f"knn_hog_scaler_pca{self.best_dim}.pkl"
        p_pca  = self.knn_root / f"knn_hog_pca_pca{self.best_dim}.pkl"

        if not p_scal.exists() or not p_pca.exists():
            # fallback: 폴더 내에서 dim 문자열 포함된 것 찾기
            want = f"dim{self.best_dim}"
            cand_scal = [fn for fn in os.listdir(self.knn_root) if fn.startswith("knn_hog_scaler_") and want in fn and fn.endswith(".pkl")]
            cand_pca  = [fn for fn in os.listdir(self.knn_root) if fn.startswith("knn_hog_pca_") and want in fn and fn.endswith(".pkl")]
            if not cand_scal or not cand_pca:
                raise FileNotFoundError(f"KNN scaler/pca for {want} not found under {self.knn_root}")
            p_scal = self.knn_root / cand_scal[0]
            p_pca  = self.knn_root / cand_pca[0]

        self.knn_hog_scaler = joblib.load(p_scal)
        self.knn_hog_pca    = joblib.load(p_pca)
        # =========================
        # 5) detail SVM 모델 로드(있으면 사용)
        # =========================
        self.detail_models = {}
        detail_root = self.model_dir / "detail_models_patch3x3"
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
                p_cals = pick("detail_color_cal_sigmoid.pkl")
                p_hcals = pick("detail_hog_cal_sigmoid.pkl")

                if all([p_color, p_hog, p_scal, p_pca, p_map]):
                    self.detail_models[r] = {
                        "color": joblib.load(p_color),
                        "hog": joblib.load(p_hog),
                        "hog_scaler": joblib.load(p_scal),
                        "hog_pca": joblib.load(p_pca),
                        "label_map": joblib.load(p_map),
                        "color_cal": joblib.load(p_cals) if p_cals else None,  
                        "hog_cal": joblib.load(p_hcals) if p_hcals else None,   
                    }

    

    # ---- 단일 이미지 전처리
    def preprocess_img(self, img_bgr):
        img_bgr = resize_keep_direction(img_bgr)          # training과 동일
        img = img_bgr.astype(np.float32) / 255.0
        return img_bgr, img

    # ---- 단일 이미지 feature (HOG raw, Color)
    def features_one(self, img_float01):
        X_hog = extract_hog_3x3([img_float01], hog_size=(128, 128),
                                orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X_col = extract_color_hs_3x3([img_float01], h_bins=45, s_bins=48)

        return X_hog, X_col

    # ---- road TOP3
    def predict_roads_top3(self, X_hog_raw, X_col):
        # ---- Color proba (calibrator 우선)
        # ---- Color proba (calibrator 우선/강제)
        # ---- Color proba  
        if self.color_cal is not None and hasattr(self.color_cal, "predict_proba"):
            # sklearn calibrator
            P_color_raw = self.color_cal.predict_proba(X_col)
            color_classes = np.asarray(self.color_cal.classes_)

        elif isinstance(self.color_cal, dict):
            # dict calibrator (Platt params)
            P_color_raw = predict_proba_custom(self.color_svm, X_col, calibrator=self.color_cal, power=1.0)
            # ✅ dict에는 classes_가 없고 classes 키가 있음
            color_classes = np.asarray(self.color_cal.get("classes", self.color_svm.classes_))

        else:
            raise RuntimeError("color_cal is missing/invalid. Expected sklearn calibrator or dict.")


        # ---- HOG proba
        Xh_s = self.hog_scaler.transform(X_hog_raw)
        Xh_p = self.hog_pca.transform(Xh_s)

        if self.hog_cal is not None and hasattr(self.hog_cal, "predict_proba"):
            P_shape_raw = self.hog_cal.predict_proba(Xh_p)
            hog_classes = np.asarray(self.hog_cal.classes_)

        elif isinstance(self.hog_cal, dict):
            P_shape_raw = predict_proba_custom(self.hog_svm, Xh_p, calibrator=self.hog_cal, power=1.0)
            hog_classes = np.asarray(self.hog_cal.get("classes", self.hog_svm.classes_))

        else:
            raise RuntimeError("hog_cal is missing/invalid. Expected sklearn calibrator or dict.")


        # ---- LR이 기대하는 클래스 순서(self.road_classes)로 열 정렬
        P_shape = P_shape_raw
        P_color = P_color_raw

        if not np.array_equal(hog_classes, self.road_classes):
            P_shape = reorder_proba(P_shape_raw, hog_classes, self.road_classes)

        if not np.array_equal(color_classes, self.road_classes):
            P_color = reorder_proba(P_color_raw, color_classes, self.road_classes)

        # ---- Fusion
        # X_fuse = np.hstack([P_shape, P_color])
        # P_fuse = self.fusion_lr.predict_proba(np.hstack([P_shape, P_color]))
        P_fuse = self.alpha_shape * P_shape + (1.0 - self.alpha_shape) * P_color


        classes = self.fusion_lr.classes_
        top3 = proba_topk(P_fuse, classes, self.inv_road, k=3)

        # (디버그) 확률합 체크
        s = float(P_fuse[0].sum())
        if abs(s - 1.0) > 1e-3:
            print(f"[WARN] fusion prob sum != 1 : sum={s:.6f}")

        return top3, P_fuse


    # ---- detail 예측(있으면)
    def predict_detail_if_any(self, road, X_hog_raw, X_col):

        if road not in self.detail_models:
            return None, None

        m = self.detail_models[road]

        print("[DEBUG] detail color expects:", m["color"].n_features_in_, "X_color:", X_col.shape[1])
        # detail color proba (dict calibrator)
        Xc = X_col
        P_c = predict_proba_custom(m["color"], Xc, calibrator=m.get("color_cal"), power=1.0)

        Xh_s = m["hog_scaler"].transform(X_hog_raw)
        Xh_p = m["hog_pca"].transform(Xh_s)
        P_h = predict_proba_custom(m["hog"], Xh_p, calibrator=m.get("hog_cal"), power=1.0)

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
    ALPHA_SHAPE = float(os.environ.get("FPL_ALPHA_SHAPE", "0.389"))  
    ALPHA_SHAPE = float(0.001)  # 테스트용 고정

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

    # ✅ add.py / html 둘 다 대응
    f = request.files.get("image") or request.files.get("file")
    if f is None or f.filename.strip() == "":
        return render_template("index.html", result={"error": f"이미지 업로드 실패. keys={list(request.files.keys())}"})

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
    app.run(host="127.0.0.1", port=5000, debug=True)
