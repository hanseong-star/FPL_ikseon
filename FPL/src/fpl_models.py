# src/models.py
import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -----------------------------
# Label encoding
# -----------------------------
def encode_road_labels(train_roads, test_roads):
    """
    도로명 문자열 -> 정수 라벨
    """
    unique_roads = sorted(set(train_roads) | set(test_roads))
    road_label_map = {r: i for i, r in enumerate(unique_roads)}

    y_train = np.array([road_label_map[r] for r in train_roads], dtype=np.int64)
    y_test  = np.array([road_label_map[r] for r in test_roads], dtype=np.int64)
    return y_train, y_test, road_label_map


# -----------------------------
# Train single-feature SVMs
# -----------------------------
def train_color_svm(X_train, y_train, C=10, gamma="scale"):
    """
    Color feature용 SVM
    * decision_function_shape='ovr'로 둬야 decision score -> 확률 변환이 쉬움
    """
    svm = SVC(kernel="rbf", C=C, gamma=gamma,
              probability=True, random_state=42,
              decision_function_shape="ovr")
    svm.fit(X_train, y_train)
    return svm


def train_lbp_svm(X_train, y_train, C=10, gamma="scale"):
    """
    LBP feature용 SVM
    """
    svm = SVC(kernel="rbf", C=C, gamma=gamma,
              probability=True, random_state=42,
              decision_function_shape="ovr")
    svm.fit(X_train, y_train)
    return svm


def train_hog_pca_svm_by_dims(X_train, y_train, X_test, y_test, pca_dims, C=10, gamma="scale"):
    """
    HOG -> StandardScaler -> PCA(dim) -> SVM
    dim별로 모델/feature/정확도/시간 저장.
    """
    hog_pca_models = {}
    hog_svm_models = {}
    hog_pca_train_features = {}
    hog_pca_test_features = {}
    train_acc = {}
    test_acc = {}
    time_report = {}

    for dim in pca_dims:
        t_total = time.perf_counter()

        # 1) scaling
        t0 = time.perf_counter()
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        t_scale = time.perf_counter() - t0

        # 2) PCA
        t0 = time.perf_counter()
        pca = PCA(n_components=dim, random_state=42)
        Xtr_p = pca.fit_transform(Xtr)
        Xte_p = pca.transform(Xte)
        t_pca = time.perf_counter() - t0

        # 3) SVM fit
        t0 = time.perf_counter()
        svm = SVC(kernel="rbf", C=C, gamma=gamma,
                  probability=True, random_state=42,
                  decision_function_shape="ovr")
        svm.fit(Xtr_p, y_train)
        t_svm = time.perf_counter() - t0

        # 4) predict
        t0 = time.perf_counter()
        ytr_pred = svm.predict(Xtr_p)
        yte_pred = svm.predict(Xte_p)
        t_pred = time.perf_counter() - t0

        t_all = time.perf_counter() - t_total

        hog_pca_models[dim] = (scaler, pca)  # scaler도 같이 저장
        hog_svm_models[dim] = svm
        hog_pca_train_features[dim] = Xtr_p
        hog_pca_test_features[dim] = Xte_p

        train_acc[dim] = accuracy_score(y_train, ytr_pred)
        test_acc[dim] = accuracy_score(y_test, yte_pred)

        time_report[dim] = {
            "scale_sec": t_scale,
            "pca_sec": t_pca,
            "svm_fit_sec": t_svm,
            "predict_sec": t_pred,
            "total_sec": t_all,
        }

    return {
        "hog_pca_models": hog_pca_models,
        "hog_svm_models": hog_svm_models,
        "hog_pca_train_features": hog_pca_train_features,
        "hog_pca_test_features": hog_pca_test_features,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "time_report": time_report,
    }


# -----------------------------
# Evaluation helpers
# -----------------------------
def eval_svm(model, X_train, y_train, X_test, y_test, name="model"):
    """
    각 SVM 성능 평가 (train/test accuracy)
    """
    t0 = time.perf_counter()
    ytr = model.predict(X_train)
    yte = model.predict(X_test)
    t = time.perf_counter() - t0

    return {
        "name": name,
        "train_acc": float(accuracy_score(y_train, ytr)),
        "test_acc": float(accuracy_score(y_test, yte)),
        "pred_sec": float(t),
    }


# -----------------------------
# Custom "probability" from decision_function
# -----------------------------
def _decision_scores_ovr(model, X):
    """
    SVC(decision_function_shape='ovr') 기준으로:
      scores: (N, K)

    binary일 때 sklearn이 (N,)로 줄여주는 경우도 있어서 (N,2)로 변환해줌.
    """
    scores = model.decision_function(X)

    if scores.ndim == 1:
        # binary: score > 0 이면 classes_[1] 쪽
        scores = np.vstack([-scores, scores]).T  # (N,2)
    return scores.astype(np.float32, copy=False)


def fit_range_calibrator(model, X_train, y_train, q_lo=0.10, q_hi=0.90):
    """
    "이 정도면 100%, 멀면 0%"를 만들기 위한 캘리브레이션.
    클래스별로 decision score의 분포를 보고,
      lo_k = quantile(q_lo)  (이 밑이면 거의 0%)
      hi_k = quantile(q_hi)  (이 이상이면 거의 100%)
    를 저장한다.

    반환 dict:
      {"lo": (K,), "hi": (K,), "classes": model.classes_}
    """
    scores = _decision_scores_ovr(model, X_train)
    classes = model.classes_
    K = len(classes)

    lo = np.zeros(K, dtype=np.float32)
    hi = np.zeros(K, dtype=np.float32)

    for k in range(K):
        # true class k인 샘플들의 'k score'만 모음
        mask = (y_train == classes[k])
        s_k = scores[mask, k]
        if s_k.size == 0:
            lo[k], hi[k] = 0.0, 1.0
            continue
        lo[k] = np.quantile(s_k, q_lo)
        hi[k] = np.quantile(s_k, q_hi)

        # hi==lo면 분모 0 방지용으로 조금 벌림
        if abs(float(hi[k] - lo[k])) < 1e-6:
            hi[k] = lo[k] + 1e-3

    return {"lo": lo, "hi": hi, "classes": classes}


def fit_sigmoid_calibrator(model, X_train, y_train,
                           q_lo=0.10, q_hi=0.90,
                           p_lo=0.05, p_hi=0.95):
    """
    decision_function score를 로지스틱(sigmoid)로 0~1로 매핑하기 위한 캘리브레이션.

    클래스 k에 대해:
      s_lo = quantile(q_lo) of score_k on true-class-k samples
      s_hi = quantile(q_hi) of score_k on true-class-k samples

    그리고
      sigmoid(a_k * s_lo + b_k) = p_lo
      sigmoid(a_k * s_hi + b_k) = p_hi
    를 만족하도록 (a_k, b_k)를 계산한다.

    반환 dict:
      {"a": (K,), "b": (K,), "classes": model.classes_,
       "q_lo":..., "q_hi":..., "p_lo":..., "p_hi":...}
    """
    def _logit(p):
        p = float(p)
        eps = 1e-9
        p = min(max(p, eps), 1.0 - eps)
        return np.log(p / (1.0 - p))

    if not (0.0 < p_lo < 1.0 and 0.0 < p_hi < 1.0):
        raise ValueError("p_lo and p_hi must be in (0,1)")
    if p_lo >= p_hi:
        raise ValueError("Require p_lo < p_hi")
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("Require 0<=q_lo<q_hi<=1")

    scores = _decision_scores_ovr(model, X_train)
    classes = model.classes_
    K = len(classes)

    a = np.zeros(K, dtype=np.float32)
    b = np.zeros(K, dtype=np.float32)

    L_lo = _logit(p_lo)
    L_hi = _logit(p_hi)

    for k in range(K):
        mask = (y_train == classes[k])
        s_k = scores[mask, k]
        if s_k.size == 0:
            a[k], b[k] = 1.0, 0.0
            continue

        s_lo = float(np.quantile(s_k, q_lo))
        s_hi = float(np.quantile(s_k, q_hi))

        # avoid zero division
        if abs(s_hi - s_lo) < 1e-6:
            s_hi = s_lo + 1e-3

        a_k = (L_hi - L_lo) / (s_hi - s_lo)
        b_k = L_lo - a_k * s_lo

        a[k] = np.float32(a_k)
        b[k] = np.float32(b_k)

    return {
        "a": a,
        "b": b,
        "classes": classes,
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
    }


def scores_to_probs_sigmoid(scores, calibrator, power=1.0):
    """
    sigmoid 기반 확률화:
      p_k = sigmoid(a_k * score_k + b_k)
    그 후 sample별로 normalize해서 합이 1이 되게 함.

    - power > 1 : 더 날카롭게(확신 강해짐)  (sigmoid 이후에 적용)
    - power < 1 : 더 부드럽게
    """
    a = calibrator["a"][None, :]
    b = calibrator["b"][None, :]

    z = scores * a + b
    # 안정적인 sigmoid
    z = np.clip(z, -60.0, 60.0)
    p = 1.0 / (1.0 + np.exp(-z))

    if power != 1.0:
        p = np.power(p, power)

    s = p.sum(axis=1, keepdims=True)
    p = np.where(s > 0, p / s, np.full_like(p, 1.0 / p.shape[1]))
    return p.astype(np.float32, copy=False)

def scores_to_probs_range(scores, calibrator, power=1.0):
    """
    range 기반 확률화:
      p_raw_k = clip((score_k - lo_k)/(hi_k - lo_k), 0, 1) ^ power
    그 후 sample별로 normalize해서 합이 1이 되게 함.

    - power > 1 : 더 날카롭게(확신 강해짐)
    - power < 1 : 더 부드럽게
    """
    lo = calibrator["lo"][None, :]
    hi = calibrator["hi"][None, :]

    p = (scores - lo) / (hi - lo)
    p = np.clip(p, 0.0, 1.0)

    if power != 1.0:
        p = np.power(p, power)

    s = p.sum(axis=1, keepdims=True)
    # 전부 0이면 균등 분포
    p = np.where(s > 0, p / s, np.full_like(p, 1.0 / p.shape[1]))
    return p.astype(np.float32, copy=False)


def scores_to_probs_softmax(scores, temp=1.0):
    """
    softmax 기반 확률화 (temp 낮으면 더 날카로움)
    """
    scores = scores / float(temp)
    scores = scores - scores.max(axis=1, keepdims=True)
    expv = np.exp(scores)
    p = expv / expv.sum(axis=1, keepdims=True)
    return p.astype(np.float32, copy=False)


def predict_proba_custom(model, X, method="range", calibrator=None, power=1.0, temp=1.0):
    """
    method:
      - "range": fit_range_calibrator() 결과(calibrator) 필요
      - "sigmoid": fit_sigmoid_calibrator() 결과(calibrator) 필요
      - "softmax": calibrator 불필요

    반환: (N, K) in [0,1], row-sum=1
    """
    scores = _decision_scores_ovr(model, X)

    if method == "softmax":
        return scores_to_probs_softmax(scores, temp=temp)

    if method == "range":
        if calibrator is None:
            raise ValueError("method='range' requires calibrator (use fit_range_calibrator)")
        return scores_to_probs_range(scores, calibrator, power=power)

    if method == "sigmoid":
        if calibrator is None:
            raise ValueError("method='sigmoid' requires calibrator (use fit_sigmoid_calibrator)")
        return scores_to_probs_sigmoid(scores, calibrator, power=power)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Weighted fusion
# -----------------------------
def fuse_probabilities(prob_list, weights):
    """
    prob_list: [P1, P2, ...] each (N,K)
    weights:   [w1, w2, ...]

    최종:
      P = normalize( sum_i w_i * P_i )
    """
    if len(prob_list) == 0:
        raise ValueError("prob_list is empty")
    if len(prob_list) != len(weights):
        raise ValueError("prob_list and weights length mismatch")

    w = np.asarray(weights, dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    s = float(w.sum())
    if s <= 0:
        raise ValueError("sum(weights) must be > 0")
    w = w / s

    P = np.zeros_like(prob_list[0], dtype=np.float32)
    for Pi, wi in zip(prob_list, w):
        P += Pi.astype(np.float32, copy=False) * float(wi)

    # row normalize (안전)
    row = P.sum(axis=1, keepdims=True)
    P = np.where(row > 0, P / row, np.full_like(P, 1.0 / P.shape[1]))
    return P.astype(np.float32, copy=False)


def evaluate_fusion(P_fused, y_true):
    """
    P_fused: (N,K) 최종 확률
    y_true:  (N,)
    """
    y_pred = np.argmax(P_fused, axis=1)
    return float(accuracy_score(y_true, y_pred)), y_pred