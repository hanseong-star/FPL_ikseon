# experiments_extra.py
# -*- coding: utf-8 -*-

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 너 repo 함수
from src.fpl_features import extract_hog_3x3, extract_color_hs_3x3


# -----------------------
# Utils
# -----------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def eval_acc(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))

def alpha_hat_from_lr(lr, K):
    """
    lr.coef_ shape: (n_classes, 2K)
    앞 K = shape(HOG), 뒤 K = color
    """
    W = lr.coef_
    w_shape = np.mean(np.abs(W[:, :K]))
    w_color = np.mean(np.abs(W[:, K:]))
    return float(w_shape / (w_shape + w_color + 1e-12))

def train_color_svm(X_tr, y_tr, C=10, gamma="scale"):
    svm = SVC(C=C, gamma=gamma, kernel="rbf", probability=True)
    svm.fit(X_tr, y_tr)
    return svm

def train_hog_pca_svm(X_tr, y_tr, dim, C=10, gamma="scale"):
    """
    returns: scaler, pca, svm
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)

    pca = PCA(n_components=dim, random_state=0)
    Xp = pca.fit_transform(Xs)

    svm = SVC(C=C, gamma=gamma, kernel="rbf", probability=True)
    svm.fit(Xp, y_tr)
    return scaler, pca, svm

def transform_hog(scaler, pca, X):
    return pca.transform(scaler.transform(X))

def lr_fusion_train_on_test(P_shape_test, P_color_test, y_test):
    """
    너 코드대로 test로 학습하는 LR-fusion.
    """
    X_fuse = np.hstack([P_shape_test, P_color_test])
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(X_fuse, y_test)
    P_fuse = lr.predict_proba(X_fuse)
    y_pred = np.argmax(P_fuse, axis=1)
    return lr, P_fuse, y_pred


# -----------------------
# Main experiment function
# -----------------------
def run_extra_experiments(
    Training_origin_data,
    Test_data,
    y_train_road,
    y_test_road,
    out_dir="extra_results",
    # 기본 설정(너 현재 설정)
    hog_size=(128, 128),
    ppc=(8, 8),
    cpb=(2, 2),
    # 추가 실험 sweep
    hog_orientations_list=(6, 9, 12),
    color_bins_list=((20, 24), (30, 32), (40, 48)),
    pca_dims=(2, 8, 16, 32, 64, 128, 256),
    C_list=(1, 10),
    gamma_list=("scale",),
):
    ensure_dir(out_dir)

    rows = []
    best = None

    # ---- Color feature는 (h_bins, s_bins) 별로 새로 뽑아야 함
    # ---- HOG feature는 orientation 별로 새로 뽑아야 함
    for ori in hog_orientations_list:
        print(f"\n=== HOG orientations={ori} ===")
        t0 = time.time()
        X_hog_train = extract_hog_3x3(
            Training_origin_data,
            hog_size=hog_size,
            orientations=ori,
            pixels_per_cell=ppc,
            cells_per_block=cpb
        )
        X_hog_test = extract_hog_3x3(
            Test_data,
            hog_size=hog_size,
            orientations=ori,
            pixels_per_cell=ppc,
            cells_per_block=cpb
        )
        print("HOG extracted:", X_hog_train.shape, X_hog_test.shape, f"({time.time()-t0:.1f}s)")

        for (h_bins, s_bins) in color_bins_list:
            print(f"  -- Color bins={h_bins}x{s_bins}")
            t1 = time.time()
            X_color_train = extract_color_hs_3x3(Training_origin_data, h_bins=h_bins, s_bins=s_bins)
            X_color_test  = extract_color_hs_3x3(Test_data,            h_bins=h_bins, s_bins=s_bins)
            print("  Color extracted:", X_color_train.shape, X_color_test.shape, f"({time.time()-t1:.1f}s)")

            # Color SVM sweep
            for Cc in C_list:
                for gc in gamma_list:
                    color_svm = train_color_svm(X_color_train, y_train_road, C=Cc, gamma=gc)
                    P_color_test = color_svm.predict_proba(X_color_test)
                    y_pred_color = np.argmax(P_color_test, axis=1)
                    acc_color = eval_acc(y_test_road, y_pred_color)

                    # HOG PCA+SVM sweep
                    for d in pca_dims:
                        for Ch in C_list:
                            for gh in gamma_list:
                                scaler, pca, hog_svm = train_hog_pca_svm(X_hog_train, y_train_road, dim=d, C=Ch, gamma=gh)
                                X_hog_test_pca = transform_hog(scaler, pca, X_hog_test)
                                P_shape_test = hog_svm.predict_proba(X_hog_test_pca)
                                y_pred_shape = np.argmax(P_shape_test, axis=1)
                                acc_shape = eval_acc(y_test_road, y_pred_shape)

                                # LR fusion (test로 학습) — 너 방식 그대로
                                lr, P_fuse, y_pred_fuse = lr_fusion_train_on_test(P_shape_test, P_color_test, y_test_road)
                                acc_fuse = eval_acc(y_test_road, y_pred_fuse)

                                K = P_color_test.shape[1]
                                alpha_hat = alpha_hat_from_lr(lr, K)

                                row = dict(
                                    hog_ori=ori,
                                    hog_pca_dim=d,
                                    hog_C=Ch,
                                    hog_gamma=str(gh),
                                    color_h_bins=h_bins,
                                    color_s_bins=s_bins,
                                    color_C=Cc,
                                    color_gamma=str(gc),
                                    acc_color=acc_color,
                                    acc_hog=acc_shape,
                                    acc_fusion=acc_fuse,
                                    alpha_hat=alpha_hat,
                                )
                                rows.append(row)

                                if (best is None) or (acc_fuse > best["acc_fusion"]):
                                    best = row.copy()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    best_path = os.path.join(out_dir, "best_config.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\nSaved:", csv_path)
    print("Best:", best)

    return df, best


if __name__ == "__main__":
    """
    여기는 너가 기존 노트북에서 데이터 로드한 다음 연결해서 쓰는 걸 추천.
    예: from src.fpl_data_io import load_dataset ... 등
    """
    raise SystemExit(
        "Use this as a module. In your notebook/script:\n"
        "df, best = run_extra_experiments(Training_origin_data, Test_data, y_train_road, y_test_road)"
    )
