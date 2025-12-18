# plots_extra.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def main(csv_path="extra_results/sweep_results.csv", out_dir="extra_results/plots"):
    ensure_dir(out_dir)
    df = pd.read_csv(csv_path)

    # 1) PCA dim vs Fusion accuracy (최고값 기준으로 envelope)
    g = df.groupby("hog_pca_dim")["acc_fusion"].max().reset_index()
    plt.figure()
    plt.plot(g["hog_pca_dim"], g["acc_fusion"], marker="o")
    plt.xlabel("PCA dim")
    plt.ylabel("Best Fusion Accuracy")
    plt.title("Fusion Accuracy vs PCA dim (best over other params)")
    p1 = os.path.join(out_dir, "fusion_acc_vs_pca_dim.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) HOG orientation 별 최고 Fusion accuracy
    g2 = df.groupby("hog_ori")["acc_fusion"].max().reset_index()
    plt.figure()
    plt.bar(g2["hog_ori"].astype(str), g2["acc_fusion"])
    plt.xlabel("HOG orientations")
    plt.ylabel("Best Fusion Accuracy")
    plt.title("Best Fusion Accuracy by HOG orientations")
    p2 = os.path.join(out_dir, "best_fusion_by_hog_ori.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # 3) alpha_hat vs PCA dim (최고 fusion 구성에서)
    # 각 PCA dim에서 acc_fusion 최고인 행만 뽑기
    idx = df.groupby("hog_pca_dim")["acc_fusion"].idxmax()
    best_each_dim = df.loc[idx].sort_values("hog_pca_dim")
    plt.figure()
    plt.plot(best_each_dim["hog_pca_dim"], best_each_dim["alpha_hat"], marker="o")
    plt.xlabel("PCA dim")
    plt.ylabel("alpha_hat (shape contribution)")
    plt.title("alpha_hat vs PCA dim (best fusion config per dim)")
    p3 = os.path.join(out_dir, "alpha_hat_vs_pca_dim.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

if __name__ == "__main__":
    main()
