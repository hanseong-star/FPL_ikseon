import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
npy_path = BASE_DIR / "data.npy"

data = np.load(BASE_DIR / "donhwamunro_11_da_A_raw_0894_hog3x3_pca_dim128.npy")

print(type(data))
print(data.shape)
print(data)
