# src/cache_io.py
import os
import json
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _shape_key(img):
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    return f"{h}x{w}x{c}"

def save_images_split_atomic(images, prefix: str, local_dir: str, final_dir: str):
    """
    이미지(list[np.ndarray])를 shape별로 나누어 저장.
    저장은 local_dir에 먼저 저장 후 os.replace로 final_dir에 원자적 이동.
    반환: dict(shape_key -> saved_path)
    """
    ensure_dir(local_dir)
    ensure_dir(final_dir)

    groups = {}
    for img in images:
        key = _shape_key(img)
        groups.setdefault(key, []).append(img)

    saved = {}
    for key, imgs in groups.items():
        arr = np.stack(imgs).astype(np.float32)

        tmp_path = os.path.join(local_dir, f"{prefix}_{key}.npy")
        final_path = os.path.join(final_dir, f"{prefix}_{key}.npy")

        np.save(tmp_path, arr)
        os.replace(tmp_path, final_path)  # ✅ atomic replace/move

        saved[key] = final_path

    return saved

def save_metadata_np(out_dir: str, **arrays):
    """
    메타데이터(np.array)를 out_dir에 npy로 저장.
    예: save_metadata_np(cache_dir, training_road_label=..., test_photo_id=...)
    """
    ensure_dir(out_dir)
    for name, arr in arrays.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr, allow_pickle=True)

def load_metadata_np(in_dir: str, *names):
    """
    저장된 메타데이터 npy들을 로드해서 dict로 반환.
    """
    out = {}
    for name in names:
        path = os.path.join(in_dir, f"{name}.npy")
        out[name] = np.load(path, allow_pickle=True)
    return out

def save_cache_bundle(
    cache_dir: str,
    local_tmp_dir: str,
    train_images,
    test_images,
    meta: dict,
    bundle_name: str = "bundle"
):
    """
    이미지 + 메타데이터를 한 번에 캐시 저장.
    - train/test 이미지는 shape별로 분할 저장
    - meta는 npy로 저장
    - manifest.json 저장(어떤 파일이 저장됐는지 기록)
    """
    ensure_dir(cache_dir)
    ensure_dir(local_tmp_dir)

    train_paths = save_images_split_atomic(train_images, "X_train_img", local_tmp_dir, cache_dir)
    test_paths  = save_images_split_atomic(test_images,  "X_test_img",  local_tmp_dir, cache_dir)

    # meta 저장
    save_metadata_np(cache_dir, **meta)

    manifest = {
        "bundle_name": bundle_name,
        "train_img_paths": train_paths,
        "test_img_paths": test_paths,
        "meta_keys": list(meta.keys()),
    }

    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest

def load_cache_bundle(cache_dir: str):
    """
    manifest.json 기반으로 캐시 로드.
    반환:
      train_blocks: list[np.ndarray]  (각 블록 shape: (N,H,W,C))
      test_blocks : list[np.ndarray]
      meta        : dict(name -> np.ndarray)
      manifest    : dict
    """
    import json

    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {cache_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_blocks = []
    for _, path in manifest["train_img_paths"].items():
        train_blocks.append(np.load(path))

    test_blocks = []
    for _, path in manifest["test_img_paths"].items():
        test_blocks.append(np.load(path))

    meta = load_metadata_np(cache_dir, *manifest["meta_keys"])

    return train_blocks, test_blocks, meta, manifest

def flatten_blocks(blocks):
    """
    shape별로 저장된 list of (N,H,W,C) 블록들을
    기존처럼 list[np.ndarray] 이미지 리스트로 풀어줌.
    """
    imgs = []
    for block in blocks:
        for i in range(block.shape[0]):
            imgs.append(block[i])
    return imgs
