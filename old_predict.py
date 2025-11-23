import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
import xarray as xr
import matplotlib.colors as mcolors
import seaborn as sns

# Set the style for seaborn
sns.set_theme(style='whitegrid')

# Configuration (must match train_module.py preprocessing)
folder = "data"
memmap_path = "sst_memmap.dat"
history = 5
down_H, down_W = 512, 1080
patch_h, patch_w = 128, 128
stride_h, stride_w = 64, 64
predict_batch = 64

def compute_global_min_max(nc_files):
    gmin = np.inf
    gmax = -np.inf
    for f in nc_files:
        ds = xr.open_dataset(f)
        arr = np.nan_to_num(np.squeeze(ds["sea_surface_temperature_anomaly"].values), nan=0.0).astype(np.float32)
        arr_ds = resize(arr, (down_H, down_W), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        m, M = float(arr_ds.min()), float(arr_ds.max())
        if m < gmin: gmin = m
        if M > gmax: gmax = M
        ds.close()
    return gmin, gmax

def build_coords(T):
    H, W = down_H, down_W

    # Ensure coverage to the image edges: include last patch aligned to the right/bottom edge
    def make_positions(length, patch, stride):
        pos = list(range(0, length - patch + 1, stride))
        if pos[-1] != length - patch:
            pos.append(length - patch)
        return np.array(pos, dtype=np.int32)

    ys = make_positions(H, patch_h, stride_h)
    xs = make_positions(W, patch_w, stride_w)

    n_y = len(ys)
    n_x = len(xs)
    patches_per_frame = n_y * n_x
    n_windows = T - history
    starts = np.arange(0, n_windows, dtype=np.int64)

    starts_grid = np.repeat(starts, patches_per_frame)
    ys_tile = np.tile(np.repeat(ys, n_x), n_windows)
    xs_tile = np.tile(np.tile(xs, n_y), n_windows)
    coords = np.stack([starts_grid, ys_tile, xs_tile], axis=1)
    return coords, n_y, n_x, n_windows

def load_batch_X(coords_batch, T):
    Xs = np.empty((len(coords_batch), history, patch_h, patch_w, 1), dtype=np.float32)
    mm = np.memmap(memmap_path, dtype=np.float16, mode='r', shape=(T, down_H, down_W))
    for i, c in enumerate(coords_batch):
        start, y, x = int(c[0]), int(c[1]), int(c[2])
        patches = np.empty((history, patch_h, patch_w), dtype=np.float32)
        for t in range(history):
            frame = np.array(mm[start + t], dtype=np.float32)
            patches[t] = frame[y:y+patch_h, x:x+patch_w]
        Xs[i] = patches[..., np.newaxis]
    del mm
    return Xs

def reconstruct_full_frame_from_patches(preds, coords_used, H=down_H, W=down_W):
    acc = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    for pred, c in zip(preds, coords_used):
        y, x = int(c[1]), int(c[2])
        patch = pred[..., 0]
        acc[y:y+patch_h, x:x+patch_w] += patch
        counts[y:y+patch_h, x:x+patch_w] += 1.0
    # mark positions never covered by any patch as nan so they plot as "bad" values
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(counts > 0, acc / counts, np.nan)
    return avg, counts

if __name__ == "__main__":
    model_path = "models/best_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    nc_files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    T = len(nc_files)
    if T <= history:
        raise ValueError("Not enough data files to form a history window")

    global_min, global_max = compute_global_min_max(nc_files)
    scale = (global_max - global_min + 1e-6)

    coords, n_y, n_x, n_windows = build_coords(T)

    # pick last temporal window to predict full-frame for the most recent target time
    temporal_start = n_windows - 1
    coords_for_start = coords[coords[:,0] == temporal_start]

    # predict patches in batches
    preds_list = []
    coords_used = []
    for i in range(0, coords_for_start.shape[0], predict_batch):
        batch_coords = coords_for_start[i:i+predict_batch]
        Xb = load_batch_X(batch_coords, T)  # normalized inputs
        pred_b = model.predict(Xb, verbose=0)  # shape (B, patch_h, patch_w, 1)
        preds_list.append(pred_b)
        coords_used.extend(batch_coords.tolist())
    preds_all = np.concatenate(preds_list, axis=0)

    # reconstruct full normalized prediction and un-normalize
    pred_full_norm, counts = reconstruct_full_frame_from_patches(preds_all, coords_used, H=down_H, W=down_W)
    pred_full_un = pred_full_norm * scale + global_min

    # guard against any NaN/inf (keep NaNs so they map to white)
    pred_full_un = np.asarray(pred_full_un, dtype=np.float32)

    # display only the full predicted map (center colormap at 0 -> white)
    cmap = plt.get_cmap('bwr')
    cmap.set_bad('white')              # make NaNs white
    norm = mcolors.TwoSlopeNorm(vmin=float(global_min), vcenter=0.0, vmax=float(global_max))

    print(f"Global min: {global_min}, Global max: {global_max}")

    plt.figure(figsize=(10, 6))
    # use nearest interpolation to avoid thin artifact lines from interpolation
    plt.imshow(pred_full_un, origin='lower', cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')

    num_to_show = 6
    rows, cols = 2, 3

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_to_show):
        patch_un = preds_all[i, :, :, 0] * scale + global_min  # un-normalize
        y, x = coords_used[i][1], coords_used[i][2]

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(
            patch_un,
            cmap=cmap,
            norm=norm,
            origin='lower',
            interpolation='nearest'
        )
        ax.set_title(f"({y},{x})", fontsize=8)
        ax.axis('off')

    plt.suptitle("Six Predicted Patches (2×3 Grid)")
    plt.tight_layout()
    plt.show()