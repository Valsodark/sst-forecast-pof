import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize

# Configuration - keep consistent with train_module.py
folder = "data"
memmap_path = "sst_memmap.dat"
history = 5
down_H, down_W = 256, 540
patch_h, patch_w = 64, 64
stride_h, stride_w = 32, 32
predict_batch = 64  # how many patches to predict per model call

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
    n_y = (H - patch_h) // stride_h + 1
    n_x = (W - patch_w) // stride_w + 1
    patches_per_frame = n_y * n_x
    n_windows = T - history
    starts = np.arange(0, n_windows, dtype=np.int64)
    ys = np.arange(0, H - patch_h + 1, stride_h, dtype=np.int32)
    xs = np.arange(0, W - patch_w + 1, stride_w, dtype=np.int32)
    starts_grid = np.repeat(starts, patches_per_frame)
    ys_tile = np.tile(np.repeat(ys, n_x), n_windows)
    xs_tile = np.tile(np.tile(xs, n_y), n_windows)
    coords = np.stack([starts_grid, ys_tile, xs_tile], axis=1)
    return coords, n_y, n_x, n_windows

def load_patch_window_from_memmap(start_y_x, T):
    start, y, x = int(start_y_x[0]), int(start_y_x[1]), int(start_y_x[2])
    mm = np.memmap(memmap_path, dtype=np.float16, mode='r', shape=(T, down_H, down_W))
    patches = np.empty((history + 1, patch_h, patch_w), dtype=np.float32)
    for t in range(history + 1):
        frame = np.array(mm[start + t], dtype=np.float32)
        patches[t] = frame[y:y+patch_h, x:x+patch_w]
    del mm
    X = patches[:history][..., np.newaxis].astype(np.float32)
    Y = patches[history][..., np.newaxis].astype(np.float32)
    return X, Y

def load_batch_X(coords_batch, T):
    Xs = np.empty((len(coords_batch), history, patch_h, patch_w, 1), dtype=np.float32)
    for i, c in enumerate(coords_batch):
        X, _ = load_patch_window_from_memmap(c, T)
        Xs[i] = X
    return Xs

def reconstruct_full_frame_from_patches(preds, coords_used, frame_index, H=down_H, W=down_W):
    acc = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    for pred, c in zip(preds, coords_used):
        start, y, x = int(c[0]), int(c[1]), int(c[2])
        # pred shape (patch_h, patch_w, 1)
        patch = pred[..., 0]
        acc[y:y+patch_h, x:x+patch_w] += patch
        counts[y:y+patch_h, x:x+patch_w] += 1.0
    # avoid divide by zero
    mask = counts == 0
    counts[mask] = 1.0
    full = acc / counts
    return full

def plot_sequence_and_prediction(X_seq, Y_true, Y_pred, vmin=0.0, vmax=1.0, cmap='coolwarm'):
    frames = X_seq.shape[0]
    cols = frames + 2
    fig, axs = plt.subplots(1, cols, figsize=(3*cols, 3))
    for i in range(frames):
        ax = axs[i]
        ax.imshow(X_seq[i,...,0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Input t-{frames-i}')
        ax.axis('off')
    axs[frames].imshow(Y_true[...,0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axs[frames].set_title('Ground truth')
    axs[frames].axis('off')
    axs[frames+1].imshow(Y_pred[...,0], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axs[frames+1].set_title('Prediction (patch)')
    axs[frames+1].axis('off')
    plt.tight_layout()
    plt.show()

def plot_full_frame(original_full, predicted_full, vmin=None, vmax=None, cmap='coolwarm'):
    if vmin is None: vmin = float(min(original_full.min(), predicted_full.min()))
    if vmax is None: vmax = float(max(original_full.max(), predicted_full.max()))
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].imshow(original_full, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axs[0].set_title('Original / Ground truth full frame')
    axs[0].axis('off')
    axs[1].imshow(predicted_full, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axs[1].set_title('Predicted full frame')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

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

    # choose a temporal window (last available window) and predict full frame for that window
    # start index selects which temporal window to use (0..n_windows-1). Use the last available.
    temporal_start = n_windows - 1
    # filter coords for the chosen temporal start
    coords_for_start = coords[coords[:,0] == temporal_start]

    # prepare ground-truth full frame (assemble patches from memmap to a full frame for the chosen start)
    mm = np.memmap(memmap_path, dtype=np.float16, mode='r', shape=(T, down_H, down_W))
    ground_full = np.array(mm[temporal_start + history], dtype=np.float32)  # the true full frame at target time (normalized)
    del mm
    ground_full_un = ground_full * scale + global_min

    # predict patches in batches and reconstruct
    preds_list = []
    coords_used = []
    for i in range(0, coords_for_start.shape[0], predict_batch):
        batch_coords = coords_for_start[i:i+predict_batch]
        Xb = load_batch_X(batch_coords, T)  # normalized inputs
        pred_b = model.predict(Xb, verbose=0)  # shape (B, patch_h, patch_w, 1)
        preds_list.append(pred_b)
        coords_used.extend(batch_coords.tolist())
    preds_all = np.concatenate(preds_list, axis=0)

    # reconstruct full normalized predicted frame
    pred_full_norm = reconstruct_full_frame_from_patches(preds_all, coords_used, temporal_start, H=down_H, W=down_W)
    pred_full_un = pred_full_norm * scale + global_min

    # also show one random patch example (inputs + ground truth + patch prediction)
    sample_idx = np.random.randint(0, coords_for_start.shape[0])
    sample_coord = coords_for_start[sample_idx]
    X_sample, Y_sample = load_patch_window_from_memmap(sample_coord, T)
    Y_sample_pred = model.predict(np.expand_dims(X_sample, axis=0))[0]

    # plot patch sequence and patch prediction (un-normalized)
    X_sample_un = X_sample * scale + global_min
    Y_sample_un = Y_sample * scale + global_min
    Y_sample_pred_un = Y_sample_pred * scale + global_min
    plot_sequence_and_prediction(X_sample_un, Y_sample_un, Y_sample_pred_un, vmin=global_min, vmax=global_max)

    # plot full frame comparison
    plot_full_frame(ground_full_un, pred_full_un, vmin=global_min, vmax=global_max)
