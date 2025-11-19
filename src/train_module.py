import os
import glob
import xarray as xr
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Parameters
folder = "data"
history = 5
down_H, down_W = 512, 1080
patch_h, patch_w = 128, 128
stride_h, stride_w = 64, 64
batch_size = 8
epochs = 100
memmap_path = "sst_memmap.dat"
use_memmap = True

# Load NetCDF files
nc_files = sorted(glob.glob(os.path.join(folder, "*.nc")))
T = len(nc_files)
if T <= history:
    raise ValueError(f"Not enough files. Need > {history}, got {T}")

# Compute global min/max
global_min = np.inf
global_max = -np.inf
for f in nc_files:
    ds = xr.open_dataset(f)
    arr = np.nan_to_num(np.squeeze(ds["sea_surface_temperature_anomaly"].values), nan=0.0).astype(np.float32)
    arr_ds = resize(arr, (down_H, down_W), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    m, M = float(arr_ds.min()), float(arr_ds.max())
    if m < global_min: global_min = m
    if M > global_max: global_max = M
    ds.close()

# Create memmap if needed
if use_memmap and not os.path.exists(memmap_path):
    mm = np.memmap(memmap_path, dtype=np.float16, mode='w+', shape=(T, down_H, down_W))
    for i, f in enumerate(nc_files):
        ds = xr.open_dataset(f)
        arr = np.nan_to_num(np.squeeze(ds["sea_surface_temperature_anomaly"].values), nan=0.0).astype(np.float32)
        arr_ds = resize(arr, (down_H, down_W), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        arr_norm = (arr_ds - global_min) / (global_max - global_min + 1e-6)
        mm[i, ...] = arr_norm.astype(np.float16)
        ds.close()
    del mm

# Compute patch coordinates and counts
H, W = down_H, down_W
n_y = (H - patch_h) // stride_h + 1
n_x = (W - patch_w) // stride_w + 1
patches_per_frame = n_y * n_x
n_windows = T - history
total_samples = n_windows * patches_per_frame

# Build coords array
starts = np.arange(0, n_windows, dtype=np.int64)
ys = np.arange(0, H - patch_h + 1, stride_h, dtype=np.int32)
xs = np.arange(0, W - patch_w + 1, stride_w, dtype=np.int32)

starts_grid = np.repeat(starts, patches_per_frame)
ys_tile = np.tile(np.repeat(ys, n_x), n_windows)
xs_tile = np.tile(np.tile(xs, n_y), n_windows)
coords = np.stack([starts_grid, ys_tile, xs_tile], axis=1)

# Load patch window
def py_load_patch_window(start_y_x):
    start, y, x = int(start_y_x[0]), int(start_y_x[1]), int(start_y_x[2])
    mm = np.memmap(memmap_path, dtype=np.float16, mode='r', shape=(T, H, W))
    patches = np.empty((history + 1, patch_h, patch_w), dtype=np.float32)
    for t in range(history + 1):
        frame = np.array(mm[start + t], dtype=np.float32)
        patches[t] = frame[y:y+patch_h, x:x+patch_w]
    del mm
    X = patches[:history][..., np.newaxis].astype(np.float32)
    Y = patches[history][..., np.newaxis].astype(np.float32)
    return X, Y

# Build tf.data pipeline
def tf_map_fn(idx_y_x):
    def _py(arr):
        X, Y = py_load_patch_window(arr)
        return X, Y
    X, Y = tf.py_function(func=_py, inp=[idx_y_x], Tout=(tf.float32, tf.float32))
    X.set_shape((history, patch_h, patch_w, 1))
    Y.set_shape((patch_h, patch_w, 1))
    return X, Y

coords_ds = tf.data.Dataset.from_tensor_slices(coords)
coords_ds = coords_ds.shuffle(buffer_size=min(20000, total_samples), reshuffle_each_iteration=True)
coords_ds = coords_ds.map(tf_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

split = int(total_samples * 0.8)
train_ds = coords_ds.take(split).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = coords_ds.skip(split).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Model definition
model = Sequential([
    ConvLSTM2D(8, (3,3), padding='same', return_sequences=True, activation='tanh', input_shape=(history, patch_h, patch_w, 1)),
    BatchNormalization(),
    ConvLSTM2D(4, (3,3), padding='same', return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Conv2D(1, (3,3), padding='same', activation='linear', dtype='float32')
])

# Compile model
initial_lr = 5e-4
opt = tf.keras.optimizers.Adam(learning_rate=initial_lr)
model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# Callbacks
chkpt = ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

# Train model
history_model = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early, chkpt, reduce])