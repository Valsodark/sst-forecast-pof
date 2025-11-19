from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, glob, io, base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import xarray as xr
from skimage.transform import resize
import uvicorn
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")

# --- Configuration (must match your preprocessing) ---
folder = "data"
memmap_path = "sst_memmap.dat"
history = 5
down_H, down_W = 512, 1080
patch_h, patch_w = 128, 128
stride_h, stride_w = 64, 64
predict_batch = 64

# --- FastAPI app + request model ---
app = FastAPI(title="SST Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    day_index: int  # temporal start index (0..n_windows-1)

# --- Utilities (adapted from your predict script) ---
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
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(counts > 0, acc / counts, np.nan)
    return avg, counts

def compute_extent(nc_files):
    """Infer (lon_min, lon_max, lat_min, lat_max) from a sample NetCDF."""
    ds = xr.open_dataset(nc_files[0])
    lon_name = None
    lat_name = None
    # try coords first, then data_vars
    for name in list(ds.coords) + list(ds.data_vars):
        n = name.lower()
        if 'lon' in n and lon_name is None:
            lon_name = name
        if 'lat' in n and lat_name is None:
            lat_name = name
    if lon_name is None or lat_name is None:
        ds.close()
        raise RuntimeError("Couldn't find lon/lat coordinates in NetCDF")
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    ds.close()
    # handle 2D coordinate arrays by flattening
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    return [lon_min, lon_max, lat_min, lat_max]
# ...existing code...

# --- Startup: load model, files, computed values once ---
nc_files = sorted(glob.glob(os.path.join(folder, "*.nc")))
T = len(nc_files)
if T <= history:
    raise RuntimeError(f"Not enough files in {folder} (need > {history})")

global_min, global_max = compute_global_min_max(nc_files)
coords, n_y, n_x, n_windows = build_coords(T)
# infer geographic extent and store globally
global_extent = compute_extent(nc_files)

def render_png_base64(array2d, vmin, vmax, extent=None):
    cmap = plt.get_cmap('bwr').copy()
    cmap.set_bad('white')
    norm = mcolors.TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))

    fig = plt.figure(figsize=(20,12))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor('white')

    # use provided extent or the inferred global_extent
    if extent is None:
        lon0, lon1, lat0, lat1 = global_extent
    else:
        lon0, lon1, lat0, lat1 = extent

    # ensure lon/lat ordering consistent for imshow/extent
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())

    ax.imshow(array2d, origin='lower', cmap=cmap, norm=norm,
              interpolation='nearest', extent=[lon0, lon1, lat0, lat1],
              transform=ccrs.PlateCarree(), zorder=0, alpha=0.95)

    # coastlines on top
    ax.coastlines(resolution='110m', linewidth=1.2, color='black', zorder=30)

    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{encoded}"

# --- Startup: load model, files, computed values once ---
nc_files = sorted(glob.glob(os.path.join(folder, "*.nc")))
T = len(nc_files)
if T <= history:
    raise RuntimeError(f"Not enough files in {folder} (need > {history})")

global_min, global_max = compute_global_min_max(nc_files)
coords, n_y, n_x, n_windows = build_coords(T)

model_path = "models/best_model.keras"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

# --- API endpoint ---
@app.post("/predict")
def predict(req: PredictRequest):
    body = req.model_dump_json()
    print(f"Received request: {body}")
    day_index = int(req.day_index)
    if day_index < 0 or day_index >= n_windows:
        raise HTTPException(status_code=400, detail=f"day_index must be in [0, {n_windows-1}]")

    coords_for_start = coords[coords[:,0] == day_index]
    if coords_for_start.size == 0:
        raise HTTPException(status_code=404, detail="No patches for requested day_index")

    preds_list = []
    coords_used = []
    for i in range(0, coords_for_start.shape[0], predict_batch):
        batch_coords = coords_for_start[i:i+predict_batch]
        Xb = load_batch_X(batch_coords, T)
        pred_b = model.predict(Xb, verbose=0)
        preds_list.append(pred_b)
        coords_used.extend(batch_coords.tolist())
    preds_all = np.concatenate(preds_list, axis=0)

    pred_full_norm, counts = reconstruct_full_frame_from_patches(preds_all, coords_used, H=down_H, W=down_W)
    pred_full_un = pred_full_norm * (global_max - global_min + 1e-6) + global_min
    pred_full_un = np.asarray(pred_full_un, dtype=np.float32)

    img_b64 = render_png_base64(pred_full_un, vmin=global_min, vmax=global_max)

    return {"min_temperature": float(np.nanmin(pred_full_un)), "max_temperature": float(np.nanmax(pred_full_un)), "image": img_b64}

# Run with: uvicorn src.app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)