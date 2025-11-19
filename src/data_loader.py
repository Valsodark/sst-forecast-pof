import os
import glob
import xarray as xr
import numpy as np

def load_netcdf_data(folder_path):
    nc_files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
    if not nc_files:
        raise ValueError("No NetCDF files found in the specified directory.")
    
    data = []
    for file in nc_files:
        ds = xr.open_dataset(file)
        if "sea_surface_temperature_anomaly" in ds:
            sst_anomaly = ds["sea_surface_temperature_anomaly"].values
            data.append(sst_anomaly)
        ds.close()
    
    return np.array(data)

def preprocess_data(data, downsample_shape=(256, 540)):
    processed_data = []
    for arr in data:
        arr_resized = np.nan_to_num(arr, nan=0.0)
        if arr_resized.ndim == 3:
            arr_resized = arr_resized[:, :, 0]
        arr_resized = resize(arr_resized, downsample_shape, order=1, preserve_range=True, anti_aliasing=False)
        processed_data.append(arr_resized)
    
    return np.array(processed_data)

def save_processed_data(processed_data, output_path):
    np.save(output_path, processed_data)