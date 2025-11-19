import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Set the style for seaborn
sns.set_theme(style='whitegrid')

# Load the dataset
def load_data(file_path):
    ds = xr.open_dataset(file_path)
    return ds

# Example file path (update this to your actual file path)
file_path = 'data/317_sst-anomaly_2025-11-13.nc'
dataset = load_data(file_path)

# Display the dataset information
print(dataset)

# Visualize the sea surface temperature anomaly
arr = np.nan_to_num(np.squeeze(dataset["sea_surface_temperature_anomaly"].values), nan=0.0).astype(np.float32)
gmin, gmax = float(arr.min()), float(arr.max())

cmap = plt.get_cmap('bwr')
cmap.set_bad('white')              # make NaNs white
norm = mcolors.TwoSlopeNorm(vmin=float(gmin), vcenter=0.0, vmax=float(gmax))

print(f"Global min: {gmin}, Global max: {gmax}")

def plot_sst_anomaly(ds):
    sst_anomaly = ds['sea_surface_temperature_anomaly'].isel(time=0)
    plt.figure(figsize=(12, 6))
    plt.title('Sea Surface Temperature Anomaly')
    plt.imshow(sst_anomaly, cmap=cmap, origin='lower', norm=norm, aspect='auto')
    plt.colorbar(label='Temperature Anomaly (°C)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Call the plotting function
plot_sst_anomaly(dataset)

# Further analysis can be added here, such as time series analysis, correlation studies, etc.