# Sea Surface Temperature Anomaly Forecasting

This project aims to develop an AI model that predicts sea surface temperature anomalies one week ahead using historical data stored in NetCDF files. The model is trained on processed data and is designed to assist in understanding and forecasting climate patterns.

## Project Structure

- **src/**: Contains the main source code for the project.
  - **train_module.py**: Main training logic for the AI model.
  - **data_loader.py**: Responsible for loading and preprocessing data from NetCDF files.
  - **model.py**: Defines the architecture of the AI model.
  - **utils.py**: Contains utility functions for data handling and model support.

- **data/**: Directory for storing data.
  - **raw/**: Contains raw NetCDF files with sea surface temperature anomaly data.
  - **processed/**: Stores processed data ready for training.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - **exploration.ipynb**: Notebook for visualizing and analyzing the sea surface temperature anomaly data.

- **models/**: Directory where trained models will be saved.

- **requirements.txt**: Lists the dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sst-forecast-poc
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your raw NetCDF files in the `data/raw/` directory.

## Usage

To train the model, run the following command:
```
python src/train_module.py
```

This will load the data, preprocess it, and train the model to predict sea surface temperature anomalies one week ahead. The trained model will be saved in the `models/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.