import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
DATA_DIR = r"C:\Desktop\EY\UHI_Model"
FILE_PATHS = {
    "uhi": os.path.join(DATA_DIR, "Training_data_uhi_index.csv"),
    "ndvi": os.path.join(DATA_DIR, "sentinel2_ndvi.csv"),
    "lst": os.path.join(DATA_DIR, "landsat_lst.csv"),
    "submission": os.path.join(DATA_DIR, "Submission_template.csv"),
    "weather": os.path.join(DATA_DIR, "NY_Mesonet_Weather.xlsx"),
    "building": os.path.join(DATA_DIR, "Building_Footprint.kml")
}

# Load datasets with error handling
def load_data():
    try:
        uhi_df = pd.read_csv(FILE_PATHS["uhi"])
        ndvi_df = pd.read_csv(FILE_PATHS["ndvi"])
        lst_df = pd.read_csv(FILE_PATHS["lst"])
        submission_template = pd.read_csv(FILE_PATHS["submission"])
        weather_df = pd.read_excel(FILE_PATHS["weather"], engine='openpyxl')
        buildings = gpd.read_file(FILE_PATHS["building"], driver="KML")
        logging.info("All datasets loaded successfully.")
        return uhi_df, ndvi_df, lst_df, submission_template, weather_df, buildings
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

uhi_df, ndvi_df, lst_df, submission_template, weather_df, buildings = load_data()

# Standardize column names
rename_cols = {'longitude': 'Longitude', 'latitude': 'Latitude'}
ndvi_df.rename(columns=rename_cols, inplace=True)
lst_df.rename(columns=rename_cols, inplace=True)

# Ensure 'datetime' column is renamed if needed
if 'datetime' in uhi_df.columns:
    uhi_df.rename(columns={'datetime': 'Date'}, inplace=True)

# Convert Date column to datetime safely
if 'Date' in uhi_df.columns:
    uhi_df['Date'] = pd.to_datetime(uhi_df['Date'], errors='coerce', dayfirst=True)

    # Drop NaT values if conversion fails
    uhi_df = uhi_df.dropna(subset=['Date'])

    # Extract datetime components
    uhi_df['Year'] = uhi_df['Date'].dt.year
    uhi_df['Month'] = uhi_df['Date'].dt.month
    uhi_df['Day'] = uhi_df['Date'].dt.day
    uhi_df['Hour'] = uhi_df['Date'].dt.hour
else:
    logging.error("Column 'Date' (or 'datetime') is missing in uhi_df. Available columns:")
    print(uhi_df.columns)

# Merge datasets again
final_df = uhi_df.merge(ndvi_df, on=['Longitude', 'Latitude'], how='left')
final_df = final_df.merge(lst_df, on=['Longitude', 'Latitude'], how='left')

# Verify the presence of the extracted datetime columns
print(final_df[['Year', 'Month', 'Day', 'Hour']].head())


# Merge datasets
final_df = uhi_df.merge(ndvi_df, on=['Longitude', 'Latitude'], how='left')
final_df = final_df.merge(lst_df, on=['Longitude', 'Latitude'], how='left')

print(final_df.columns)

# Handle missing values before calculations
final_df.fillna(final_df.mean(numeric_only=True), inplace=True)

# Feature Engineering
final_df.loc[:, 'lwir11'] = final_df['lwir11'].replace(0, np.nan)
final_df['NDVI_LST_Ratio'] = final_df['NDVI'] / (final_df['lwir11'].fillna(final_df['lwir11'].mean()) + 1e-6)
final_df['NDVI_Squared'] = final_df['NDVI'] ** 2
final_df['LST_Squared'] = final_df['lwir11'] ** 2

# Select features & target (including temporal features)
X = final_df[['Longitude', 'Latitude', 'NDVI', 'lwir11', 'NDVI_LST_Ratio', 'NDVI_Squared', 'LST_Squared', 'Year', 'Month', 'Day', 'Hour']]
y = final_df['UHI Index']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
param_grid = {
    'num_leaves': [20, 30, 40],
    'max_depth': [10, 20, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

lgbm = lgb.LGBMRegressor()
random_search = RandomizedSearchCV(lgbm, param_grid, scoring='r2', cv=3, n_iter=20, random_state=42)
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
logging.info(f'Model Performance -> R² Score: {r2:.4f}, RMSE: {rmse:.4f}')

# Prepare submission
X_submission = submission_template[['Longitude', 'Latitude']]
X_submission = X_submission.merge(final_df, on=['Longitude', 'Latitude'], how='left')
X_submission = X_submission[['Longitude', 'Latitude', 'NDVI', 'lwir11', 'NDVI_LST_Ratio', 'NDVI_Squared', 'LST_Squared', 'Year', 'Month', 'Day', 'Hour']]
X_submission.fillna(X_submission.mean(numeric_only=True), inplace=True)

X_submission = X_submission.copy()  # Ensure it's a new DataFrame
X_submission['Year'] = X_submission['Year'].fillna(2021)
X_submission['Month'] = X_submission['Month'].fillna(7)
X_submission['Day'] = X_submission['Day'].fillna(24)
X_submission['Hour'] = X_submission['Hour'].fillna(12)


submission_template['Predicted_UHI_Index'] = best_model.predict(X_submission)

# Save submission
submission_output = os.path.join(DATA_DIR, "submission.csv")
submission_template.to_csv(submission_output, index=False)
logging.info(f'Submission file saved successfully at {submission_output}')

import matplotlib.pyplot as plt
# Plot feature importance
feature_importance = best_model.feature_importances_
feature_names = X_train.columns
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance of LGBM Model")
plt.show()

import joblib
joblib.dump(best_model, os.path.join(DATA_DIR, "lgbm_model.pkl"))
logging.info("Model saved successfully.")

