import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import StackingRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from scipy.spatial import cKDTree
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
DATA_DIR = r"C:\Desktop\EY\UHI_Model"
FILE_PATHS = {
    "uhi": os.path.join(DATA_DIR, "Training_data_uhi_index.csv"),
    "ndvi": os.path.join(DATA_DIR, "sentinel2_ndvi.csv"),
    "lst": os.path.join(DATA_DIR, "landsat_lst.csv"),
    "weather": os.path.join(DATA_DIR, "NY_Mesonet_Weather.xlsx"),
    "building": os.path.join(DATA_DIR, "Building_Footprint.kml")
}

# Load datasets
def load_data():
    try:
        uhi_df = pd.read_csv(FILE_PATHS["uhi"])
        ndvi_df = pd.read_csv(FILE_PATHS["ndvi"])
        lst_df = pd.read_csv(FILE_PATHS["lst"])
        weather_xls = pd.ExcelFile(FILE_PATHS["weather"], engine='openpyxl')
        buildings = gpd.read_file(FILE_PATHS["building"], driver="KML")
        logging.info("All datasets loaded successfully.")
        return uhi_df, ndvi_df, lst_df, weather_xls, buildings
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

uhi_df, ndvi_df, lst_df, weather_xls, buildings = load_data()

# Standardize column names
rename_cols = {'latitude': 'Latitude', 'longitude': 'Longitude'}
ndvi_df.rename(columns=rename_cols, inplace=True)
lst_df.rename(columns=rename_cols, inplace=True)

# Process Date Column in UHI Data
uhi_df.rename(columns={'datetime': 'Date'}, inplace=True)
uhi_df['Date'] = pd.to_datetime(uhi_df['Date'], errors='coerce', dayfirst=True)
uhi_df['Year'] = uhi_df['Date'].dt.year
uhi_df['Month'] = uhi_df['Date'].dt.month
uhi_df['Day'] = uhi_df['Date'].dt.day
uhi_df['Hour'] = uhi_df['Date'].dt.hour
uhi_df['Minute'] = uhi_df['Date'].dt.minute

# Fourier Time Features
uhi_df['Sin_Hour'] = np.sin(2 * np.pi * uhi_df['Hour'] / 24)
uhi_df['Cos_Hour'] = np.cos(2 * np.pi * uhi_df['Hour'] / 24)

# Load & Process Weather Data
bronx_weather = pd.read_excel(weather_xls, sheet_name="Bronx", engine="openpyxl")
manhattan_weather = pd.read_excel(weather_xls, sheet_name="Manhattan", engine="openpyxl")

# Rename 'Date / Time' to 'Date'
bronx_weather.rename(columns={"Date / Time": "Date"}, inplace=True)
manhattan_weather.rename(columns={"Date / Time": "Date"}, inplace=True)

bronx_weather["Date"] = pd.to_datetime(bronx_weather["Date"].str[:10], errors='coerce')
manhattan_weather["Date"] = pd.to_datetime(manhattan_weather["Date"].str[:10], errors='coerce')

# Merge weather data
weather_df = pd.concat([bronx_weather, manhattan_weather], ignore_index=True).dropna(subset=["Date"])

# Merge datasets
final_df = uhi_df.merge(ndvi_df, on=['Longitude', 'Latitude'], how='left')
final_df = final_df.merge(lst_df, on=['Longitude', 'Latitude'], how='left')
final_df = final_df.merge(weather_df, on=['Date'], how='left')

# Feature Engineering
final_df['NDVI_LST_Interaction'] = final_df['NDVI'] * final_df['lwir11']
final_df['NDVI_LST_Ratio'] = final_df['NDVI'] / (final_df['lwir11'] + 1e-6)
final_df['NDVI_Squared'] = final_df['NDVI'] ** 2
final_df['LST_Squared'] = final_df['lwir11'] ** 2
final_df['LST_Log'] = np.log1p(final_df['lwir11'])
final_df['Is_Daytime'] = ((final_df['Hour'] >= 6) & (final_df['Hour'] <= 18)).astype(int)

# Compute Building Density
buildings = buildings.to_crs(epsg=3857)  # Convert to projected CRS
buildings['geometry'] = buildings['geometry'].centroid
tree = cKDTree(np.vstack([buildings.geometry.x, buildings.geometry.y]).T)
dists, _ = tree.query(np.vstack([final_df.Longitude, final_df.Latitude]).T, k=10)
final_df['Building_Density'] = 1 / (np.mean(dists, axis=1) + 1e-6)

from sklearn.impute import SimpleImputer

# List of columns for polynomial transformation
poly_columns = ['NDVI', 'lwir11', 'Building_Density']

# Check for missing values before imputation
missing_values = final_df[poly_columns].isnull().sum()
print(f"🚨 Missing Values Before Imputation:\n{missing_values}")

# Manually fill completely missing columns with 0 before imputation
for col in poly_columns:
    if final_df[col].isnull().all():  
        print(f"⚠️ {col} has all NaNs! Filling with 0.")
        final_df[col].fillna(0, inplace=True)  

# Apply imputation again after manual filling
imputer = SimpleImputer(strategy="mean")
final_df[poly_columns] = imputer.fit_transform(final_df[poly_columns])

# Verify missing values are removed
missing_values_after = final_df[poly_columns].isnull().sum()
print(f"✅ Missing Values After Imputation:\n{missing_values_after}")

# Apply Polynomial Features Transformation
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(final_df[poly_columns])
poly_feature_names = poly.get_feature_names_out(poly_columns)

# Convert to DataFrame and merge back
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=final_df.index)
final_df = pd.concat([final_df, poly_df], axis=1)


# Feature Selection & Normalization
features = final_df.select_dtypes(include=[np.number]).columns.tolist()
X = final_df[features]
y = final_df['UHI Index']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))  # Save the trained scaler
logging.info("✅ Scaler saved successfully.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Optimization using Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 6, 16),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

# Stacking Model with Ridge Final Estimator
stacked_model = StackingRegressor(
    estimators=[
        ('lgbm', lgb.LGBMRegressor(**best_params)),
        ('xgb', xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=12)),
        ('cat', cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=12, verbose=0))
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)

# Evaluate Model
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
logging.info(f'Model Performance -> R² Score: {r2:.4f}, RMSE: {rmse:.4f}')

joblib.dump(stacked_model, os.path.join(DATA_DIR, "best_uhi_model.pkl"))
logging.info("Best model saved successfully.")
