import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# 📌 Define file paths
base_dir = r"C:\Desktop\EY\UHI_Model"

uhi_file = os.path.join(base_dir, "Training_data_uhi_index.csv")
ndvi_file = os.path.join(base_dir, "sentinel2_ndvi.csv")
lst_file = os.path.join(base_dir, "landsat_lst.csv")
submission_template_file = os.path.join(base_dir, "Submission_template.csv")

# 📌 Load datasets
uhi_df = pd.read_csv(uhi_file)
ndvi_df = pd.read_csv(ndvi_file)
lst_df = pd.read_csv(lst_file)
submission_template = pd.read_csv(submission_template_file)

print("✅ Files loaded successfully!")

# 📌 Standardize column names
ndvi_df.rename(columns={"longitude": "Longitude", "latitude": "Latitude"}, inplace=True)
lst_df.rename(columns={"longitude": "Longitude", "latitude": "Latitude"}, inplace=True)

# 📌 Merge datasets
uhi_ndvi = pd.merge_asof(uhi_df.sort_values("Longitude"), ndvi_df.sort_values("Longitude"), on="Longitude", direction="nearest")
final_df = pd.merge_asof(uhi_ndvi.sort_values("Longitude"), lst_df.sort_values("Longitude"), on="Longitude", direction="nearest")

# ✅ Ensure `final_df` is created before using it
print("✅ Merged Dataset Successfully!")
print("Columns in final_df before merging:", final_df.columns.tolist())  # Print columns to check

# ✅ Fix Duplicate Latitude Column
if "Latitude_x" in final_df.columns and "Latitude_y" in final_df.columns:
    final_df.rename(columns={"Latitude_x": "Latitude"}, inplace=True)
    final_df.drop(columns=["Latitude_y"], errors='ignore', inplace=True)

# ✅ Ensure unique column names
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

print("✅ Updated final_df columns:", final_df.columns.tolist())

# 📌 Select Features & Target
X = final_df[['Longitude', 'Latitude', 'NDVI', 'lwir11']]  # Features
y = final_df['UHI Index']  # Target variable

# 📌 Handle missing values
X = X.fillna(X.mean())  # ✅ Correct way to avoid warning

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train RandomForest Model with Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

print("✅ Best Model Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 📌 Evaluate Model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"✅ Model Performance:\n R² Score: {r2:.4f}\n RMSE: {rmse:.4f}")

# 📌 Predict UHI Index for Submission Locations
X_submission = submission_template[['Longitude', 'Latitude']].copy()

# ✅ Drop duplicate columns in `final_df` before merging
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# ✅ Merge with submission template
X_submission = X_submission.merge(final_df, on=["Longitude", "Latitude"], how="left", suffixes=('', '_drop'))

# ✅ Drop unnecessary "_drop" columns if created
X_submission = X_submission.loc[:, ~X_submission.columns.str.endswith('_drop')]

# 📌 Select required columns
X_submission = X_submission[['Longitude', 'Latitude', 'NDVI', 'lwir11']]

# 📌 Handle missing values
X_submission.fillna(X_submission.mean(), inplace=True)

# 📌 Generate Predictions
submission_template['Predicted_UHI_Index'] = best_model.predict(X_submission)

# 📌 Save submission file
submission_template.to_csv(os.path.join(base_dir, "submission.csv"), index=False)


# Save submission file correctly, ensuring no extra columns
submission_template.to_csv("C:\\Desktop\\EY\\UHI_Model\\submission.csv", index=False, columns=['Longitude', 'Latitude', 'UHI Index'])

print("✅ Submission file saved successfully with correct Longitude & Latitude!")
