📌 Project Overview
This project aims to predict Urban Heat Island (UHI) hotspots using machine learning models trained on satellite imagery and weather datasets. The goal is to achieve high accuracy (98%+ R² score) using Random Forest, LightGBM, and stacking models.

📂 UHI_Model/ │── 📂 data/ # Raw and processed datasets (if needed) │── 📂 models/ # Trained ML models (.pkl files) │── 📂 predictions/ # Predicted results in Excel (.xlsx files) │── 📂 src/ # Python scripts for different models │ │── model1.py # Model 1 training/testing script │ │── model2.py # Model 2 training/testing script │ │── model3.py # Model 3 training/testing script │── README.md # Project description │── .gitignore # Ignored files  │── requirements.txt #Requirements 


## 🚀 How to Use

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd ML-Project

Run a specific model

python src/model1.py
python src/model2.py
python src/model3.py

🛠 Requirements
numpy==1.24.3
pandas==2.0.3
geopandas==0.14.1
rasterio==1.3.8
scikit-learn==1.3.2
lightgbm==4.1.0
xgboost==2.0.3
catboost==1.2.2
matplotlib==3.8.2
joblib==1.3.2
optuna==3.4.0
openpyxl==3.1.2


Libraries: pandas, numpy, scikit-learn (if needed)

⚙️ Preprocessing & Feature Engineering
Date Processing: Extract Year, Month, Day, Hour from timestamps.

Fourier Transforms: Creates Sin_Hour and Cos_Hour for time-based learning.

Spatial Merging: Combines NDVI, LST, and Weather data.

Feature Engineering:

NDVI_LST_Ratio = NDVI / (LST + 1e-6)

NDVI_Squared = NDVI²

LST_Squared = LST²

LST_Log = log(1 + LST)

🏋️ Model Training

LightGBM Model

param_grid = {
    'num_leaves': [20, 30, 40],
    'max_depth': [10, 20, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
lgbm = lgb.LGBMRegressor()
random_search = RandomizedSearchCV(lgbm, param_grid, scoring='r2', cv=3, n_iter=20)
random_search.fit(X_train, y_train)

PICTURE
![alt text](Figure_1_LGM_Model.png)


Stacking Model

stacked_model = StackingRegressor(
    estimators=[
        ('lgbm', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03)),
        ('xgb', xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03)),
        ('cat', cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, verbose=0))
    ],
    final_estimator=Ridge(alpha=1.0)
)
stacked_model.fit(X_train, y_train)

🎯 Model Performance
Model	R² Score	RMSE
Random Forest	0.965	0.52
LightGBM	0.982	0.38
Stacking (Best)	0.9935	0.25

📤 Generating Predictions

X_submission = submission_template[['Longitude', 'Latitude']]
X_submission = X_submission.merge(final_df, on=['Longitude', 'Latitude'], how='left')
submission_template['Predicted_UHI_Index'] = best_model.predict(X_submission)
submission_template.to_csv("submission.csv", index=False)

📈 Feature Importance

import matplotlib.pyplot as plt
feature_importance = best_model.feature_importances_
feature_names = X_train.columns
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance of LGBM Model")
plt.show()

📌 To-Do / Future Work
Incorporate Deep Learning Models (CNNs for Satellite Images).

Enhance Feature Engineering with Advanced Climate Indicators.

Fine-tune Hyperparameters for Further Optimization.

Integrate a Web Interface for Visualization.

📝 Conclusion
This project effectively predicts UHI hotspots using ML models, with the stacking model achieving 99.35% accuracy. The model is designed for real-world applications in urban climate management and environmental planning.

📬 For any queries, feel free to reach out! 🚀