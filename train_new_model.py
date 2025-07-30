import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json
import os

# --- 1. Load and Prepare Data ---
print("Loading Miami housing data...")
df = pd.read_csv('data/miami-housing.csv')
df_clean = df.copy()
df_clean.dropna(subset=['SALE_PRC', 'TOT_LVG_AREA', 'LND_SQFOOT'], inplace=True)
df_clean = df_clean[df_clean['SALE_PRC'] > 10000]

# --- 2. Feature Engineering & Log Transformations ---
print("Engineering features and applying log transforms...")
df_clean['log_OCEAN_DIST'] = np.log1p(df_clean['OCEAN_DIST'])
df_clean['log_WATER_DIST'] = np.log1p(df_clean['WATER_DIST'])
df_clean['log_CNTR_DIST'] = np.log1p(df_clean['CNTR_DIST'])
df_clean['log_SALE_PRC'] = np.log1p(df_clean['SALE_PRC'])

feature_columns = [
    'TOT_LVG_AREA', 'LND_SQFOOT', 'age', 'structure_quality',
    'log_OCEAN_DIST', 'log_WATER_DIST', 'log_CNTR_DIST'
]

X = df_clean[feature_columns]
y = df_clean['log_SALE_PRC']

# --- 3. ROBUST DATA CLEANING: Handle Missing Values ---
print("Handling missing values in the feature set...")
for col in feature_columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"  - Missing values in '{col}' filled with median value ({median_val:.2f})")

# --- 4. Train the Model ---
print("Training the Random Forest model on cleaned, transformed data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("Evaluating model performance on original price scale...")
y_pred_log = model.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

mae = np.mean(np.abs(y_pred_orig - y_test_orig))
r2 = 1 - (np.sum((y_test_orig - y_pred_orig)**2) / np.sum((y_test_orig - y_test_orig.mean())**2))

print(f"\n--- Model Evaluation (on Original Price Scale) ---")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: ${mae:,.0f}")
print("--------------------------------------------------\n")

# --- 6. Save the Model, Metadata, and Feature Importance ---
print("Saving updated model, metadata, and feature importance...")
os.makedirs("docs", exist_ok=True)
joblib.dump(model, 'docs/price_prediction_model.pkl')

# Feature Importance
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv('docs/feature_importance.csv', index=False)

# Metadata
model_metadata = {
    'model_type': 'RandomForestRegressor (Log-Transformed Target)',
    'features': feature_columns,
    'r2_score': float(r2),
    'mae': float(mae),
    'top_features': importance_df['feature'].head(3).tolist()
}
with open('docs/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ New model and assets saved to 'docs/' directory.")
print("\n🚀 Training complete!") 