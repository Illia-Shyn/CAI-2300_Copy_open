# Machine Learning Model Overview

This document provides a detailed summary of the data science and machine learning workflow used in the Miami Real Estate Value Analyzer project.

## 1. The Goal: Predicting Property Value

The primary objective of the machine learning model is to predict the sale price of a residential property in the Miami-Dade area based on its core features. The model is designed to provide a data-driven baseline valuation that can then be adjusted for real-time market conditions.

## 2. The Data

### Primary Dataset
- **Source:** `data/miami-housing.csv`
- **Content:** A rich dataset containing ~14,000 individual property transactions from 2016.
- **Key Features Used:**
    - `TOT_LVG_AREA`: The total interior living space in square feet.
    - `LND_SQFOOT`: The total area of the land/lot in square feet.
    - `age`: The age of the property in years.
    - `structure_quality`: A numerical rating of the building's quality.
    - `OCEAN_DIST`, `WATER_DIST`, `CNTR_DIST`: Distances to the ocean, water, and city center.

### Supporting Datasets
- **`data/MEDLISPRI12086.csv`:** Used to provide a visual time-series context for the historical market median price.
- **`data/miami-housing.csv` (for calibration):** The full dataset was used to calculate a mean-to-median ratio to calibrate the model's output for statistical skew.

## 3. The Machine Learning Workflow

The project followed a standard, robust machine learning workflow to ensure the model was both accurate and reliable.

### Step 1: Feature Engineering
- **Logarithmic Transformation (Target Variable):** The model was trained to predict the natural logarithm of the sale price (`np.log1p(SALE_PRC)`). This is a critical technique for price prediction, as it helps the model learn percentage-based relationships and better handles the wide, skewed distribution of housing prices.
- **Logarithmic Transformation (Distance Features):** The distance features (`OCEAN_DIST`, etc.) were also log-transformed. This makes the model more sensitive to changes when a property is very close to a key location and less sensitive to changes when it is already far away, reflecting real-world market dynamics.

### Step 2: Data Cleaning
- **Missing Value Imputation:** The training script performs a robust data cleaning step. Any missing values in the feature columns are filled with the **median** value of that column. This ensures the model is trained on a complete and clean dataset, which was crucial for fixing the erratic prediction behavior observed in earlier iterations.

### Step 3: Model Selection & Training
- **Algorithm:** We used a **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`).
- **Why Random Forest?:** This algorithm is powerful, versatile, and highly effective for tabular data. It is an ensemble of many decision trees, which makes it robust to outliers and capable of capturing complex, non-linear relationships between a property's features and its final sale price.
- **Training:** The model was trained on an 80/20 split of the cleaned, engineered dataset.

## 4. Model Evaluation & Performance

The model's performance was evaluated on the 20% holdout test set. To make the metrics intuitive, the model's logarithmic predictions were converted back to their original dollar scale (`np.expm1()`) before being compared to the actual sale prices.

- **R² Score: 0.898**
    - This indicates that our model can explain approximately **89.8%** of the variance in property prices in the test set. This is a very strong result for a real estate model.
- **Mean Absolute Error (MAE): ~$51,873**
    - On average, the model's price prediction is off by about $51,873. Given the high value of Miami real estate, this represents a strong level of accuracy.

## 5. Post-Modeling Calibration

A key part of this project was acknowledging that a model trained on 2016 data cannot know about today's market. To solve this, two data-driven calibration factors were applied in the Streamlit application:

1.  **Market Correction Factor (1.8x):** Based on external research confirming that the market has appreciated by at least 80% since 2016, this factor scales the model's 2016-era predictions up to current 2024/2025 market values.
2.  **Calibration Factor (~1.25x):** This factor was calculated by dividing the mean sale price by the median sale price of the training data. It adjusts the model's median-based log predictions to better reflect the mean-skewed nature of the real estate market.

These two factors, combined with the core ML model, create a final prediction that is both historically grounded and calibrated for today's real-world market conditions. 