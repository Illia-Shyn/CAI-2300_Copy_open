# Real Estate Listing Analyzer

## Problem Statement
Homebuyers and agents are overwhelmed by lengthy, unstructured property listings. Our solution uses AI-driven market analytics and Generative AI to transform property data into concise, actionable insights—improving confidence and decision-making in fast-moving markets like Miami.

## Features
- **Machine Learning Price Prediction:** A Random Forest model predicts property values based on key features like size, age, and location.
- **AI-Powered Insights:** Utilizes a Large Language Model (LLM) via OpenRouter to generate qualitative summaries, including positive attributes and potential considerations for a property.
- **Interactive UI:** A Streamlit application allows users to input property details, receive an instant valuation, and assess a property's value against historical market trends.
- **Data-Driven Calibration:** The model is calibrated with multiple factors to account for market appreciation and statistical skew, ensuring estimates are aligned with current market conditions.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the model training script:**
    This will generate the necessary model files in the `docs/` directory.
    ```bash
    python3 train_new_model.py
    ```

5.  **Run the Streamlit application:**
    ```bash
    cd ui
    streamlit run app.py
    ```

## Data Source
The primary dataset used for training (`miami-housing.csv`) is publicly available and was sourced for academic purposes. It contains property transaction data from the Miami-Dade area. Note that this data includes columns such as `PARCELNO`, `LATITUDE`, and `LONGITUDE`, which could be considered sensitive in a commercial context. 