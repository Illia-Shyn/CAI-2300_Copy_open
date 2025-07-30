import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import os
from openai import OpenAI

# --- Page Configuration ---
st.set_page_config(
    page_title="Miami Real Estate Value Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .value-badge {
        text-align: center;
        padding: 0.5rem;
        border-radius: 0.5rem;
        color: white;
    }
    .ai-summary {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6a0dad; /* A nice purple for AI */
        color: #333; /* Ensure text is a dark color */
    }
</style>
""", unsafe_allow_html=True)

# --- Data and Model Loading ---
@st.cache_data
def load_assets_v2(): # Renamed to ensure cache was cleared.
    """Load all necessary assets and calculate calibration factors."""
    try:
        # --- FIX: Construct paths relative to the script's location ---
        # This makes the app runnable from any directory.
        script_dir = os.path.dirname(__file__)
        docs_path = os.path.join(script_dir, '../docs')
        data_path = os.path.join(script_dir, '../data')
        
        model_file = os.path.join(docs_path, 'price_prediction_model.pkl')
        metadata_file = os.path.join(docs_path, 'model_metadata.json')
        historical_file = os.path.join(data_path, 'MEDLISPRI12086.csv')
        base_data_file = os.path.join(data_path, 'miami-housing.csv')
        
        required_files = [model_file, metadata_file, historical_file, base_data_file]
        if not all(os.path.exists(f) for f in required_files):
            st.error("One or more required files are missing. Please run the training script.")
            return None, None, None, None, None
            
        model = joblib.load(model_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        historical_data = pd.read_csv(historical_file)
        historical_data['observation_date'] = pd.to_datetime(historical_data['observation_date'])
        
        market_correction_factor = 1.80

        base_df = pd.read_csv(base_data_file)
        mean_price = base_df['SALE_PRC'].mean()
        median_price = base_df['SALE_PRC'].median()
        mean_to_median_ratio = mean_price / median_price
            
        return model, metadata, historical_data, market_correction_factor, mean_to_median_ratio
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None, None

# --- API Call to LLM ---
def get_ai_insights(property_data):
    """Generates property insights using an LLM from OpenRouter."""
    if not st.session_state.get("openrouter_api_key"):
        st.warning("Please enter your OpenRouter API key in the sidebar to use this feature.")
        return None

    try:
        # Correctly initialize the client to point to the OpenRouter API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=st.session_state.openrouter_api_key,
        )
        
        prompt = f"""
        You are an expert real estate analyst in Miami. Based on the following data, generate a short, insightful summary for a potential homebuyer. Structure your response with three sections: "Property Overview," "Positive Attributes," and "Potential Considerations." Be concise and professional. Do not give financial advice.

        **Property Data:**
        - Living Area: {property_data['Living Area']}
        - Land Area: {property_data['Land Area']}
        - Age: {property_data['Age']}
        - Has Pool: {'Yes' if property_data['Has Pool'] else 'No'}
        - Has Garage: {'Yes' if property_data['Has Garage'] else 'No'}
        - Distance to Ocean: {property_data['Distance to Ocean']}
        - Our Estimated Value: ${property_data['Estimated Value']:,.0f}
        - Listed Price: ${property_data['Listed Price']:,.0f}
        """

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Could not connect to OpenRouter. Error: {e}")
        return None

# --- Price Prediction ---
def predict_price(model, features, feature_order):
    """Make price prediction and handle log-transformed output."""
    try:
        features['log_OCEAN_DIST'] = np.log1p(features.pop('OCEAN_DIST'))
        features['log_WATER_DIST'] = np.log1p(features.pop('WATER_DIST'))
        features['log_CNTR_DIST'] = np.log1p(features.pop('CNTR_DIST'))
        
        input_df = pd.DataFrame([features], columns=feature_order)
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
        return prediction, None
    except Exception as e:
        return None, str(e)

# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">🏠 Miami Real Estate Value Analyzer</h1>', unsafe_allow_html=True)
    
    model, metadata, historical_data, market_correction_factor, calibration_factor = load_assets_v2()
    
    if model is None or historical_data.empty:
        st.stop()
        
    # --- Sidebar ---
    st.sidebar.title("Configuration")
    st.sidebar.markdown("### 🤖 GenAI Feature")
    st.session_state.openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Get a free key from openrouter.ai")

    st.sidebar.markdown("### Market Factors")
    st.sidebar.info(f"**Market Correction:** {market_correction_factor:.2f}x (since 2016)")
    st.sidebar.info(f"**Calibration Factor:** {calibration_factor:.2f}x (model skew adj.)")
    
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Model Details")
    st.sidebar.metric("Model R² Score", f"{metadata['r2_score']:.2f}")
    st.sidebar.metric("Mean Prediction Error", f"${metadata['mae']:,.0f}")
    
    # --- Main UI ---
    st.markdown("### Enter Property Details to Estimate Value")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Core Property Features")
        tot_lvg_area = st.number_input("Total Living Area (sq ft)", 500, 3000, 2000, 100)
        lnd_sqfoot = st.number_input("Land Area (sq ft)", 1000, 50000, 5000, 500)
        age = st.number_input("Property Age (years)", 0, 100, 20, 1)
        structure_quality = st.select_slider("Structure Quality", [1, 2, 3, 4, 5], 4)
        
    with col2:
        st.subheader("Location & Amenities")
        ocean_dist = st.number_input("Distance to Ocean (ft)", 0, 50000, 10000, 1000)
        water_dist = st.number_input("Distance to Water (ft)", 0, 10000, 500, 100)
        cntr_dist = st.number_input("Distance to City Center (ft)", 0, 100000, 40000, 1000)
        st.markdown("---")
        has_pool = st.toggle("Has a Pool? (+8%)", False)
        has_garage = st.toggle("Has a Garage? (+7%)", False)
        
    st.markdown("---")
    st.subheader("Value Assessment")
    listed_price = st.number_input("Enter Listing Price for Comparison ($)", 0, step=10000)
    
    if st.button("🚀 Analyze Property Value", type="primary"):
        # When main button is clicked, reset AI insights and run analysis
        if 'ai_insights' in st.session_state:
            del st.session_state.ai_insights
            
        with st.spinner("Running analysis..."):
            features = {
                'TOT_LVG_AREA': tot_lvg_area, 'LND_SQFOOT': lnd_sqfoot,
                'age': age, 'structure_quality': structure_quality,
                'OCEAN_DIST': ocean_dist, 'WATER_DIST': water_dist, 'CNTR_DIST': cntr_dist,
            }
            
            base_price, error = predict_price(model, features.copy(), metadata['features'])
            
            if error:
                st.error(f"Prediction failed: {error}")
                st.session_state.analysis_run = False
            else:
                final_price = base_price * market_correction_factor * calibration_factor
                if has_pool: final_price *= 1.08
                if has_garage: final_price *= 1.07

                # Store all results in session state to persist them across reruns
                st.session_state.last_analysis = {
                    "Living Area": f"{tot_lvg_area:,} sq ft", "Land Area": f"{lnd_sqfoot:,} sq ft",
                    "Age": f"{age} years", "Has Pool": has_pool, "Has Garage": has_garage,
                    "Distance to Ocean": f"{ocean_dist:,} ft", "Estimated Value": final_price,
                    "Listed Price": listed_price
                }
                st.session_state.analysis_run = True

    # --- Persistent Results Section ---
    # This block now controls the visibility of ALL results based on session state.
    if st.session_state.get("analysis_run", False):
        analysis_data = st.session_state.last_analysis
        final_price = analysis_data["Estimated Value"]
        listed_price = analysis_data["Listed Price"]

        st.markdown("### 📊 Analysis Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Our Estimated Market Value", f"${final_price:,.0f}")
        
        if listed_price > 0:
            with res_col2:
                diff = final_price - listed_price
                diff_pct = (diff / listed_price) * 100 if listed_price > 0 else 0
                
                if abs(diff_pct) < 5: badge_text, color = "Fair Value", "green"
                elif diff_pct > 0: badge_text, color = "Potential Good Deal", "blue"
                else: badge_text, color = "Potentially Overpriced", "orange"
                    
                st.markdown(f'<div class="value-badge" style="background-color:{color};">{badge_text}</div>', unsafe_allow_html=True)
                st.write(f"Our estimate is **{abs(diff_pct):.1f}%** {'higher' if diff > 0 else 'lower'} than the list price.")

        st.markdown("---")
        st.markdown("#### How Your Property Compares to the Market")
        graph_df = pd.DataFrame({'date': historical_data['observation_date'], 'market_median': historical_data['MEDLISPRI12086']})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=graph_df['date'].tolist(), y=graph_df['market_median'].tolist(), mode='lines', name='Historical Median Price'))
        fig.add_trace(go.Scatter(x=[graph_df['date'].iloc[-1]], y=[final_price], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Property'))
        fig.update_layout(title="Your Property's Value vs. Historical Market Trend", yaxis_title="Price ($)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)

        # --- AI Insights Section (now inside the persistent results block) ---
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Property Insights")
        if st.button("Generate AI Insights"):
            with st.spinner("Contacting AI analyst..."):
                insights = get_ai_insights(st.session_state.last_analysis)
                if insights:
                    # Store insights in session state so they also persist
                    st.session_state.ai_insights = insights
        
        # Display insights if they have been generated and stored
        if "ai_insights" in st.session_state:
            st.markdown(f'<div class="ai-summary">{st.session_state.ai_insights}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 