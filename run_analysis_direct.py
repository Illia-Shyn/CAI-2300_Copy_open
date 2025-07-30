#!/usr/bin/env python3
"""
Miami Real Estate Value Analyzer - Direct Analysis Script
Runs the data exploration and model training directly without jupyter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_historical_price_data():
    """Load and analyze historical price trends"""
    try:
        # Load historical median prices
        historical_data = pd.read_csv('data/MEDLISPRI12086.csv')
        historical_data['observation_date'] = pd.to_datetime(historical_data['observation_date'])
        
        # Calculate price appreciation
        start_price = historical_data.iloc[0]['MEDLISPRI12086']  # 2016 price
        end_price = historical_data.iloc[-1]['MEDLISPRI12086']    # 2024 price
        appreciation_factor = end_price / start_price
        
        print(f"📈 Market Appreciation Analysis:")
        print(f"   Start (2016): ${start_price:,.0f}")
        print(f"   End (2024): ${end_price:,.0f}")
        print(f"   Appreciation Factor: {appreciation_factor:.2f}x ({appreciation_factor*100-100:.1f}% increase)")
        
        return appreciation_factor, historical_data
    except Exception as e:
        print(f"⚠️ Warning: Could not load historical data: {e}")
        return 1.44, None  # Default 44% increase

def prepare_features(df):
    """Prepare features with price adjustment"""
    print("🔧 Preparing features with market adjustment...")
    
    # Load market appreciation factor
    appreciation_factor, _ = load_historical_price_data()
    
    # Create derived features
    df['price_per_sqft'] = df['SALE_PRC'] / df['TOT_LVG_AREA']
    df['land_to_building_ratio'] = df['LND_SQFOOT'] / df['TOT_LVG_AREA']
    df['log_ocean_dist'] = np.log1p(df['OCEAN_DIST'])
    df['log_water_dist'] = np.log1p(df['WATER_DIST'])
    df['log_center_dist'] = np.log1p(df['CNTR_DIST'])
    df['has_special_features'] = (df['SPEC_FEAT_VAL'] > 0).astype(int)
    
    # ADJUST PRICES FOR CURRENT MARKET
    df['SALE_PRC_ADJUSTED'] = df['SALE_PRC'] * appreciation_factor
    df['price_per_sqft_adjusted'] = df['SALE_PRC_ADJUSTED'] / df['TOT_LVG_AREA']
    
    print(f"💰 Price Adjustment Applied:")
    print(f"   Original avg price: ${df['SALE_PRC'].mean():,.0f}")
    print(f"   Adjusted avg price: ${df['SALE_PRC_ADJUSTED'].mean():,.0f}")
    print(f"   Adjustment factor: {appreciation_factor:.2f}x")
    
    return df

def create_market_analysis(df, appreciation_factor):
    """Create market analysis with adjusted prices"""
    print("📈 Creating market analysis...")
    
    # Create interactive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Distribution (Adjusted)', 'Price vs Living Area', 
                       'Price per Sq Ft by Quality', 'Distance Impact on Price'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # Price distribution
    fig.add_trace(
        go.Histogram(x=df['SALE_PRC_ADJUSTED'], name='Adjusted Prices', nbinsx=50),
        row=1, col=1
    )
    
    # Price vs Living Area
    fig.add_trace(
        go.Scatter(x=df['TOT_LVG_AREA'], y=df['SALE_PRC_ADJUSTED'], 
                  mode='markers', name='Properties', opacity=0.6),
        row=1, col=2
    )
    
    # Price per sq ft by quality
    fig.add_trace(
        go.Box(y=df['price_per_sqft_adjusted'], x=df['structure_quality'], 
               name='Price/Sq Ft by Quality'),
        row=2, col=1
    )
    
    # Distance impact
    fig.add_trace(
        go.Scatter(x=df['OCEAN_DIST'], y=df['SALE_PRC_ADJUSTED'], 
                  mode='markers', name='Ocean Distance Impact', opacity=0.6),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'Miami Real Estate Market Analysis (Prices Adjusted for {appreciation_factor:.1f}x Market Appreciation)',
        height=800,
        showlegend=False
    )
    
    # Save the visualization
    fig.write_html('docs/market_analysis_adjusted.html')
    print("✅ Market analysis saved to docs/market_analysis_adjusted.html")
    
    return fig

def train_model(df):
    """Train the model with adjusted prices"""
    print("🤖 Training ML model with adjusted prices...")
    
    # Features for the model
    feature_columns = [
        'TOT_LVG_AREA', 'LND_SQFOOT', 'SPEC_FEAT_VAL', 'age',
        'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'structure_quality',
        'price_per_sqft_adjusted', 'land_to_building_ratio',
        'log_ocean_dist', 'log_water_dist', 'log_center_dist',
        'has_special_features'
    ]
    
    X = df[feature_columns]
    y = df['SALE_PRC_ADJUSTED']  # Use adjusted prices
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    avg_price = y_test.mean()
    error_percentage = (mae / avg_price) * 100
    
    print(f"🎯 Model Performance (Adjusted Prices):")
    print(f"   MAE: ${mae:,.0f}")
    print(f"   R² Score: {r2:.3f}")
    print(f"   Average Price: ${avg_price:,.0f}")
    print(f"   Error Percentage: {error_percentage:.1f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return model, feature_columns, mae, r2, error_percentage, feature_importance, len(X_train), len(X_test)

def save_model_and_metadata(model, feature_columns, mae, r2, error_percentage, feature_importance, train_samples, test_samples, df, appreciation_factor):
    """Save model and metadata"""
    print("💾 Saving model and metadata...")
    
    # Save model
    joblib.dump(model, 'docs/price_prediction_model_adjusted.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'features': feature_columns,
        'mae': mae,
        'r2_score': r2,
        'training_samples': train_samples,
        'test_samples': test_samples,
        'avg_price': df['SALE_PRC_ADJUSTED'].mean(),
        'error_percentage': error_percentage,
        'appreciation_factor': appreciation_factor,
        'original_avg_price': df['SALE_PRC'].mean(),
        'adjusted_avg_price': df['SALE_PRC_ADJUSTED'].mean()
    }
    
    with open('docs/model_metadata_adjusted.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance
    feature_importance.to_csv('docs/feature_importance_adjusted.csv', index=False)
    
    # Save market insights
    insights = {
        'total_properties': len(df),
        'avg_price': df['SALE_PRC_ADJUSTED'].mean(),
        'median_price': df['SALE_PRC_ADJUSTED'].median(),
        'avg_sqft': df['TOT_LVG_AREA'].mean(),
        'avg_price_per_sqft': df['price_per_sqft_adjusted'].mean(),
        'price_range': {
            'min': df['SALE_PRC_ADJUSTED'].min(),
            'max': df['SALE_PRC_ADJUSTED'].max()
        },
        'model_performance': {
            'mae': mae,
            'r2_score': r2,
            'error_percentage': error_percentage
        },
        'market_adjustment': {
            'appreciation_factor': appreciation_factor,
            'original_avg_price': df['SALE_PRC'].mean(),
            'adjusted_avg_price': df['SALE_PRC_ADJUSTED'].mean()
        },
        'top_features': feature_importance.head(5)['feature'].tolist()
    }
    
    with open('docs/market_insights_adjusted.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("✅ Model and metadata saved to docs/")

def main():
    print("🏠 Miami Real Estate Value Analyzer")
    print("Market-Adjusted Analysis & Model Training")
    print("=" * 50)
    
    # Create docs directory
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    print("📊 Loading data...")
    # 1. Load Miami housing data
    df = pd.read_csv('data/miami-housing.csv')
    print(f"   Loaded {len(df):,} property records")
    
    # 2. Load historical price trends
    appreciation_factor, historical_data = load_historical_price_data()
    
    print("\n🔧 Preparing features with market adjustment...")
    # 3. Feature engineering with price adjustment
    df = prepare_features(df)
    
    print("\n📈 Creating visualizations...")
    # 4. Create market analysis visualization
    create_market_analysis(df, appreciation_factor)
    
    print("\n🤖 Training ML model...")
    # 5. Train model with adjusted prices
    model, feature_columns, mae, r2, error_percentage, feature_importance, train_samples, test_samples = train_model(df)
    
    print("\n💾 Saving model and metadata...")
    # 6. Save model and metadata
    save_model_and_metadata(model, feature_columns, mae, r2, error_percentage, feature_importance, train_samples, test_samples, df, appreciation_factor)
    
    print("\n✅ Analysis Complete!")
    print(f"📁 Generated files:")
    print(f"   - docs/price_prediction_model_adjusted.pkl")
    print(f"   - docs/model_metadata_adjusted.json")
    print(f"   - docs/feature_importance_adjusted.csv")
    print(f"   - docs/market_insights_adjusted.json")
    print(f"   - docs/market_analysis_adjusted.html")
    print(f"\n🎯 Next steps:")
    print(f"   1. Update UI to use adjusted model")
    print(f"   2. Test predictions with current market prices")
    print(f"   3. Present results with market-adjusted accuracy")
    
    return True

if __name__ == "__main__":
    main() 