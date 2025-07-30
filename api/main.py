from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Miami Real Estate Value Analyzer API",
    description="AI-powered property value analysis and market insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PropertyFeatures(BaseModel):
    tot_lvg_area: float
    lnd_sqfoot: float
    spec_feat_val: float = 0
    age: int
    ocean_dist: float
    water_dist: float
    cntr_dist: float
    structure_quality: int

class PredictionResponse(BaseModel):
    predicted_price: float
    price_per_sqft: float
    confidence_score: float
    market_comparison: Dict[str, Any]
    feature_importance: Dict[str, float]

class MarketInsights(BaseModel):
    total_properties: int
    avg_price: float
    median_price: float
    avg_sqft: float
    price_range: Dict[str, float]
    model_performance: Dict[str, float]

# Load model and data (in production, this would be done at startup)
def load_model():
    try:
        model = joblib.load('docs/price_prediction_model.pkl')
        with open('docs/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        with open('docs/market_insights.json', 'r') as f:
            insights = json.load(f)
        return model, metadata, insights
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

model, metadata, insights = load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Miami Real Estate Value Analyzer API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Predict property price",
            "/market-insights": "GET - Get market insights",
            "/model-info": "GET - Get model information",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(property: PropertyFeatures):
    """Predict property price based on features"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Prepare features in the same order as training
        feature_order = metadata['features']
        
        # Calculate derived features
        price_per_sqft = 0  # Will be calculated after prediction
        land_to_building_ratio = property.lnd_sqfoot / property.tot_lvg_area if property.tot_lvg_area > 0 else 0
        log_ocean_dist = np.log1p(property.ocean_dist)
        log_water_dist = np.log1p(property.water_dist)
        log_center_dist = np.log1p(property.cntr_dist)
        has_special_features = 1 if property.spec_feat_val > 0 else 0
        
        # Create feature vector
        features = {
            'TOT_LVG_AREA': property.tot_lvg_area,
            'LND_SQFOOT': property.lnd_sqfoot,
            'SPEC_FEAT_VAL': property.spec_feat_val,
            'age': property.age,
            'OCEAN_DIST': property.ocean_dist,
            'WATER_DIST': property.water_dist,
            'CNTR_DIST': property.cntr_dist,
            'structure_quality': property.structure_quality,
            'price_per_sqft': price_per_sqft,
            'land_to_building_ratio': land_to_building_ratio,
            'log_ocean_dist': log_ocean_dist,
            'log_water_dist': log_water_dist,
            'log_center_dist': log_center_dist,
            'has_special_features': has_special_features
        }
        
        # Prepare input features in correct order
        input_features = []
        for feature in feature_order:
            if feature in features:
                input_features.append(features[feature])
            else:
                input_features.append(0)
        
        # Make prediction
        predicted_price = model.predict([input_features])[0]
        
        # Calculate additional metrics
        price_per_sqft = predicted_price / property.tot_lvg_area if property.tot_lvg_area > 0 else 0
        
        # Calculate confidence score (simplified - in real app, use model uncertainty)
        confidence_score = min(0.95, max(0.7, 0.85 + (metadata['r2_score'] - 0.5) * 0.2))
        
        # Market comparison
        market_comparison = {
            "market_avg_price": insights['avg_price'] if insights else 0,
            "market_avg_price_per_sqft": insights['avg_price_per_sqft'] if insights else 0,
            "price_vs_market": (predicted_price / insights['avg_price'] - 1) * 100 if insights else 0,
            "sqft_vs_market": (property.tot_lvg_area / insights['avg_sqft'] - 1) * 100 if insights else 0
        }
        
        # Feature importance (simplified - in real app, use SHAP values)
        feature_importance = {
            "square_footage": 0.25,
            "location": 0.20,
            "age": 0.15,
            "quality": 0.15,
            "special_features": 0.10,
            "land_size": 0.10,
            "other": 0.05
        }
        
        return PredictionResponse(
            predicted_price=predicted_price,
            price_per_sqft=price_per_sqft,
            confidence_score=confidence_score,
            market_comparison=market_comparison,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/market-insights", response_model=MarketInsights)
async def get_market_insights():
    """Get current market insights"""
    
    if insights is None:
        raise HTTPException(status_code=500, detail="Market insights not available")
    
    return MarketInsights(
        total_properties=insights['total_properties'],
        avg_price=insights['avg_price'],
        median_price=insights['median_price'],
        avg_sqft=insights['avg_sqft'],
        price_range=insights['price_range'],
        model_performance=insights['model_performance']
    )

@app.get("/model-info")
async def get_model_info():
    """Get model information and performance metrics"""
    
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model information not available")
    
    return {
        "model_type": metadata['model_type'],
        "features": metadata['features'],
        "performance": {
            "mae": metadata['mae'],
            "r2_score": metadata['r2_score'],
            "error_percentage": metadata['error_percentage']
        },
        "training_info": {
            "training_samples": metadata['training_samples'],
            "test_samples": metadata['test_samples'],
            "avg_price": metadata['avg_price']
        },
        "last_updated": datetime.now().isoformat()
    }

@app.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance rankings"""
    
    try:
        feature_importance = pd.read_csv('docs/feature_importance.csv')
        return {
            "feature_importance": feature_importance.to_dict('records'),
            "top_features": feature_importance.head(10)['feature'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance not available: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 