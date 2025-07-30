#!/usr/bin/env python3
"""
Miami Real Estate Value Analyzer - Data Analysis Script
Runs the data exploration and model training automatically
"""

import subprocess
import sys
import os
from pathlib import Path

def run_notebook():
    """Run the data exploration notebook to generate model and insights"""
    
    print("🚀 Starting Miami Real Estate Analysis...")
    print("=" * 50)
    
    # Check if data files exist
    data_files = [
        "data/miami-housing.csv",
        "data/MEDLISPRI12086.csv",
        "data/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all data files are in the data/ directory.")
        return False
    
    print("✅ Data files found")
    
    # Check if requirements are installed
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        import plotly
        print("✅ Required packages installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Run the notebook using papermill (if available) or jupyter
    notebook_path = "notebooks/01_data_exploration.ipynb"
    
    try:
        # Try to use papermill for automated execution
        import papermill as pm
        
        print("📊 Running data analysis notebook...")
        pm.execute_notebook(
            notebook_path,
            "docs/analysis_output.ipynb",
            parameters={}
        )
        print("✅ Notebook executed successfully")
        
    except ImportError:
        print("📊 Papermill not available, using jupyter nbconvert...")
        
        # Use jupyter nbconvert to execute notebook
        result = subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--output", "docs/analysis_output.ipynb",
            notebook_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully")
        else:
            print("❌ Error executing notebook:")
            print(result.stderr)
            return False
    
    # Check if model files were generated
    expected_files = [
        "docs/price_prediction_model.pkl",
        "docs/model_metadata.json", 
        "docs/market_insights.json",
        "docs/feature_importance.csv"
    ]
    
    print("\n📋 Generated files:")
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")
    
    print("\n🎯 Analysis Complete!")
    print("=" * 50)
    print("Next steps:")
    print("1. Start the UI: cd ui && streamlit run app.py")
    print("2. Or start the API: cd api && uvicorn main:app --reload")
    print("3. View results in docs/ directory")
    
    return True

def main():
    """Main function"""
    print("🏠 Miami Real Estate Value Analyzer")
    print("Data Analysis & Model Training")
    print("=" * 50)
    
    success = run_notebook()
    
    if success:
        print("\n🎉 Setup complete! Ready to run the application.")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 