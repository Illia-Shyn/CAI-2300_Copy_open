#!/bin/bash

# Miami Real Estate Value Analyzer - Setup Script
# For macOS/Linux

echo "🏠 Miami Real Estate Value Analyzer"
echo "Setting up the project..."
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p docs
mkdir -p notebooks
mkdir -p ui
mkdir -p api

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run analysis: python run_analysis.py"
echo "3. Start UI: cd ui && streamlit run app.py"
echo ""
echo "📝 Note: Make sure your data files are in the data/ directory" 