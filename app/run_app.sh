#!/bin/bash

# MPCIM Dashboard Startup Script
# Author: Deni Sulaeman

echo "=========================================="
echo "MPCIM Dashboard - Starting Application"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip3 install -r requirements.txt
else
    echo "✅ Streamlit is already installed"
fi

echo ""
echo "🚀 Starting MPCIM Dashboard..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run the Streamlit app
streamlit run Home.py
