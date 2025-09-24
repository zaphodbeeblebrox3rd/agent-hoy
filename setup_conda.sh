#!/bin/bash
# Conda setup script for agent-hoy

echo "Setting up agent-hoy with conda..."
echo "=================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found"

# Update conda
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Create environment
echo "Creating conda environment..."
conda env create -f environment.yaml

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent-hoy

# Test installation
echo "Testing installation..."
python test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use the application:"
echo "1. conda activate agent-hoy"
echo "2. python main.py"
echo ""
echo "To remove the environment:"
echo "conda env remove -n agent-hoy"
