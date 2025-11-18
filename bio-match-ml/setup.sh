#!/bin/bash

# BioMatch ML Infrastructure Setup Script

set -e

echo "🧬 BioMatch ML Infrastructure Setup"
echo "===================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing Python dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Dependencies installed"
echo ""

# Download spaCy models
echo "📥 Downloading spaCy models..."
python -m spacy download en_core_web_sm > /dev/null 2>&1
echo "✅ spaCy models downloaded"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/biobert models/pubmedbert models/scibert models/custom_finetuned
mkdir -p data logs
echo "✅ Directories created"
echo ""

# Check Docker
echo "🐳 Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker found: $(docker --version)"

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        echo "✅ Docker Compose found: $(docker-compose --version)"
    else
        echo "⚠️  Docker Compose not found. Install for full functionality."
    fi
else
    echo "⚠️  Docker not found. Install Docker for containerized deployment."
fi
echo ""

# Setup summary
echo "======================================"
echo "✨ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   uvicorn src.api.main:app --reload --port 8001"
echo ""
echo "3. Or use Docker Compose:"
echo "   docker-compose up -d"
echo ""
echo "4. Access the API:"
echo "   http://localhost:8001/docs"
echo ""
echo "5. Run tests:"
echo "   pytest tests/"
echo ""
echo "📖 See README.md for detailed documentation"
echo ""
