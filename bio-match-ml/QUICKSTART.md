# Quick Start Guide

Get the BioMatch ML infrastructure running in 5 minutes.

## 🚀 Fastest Start (Docker)

```bash
# Clone and navigate
cd bio-match-ml

# Start all services
docker-compose up -d

# Wait ~30 seconds for services to initialize
# Then test the API
curl http://localhost:8001/api/v1/health
```

**Access:**
- API Documentation: http://localhost:8001/docs
- Elasticsearch: http://localhost:9200
- Redis: localhost:6379

## 🐍 Python Development Setup

```bash
# Run automated setup
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the server
uvicorn src.api.main:app --reload --port 8001
```

## 🧪 Test the API

### 1. Health Check
```bash
curl http://localhost:8001/api/v1/health
```

### 2. Search for Faculty
```bash
curl -X POST "http://localhost:8001/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CRISPR gene editing",
    "mode": "faculty",
    "limit": 5
  }'
```

### 3. Generate Embeddings
```bash
curl -X POST "http://localhost:8001/api/v1/embeddings/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["p53 tumor suppressor protein"],
    "model": "pubmedbert"
  }'
```

### 4. Extract Entities
```bash
curl -X POST "http://localhost:8001/api/v1/entities/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We used CRISPR to knock out BRCA1 in HeLa cells"
  }'
```

## 📊 Load Sample Data

```python
# Python script to load sample faculty data
import requests
import numpy as np

API_BASE = "http://localhost:8001/api/v1"

# Generate sample embedding
response = requests.post(
    f"{API_BASE}/embeddings/generate",
    json={
        "texts": ["Research on neural circuits and optogenetics"],
        "model": "pubmedbert"
    }
)

embedding = response.json()["embeddings"][0]
print(f"Generated {len(embedding)}-dimensional embedding")

# In a real scenario, you would:
# 1. Get faculty data from data collection modules
# 2. Generate embeddings for their research profiles
# 3. Insert into vector store
# 4. Index in Elasticsearch
```

## 🐳 Docker Commands

```bash
# Start core services only
docker-compose up -d ml-api elasticsearch redis

# Start with GPU support
docker-compose --profile gpu up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f ml-api

# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## 🔍 Common Operations

### Search Faculty by Research Interest
```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/search/semantic",
    json={
        "query": "machine learning in genomics",
        "mode": "faculty",
        "filters": {
            "institution": ["Stanford", "MIT"],
            "has_active_funding": True
        },
        "limit": 10
    }
)

results = response.json()["results"]
for r in results:
    print(f"{r['name']} at {r['institution']}: {r['match_score']:.3f}")
```

### Calculate Student-Faculty Matches
```python
response = requests.post(
    "http://localhost:8001/api/v1/match/calculate",
    json={
        "student_profile": {
            "research_interests": "Studying cancer metabolism using CRISPR screens",
            "techniques": ["CRISPR", "RNA-seq", "metabolomics"],
            "career_goals": "Academic research career"
        },
        "top_k": 10,
        "explain_top_n": 3
    }
)

matches = response.json()["matches"]
for m in matches:
    print(f"{m['faculty_name']}: {m['overall_score']:.3f}")
    print(f"  Strengths: {', '.join(m['strengths'])}")
```

## 🧹 Troubleshooting

### Port Already in Use
```bash
# Change ports in docker-compose.yml
ports:
  - "8002:8001"  # Use 8002 instead of 8001
```

### Out of Memory
```bash
# Reduce batch size in configs/model_configs.yaml
batch_size: 16  # Default is 32
```

### Models Not Downloading
```bash
# Manually download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')"
```

### Elasticsearch Won't Start
```bash
# Increase vm.max_map_count (Linux)
sudo sysctl -w vm.max_map_count=262144

# Or disable Elasticsearch and use FAISS only
docker-compose up -d ml-api redis
```

## 📖 Next Steps

1. **Read the Full Documentation**: See [README.md](README.md)
2. **Explore the API**: Visit http://localhost:8001/docs
3. **Run Tests**: `pytest tests/`
4. **Configure for Production**: Edit `configs/*.yaml`
5. **Integrate with Frontend**: Use the REST API endpoints

## 🎯 What You Can Do Now

✅ Search for faculty using natural language
✅ Calculate student-faculty compatibility
✅ Extract biological entities from text
✅ Generate embeddings for any text
✅ Perform semantic similarity search
✅ Classify research topics
✅ Identify research methods and techniques

## 📞 Need Help?

- **API Docs**: http://localhost:8001/docs
- **Full README**: [README.md](README.md)
- **Issues**: Create a GitHub issue
- **Examples**: See `notebooks/` directory

Ready to build intelligent research matching! 🧬🔬
