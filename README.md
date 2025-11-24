# SEC 10-K Financial Document RAG System

A production ready Retrieval-Augmented Generation (RAG) system for semantic search over SEC 10-K financial filings. Built with FAISS vector search, sentence transformers, and FastAPI.

## Features

- **Automated Data Pipeline**: Download, parse, and process SEC 10-K filings
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Production-Ready API**: FastAPI REST API with comprehensive error handling
- **Scalable Architecture**: Handles multiple companies and years of filings
- **Section-Aware**: Extracts and indexes specific 10-K sections (Risk Factors, MD&A, etc.)
- **Filtering**: Search by company ticker or document section
- **Monitoring**: Structured logging and health checks


## Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM (for embedding model)
- 5GB+ disk space (for documents and index)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sec-10k-rag.git
cd sec-10k-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline

Run the complete data pipeline:

```bash
# 1. Download 10-K filings from SEC
python -m data_pipeline.download_10k

# 2. Parse and clean documents
python -m data_pipeline.clean_parse

# 3. Chunk and generate embeddings
python -m data_pipeline.chunk_and_embed

# 4. Build FAISS index
python -m data_pipeline.build_faiss_index
```

### Start API Server

```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Access the API at `http://localhost:8000`

Interactive docs at `http://localhost:8000/docs`

## API Usage

### Search Documents

```bash
# Basic search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main risk factors?",
    "top_k": 5
  }'

# Filter by company
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth trends",
    "top_k": 5,
    "ticker_filter": "AAPL"
  }'

# Filter by section
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "market competition",
    "top_k": 3,
    "section_filter": "Risk Factors"
  }'
```

### Other Endpoints

```bash
# Health check
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/stats

# List companies
curl http://localhost:8000/companies

# Get sections for a company
curl http://localhost:8000/sections/AAPL
```

##  Project Structure

## 🔧 Configuration

Edit `config/config.py` to customize:

- Chunk size and overlap
- Embedding model
- FAISS index type
- API settings
- Company list

Or use environment variables:

```bash
export SEC_USER_AGENT="MyCompany/1.0 (contact@example.com)"
export API_PORT=8080
export EMBEDDING_MODEL="all-mpnet-base-v2"
```

##  Docker Deployment

```bash
# Build image
docker build -t sec-10k-rag:latest .

# Run container
docker run -p 8000:8000 sec-10k-rag:latest

# Or use docker-compose
docker-compose up -d
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_api.py -v
```

## Performance

- **Index Build Time**: ~5-10 minutes (14 companies, 3 years each)
- **Search Latency**: <100ms for top-5 results
- **Index Size**: ~2GB (42 documents, ~150K chunks)
- **Memory Usage**: ~4GB (model + index loaded)

##  Development

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type checking
mypy .
```

### Adding New Companies

Edit `config/config.py`:

```python
TICKER_TO_CIK = {
    'AAPL': '320193',
    'YOURNEW': '1234567',  # Add CIK here
}
```

Then re-run the data pipeline.

##  Technical Details

### Embedding Model
- **Model**: `all-mpnet-base-v2` (768-dim)
- **Performance**: 768D vectors, cosine similarity
- **Quality**: Better than MiniLM while maintaining speed

### FAISS Index
- **Type**: IndexFlatIP (exact search, inner product)
- **Alternatives**: IndexIVFFlat (approximate, faster)
- **Normalization**: L2-normalized for cosine similarity

### Chunking Strategy
- **Method**: Semantic chunking by sentences
- **Size**: 500 words per chunk
- **Overlap**: 100 words between chunks
- **Benefit**: Preserves context across boundaries

##  Use Cases

1. **Financial Research**: Query specific risk factors or business strategies
2. **Competitive Analysis**: Compare companies' MD&A sections
3. **Due Diligence**: Quick access to material information
4. **Compliance**: Search for specific disclosure language
5. **Education**: Learn about corporate governance and reporting

##  Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure code passes linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details

##  Acknowledgments

- SEC EDGAR for providing public financial data
- Sentence Transformers team
- FAISS team at Meta AI
- FastAPI framework

##  Contact

- **Author**: Jaynam Panchal
- **Email**: jpanchal1@pride.hofstra.edu

If you find this project useful, please consider starring it!