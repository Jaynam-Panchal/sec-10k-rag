# Makefile for SEC 10-K RAG System

.PHONY: help install setup clean test lint format run-api docker-build docker-run pipeline

# Default target
help:
	@echo "SEC 10-K RAG System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make setup          - Complete setup (venv + install + dirs)"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download       - Download 10-K filings from SEC"
	@echo "  make parse          - Parse and clean documents"
	@echo "  make embed          - Generate embeddings"
	@echo "  make index          - Build FAISS index"
	@echo "  make pipeline       - Run complete pipeline"
	@echo ""
	@echo "Development:"
	@echo "  make run-api        - Run API server (development)"
	@echo "  make run-prod       - Run API server (production)"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code with black & isort"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-stop    - Stop Docker container"
	@echo "  make docker-clean   - Remove Docker containers and images"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean generated files and cache"
	@echo "  make clean-data     - Remove all data files"
	@echo "  make logs           - Tail API logs"

# Setup and Installation
install:
	pip install -r requirements.txt

setup:
	python -m venv venv
ifeq ($(OS),Windows_NT)
	@echo "Virtual environment created. Activate with: source venv/Scripts/activate"
else
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
endif
	@echo "Then run: make install"


# Data Pipeline
download:
	python -m data_pipeline.download_10k

parse:
	python -m data_pipeline.clean_parse

embed:
	python -m data_pipeline.chunk_and_embed

index:
	python -m data_pipeline.build_faiss_index

pipeline: download parse embed index
	@echo "✓ Complete pipeline finished!"

# Development
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	gunicorn api.main:app \
		-w 4 \
		-k uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--timeout 120 \
		--access-logfile logs/access.log \
		--error-logfile logs/error.log

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --exclude=venv,__pycache__
	mypy . --ignore-missing-imports

format:
	black .
	isort .

# Docker
docker-build:
	docker build -t sec-10k-rag:latest .

docker-run:
	docker run -d \
		--name sec-10k-rag \
		-p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		sec-10k-rag:latest

docker-stop:
	docker stop sec-10k-rag
	docker rm sec-10k-rag

docker-clean:
	docker stop sec-10k-rag || true
	docker rm sec-10k-rag || true
	docker rmi sec-10k-rag:latest || true

# Docker Compose
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	@echo "✓ Cleaned cache files"

clean-data:
	rm -rf data/raw_10k/* data/clean_txt/* data/chunks/* data/index/*
	@echo "✓ Cleaned data directories"

logs:
	tail -f logs/api.log

# Monitoring
health:
	@curl -s http://localhost:8000/health | jq .

stats:
	@curl -s http://localhost:8000/stats | jq .

companies:
	@curl -s http://localhost:8000/companies | jq .

# Quick search test
search-test:
	@curl -s -X POST http://localhost:8000/search \
		-H "Content-Type: application/json" \
		-d '{"query": "What are the main risk factors?", "top_k": 3}' | jq .

# Documentation
docs:
	@echo "Opening API documentation..."
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs

# Installation for development
dev-install:
	pip install -e ".[dev]"
	pre-commit install

# Database migrations (if using Alembic in future)
migrate:
	alembic upgrade head

# Backup data
backup:
	mkdir -p backups
	tar -czf backups/index_backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/index/
	@echo "✓ Index backed up to backups/"