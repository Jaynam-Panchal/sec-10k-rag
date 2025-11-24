"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from api.main import app

client = TestClient(app)


@pytest.fixture
def mock_index_loader():
    """Mock FAISS index loader."""
    mock_loader = Mock()
    mock_loader.index = Mock()
    mock_loader.index.ntotal = 1000
    mock_loader.index.d = 768
    mock_loader.ids = [f"chunk_{i}" for i in range(10)]
    mock_loader.metadatas = [
        {
            'ticker': 'AAPL',
            'section': 'Risk Factors',
            'file_name': 'AAPL_test.txt',
            'chunk_index': i
        }
        for i in range(10)
    ]
    return mock_loader


@pytest.fixture
def mock_embedder():
    """Mock embedding generator."""
    mock_emb = Mock()
    mock_emb.embed_texts = Mock(return_value=np.random.rand(1, 768))
    return mock_emb


def test_root_endpoint():
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@patch('api.main.index_loader')
def test_stats_endpoint(mock_loader):
    """Test statistics endpoint."""
    mock_loader.index = Mock()
    mock_loader.index.ntotal = 1000
    mock_loader.index.d = 768
    mock_loader.metadatas = [
        {'ticker': 'AAPL'},
        {'ticker': 'MSFT'}
    ]
    
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_vectors"] == 1000
    assert data["dimension"] == 768


@patch('api.main.index_loader')
def test_companies_endpoint(mock_loader):
    """Test companies list endpoint."""
    mock_loader.metadatas = [
        {'ticker': 'AAPL'},
        {'ticker': 'MSFT'},
        {'ticker': 'AAPL'}  # Duplicate
    ]
    
    response = client.get("/companies")
    assert response.status_code == 200
    companies = response.json()
    assert 'AAPL' in companies
    assert 'MSFT' in companies
    assert len(companies) == 2  # No duplicates


@patch('api.main.embedder')
@patch('api.main.index_loader')
def test_search_endpoint(mock_loader, mock_embedder, tmp_path):
    """Test search endpoint."""
    # Setup mocks
    mock_embedder.embed_texts = Mock(return_value=np.random.rand(1, 768))
    
    mock_loader.index = Mock()
    mock_loader.ids = ['chunk_0']
    mock_loader.metadatas = [{
        'ticker': 'AAPL',
        'section': 'Risk Factors',
        'file_name': 'AAPL_test.txt',
        'chunk_index': 0
    }]
    
    mock_loader.search = Mock(return_value=(
        np.array([0.95]),
        np.array([0]),
        mock_loader.metadatas
    ))
    
    # Create mock chunk file
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunk_file = chunk_dir / "chunk_0.json"
    chunk_file.write_text('{"id": "chunk_0", "text": "Test risk factor", "meta": {}}')
    
    with patch('api.main.config.CHUNKS_DIR', chunk_dir):
        response = client.post(
            "/search",
            json={
                "query": "risk factors",
                "top_k": 5
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert data["query"] == "risk factors"


def test_search_endpoint_validation():
    """Test search request validation."""
    # Empty query
    response = client.post(
        "/search",
        json={"query": "", "top_k": 5}
    )
    assert response.status_code == 422  # Validation error
    
    # Invalid top_k
    response = client.post(
        "/search",
        json={"query": "test", "top_k": 100}
    )
    assert response.status_code == 422


@patch('api.main.index_loader')
def test_sections_endpoint(mock_loader):
    """Test sections endpoint for a ticker."""
    mock_loader.metadatas = [
        {'ticker': 'AAPL', 'section': 'Risk Factors'},
        {'ticker': 'AAPL', 'section': 'MD&A'},
        {'ticker': 'MSFT', 'section': 'Risk Factors'}
    ]
    
    response = client.get("/sections/AAPL")
    assert response.status_code == 200
    sections = response.json()
    assert 'Risk Factors' in sections
    assert 'MD&A' in sections
    assert len(sections) == 2


def test_sections_endpoint_not_found():
    """Test sections endpoint for non-existent ticker."""
    with patch('api.main.index_loader') as mock_loader:
        mock_loader.metadatas = []
        
        response = client.get("/sections/INVALID")
        assert response.status_code == 404