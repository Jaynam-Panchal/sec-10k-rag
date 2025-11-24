# ============================================
"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_10k_text():
    """Sample 10-K text for testing."""
    return """
    Item 1. Business
    
    Apple Inc. designs, manufactures, and markets smartphones, personal computers,
    tablets, wearables, and accessories worldwide. The Company also sells various
    related services.
    
    Item 1A. Risk Factors
    
    Our business is subject to numerous risks and uncertainties, including those
    highlighted in this section. These risks include global economic conditions,
    competition, and supply chain disruptions.
    """


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'ticker': 'AAPL',
        'filename': 'AAPL_0001193125_20240101.txt',
        'section': 'Risk Factors',
        'filing_date': '2024-01-01'
    }