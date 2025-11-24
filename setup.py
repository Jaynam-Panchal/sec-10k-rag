# setup.py
"""
Setup script for SEC 10-K RAG system.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sec-10k-rag",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready RAG system for SEC 10-K financial documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sec-10k-rag",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "10k-download=data_pipeline.download_10k:main",
            "10k-parse=data_pipeline.clean_parse:main",
            "10k-embed=data_pipeline.chunk_and_embed:main",
            "10k-index=data_pipeline.build_faiss_index:main",
            "10k-api=api.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)