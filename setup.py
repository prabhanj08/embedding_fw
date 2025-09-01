"""
Setup configuration for the AI Embedding Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="ai-embedding-framework",
    version="1.0.0",
    author="AI Embedding Framework Team",
    author_email="contact@embedding-framework.ai",
    description="A comprehensive, modular text embedding system supporting multiple providers and document types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ai-embedding-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (minimal for basic functionality)
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "typing-extensions>=4.0.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "chardet>=5.0.0",
    ],
    extras_require={
        # Embedding providers
        "openai": ["openai>=1.0.0", "tiktoken>=0.5.0"],
        "anthropic": ["anthropic>=0.8.0"],
        "cohere": ["cohere>=4.0.0"],
        "huggingface": [
            "sentence-transformers>=2.2.0",
            "transformers>=4.21.0",
            "torch>=1.12.0",
        ],
        
        # Document parsing
        "pdf": ["PyPDF2>=3.0.0", "pdfplumber>=0.9.0"],
        "powerpoint": ["python-pptx>=0.6.21"],
        "ocr": ["pytesseract>=0.3.10", "Pillow>=9.0.0"],
        
        # Vector stores
        "chroma": ["chromadb>=0.4.0"],
        "pinecone": ["pinecone-client>=2.2.0"],
        "opensearch": ["opensearch-py>=2.0.0"],
        
        # Text processing
        "nlp": ["nltk>=3.8", "spacy>=3.6.0"],
        "language": ["langdetect>=1.0.9"],
        
        # Development
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        
        # All optional dependencies
        "all": [
            "openai>=1.0.0", "tiktoken>=0.5.0",
            "anthropic>=0.8.0",
            "cohere>=4.0.0",
            "sentence-transformers>=2.2.0", "transformers>=4.21.0", "torch>=1.12.0",
            "PyPDF2>=3.0.0", "pdfplumber>=0.9.0",
            "python-pptx>=0.6.21",
            "pytesseract>=0.3.10", "Pillow>=9.0.0",
            "chromadb>=0.4.0",
            "pinecone-client>=2.2.0",
            "opensearch-py>=2.0.0",
            "nltk>=3.8", "spacy>=3.6.0",
            "langdetect>=1.0.9",
        ]
    },
    entry_points={
        "console_scripts": [
            "embedding-framework=embedding_fw.cli:main",
        ],
    },
    keywords=[
        "embeddings", "ai", "nlp", "machine-learning", "text-processing",
        "document-processing", "vector-search", "semantic-search",
        "openai", "huggingface", "cohere", "anthropic"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/ai-embedding-framework/issues",
        "Source": "https://github.com/your-org/ai-embedding-framework",
        "Documentation": "https://ai-embedding-framework.readthedocs.io/",
    },
)