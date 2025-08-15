"""
Configuration settings for Indonesian Legal RAG Assistant with enhanced processing.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with enhanced document processing configuration."""
    
    # Application settings
    app_name: str = "Indonesian Legal RAG Assistant"
    app_version: str = "2.0.0"
    app_host: str = Field(default="0.0.0.0", description="Application host")
    app_port: int = Field(default=8000, description="Application port")
    app_debug: bool = Field(default=False, description="Debug mode")
    
    # API Keys
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    
    # Vector database settings
    vector_db_path: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    collection_name: str = Field(default="indonesian_legal_docs", description="ChromaDB collection name")
    
    # Document processing settings
    documents_folder: str = Field(default="documents", description="Folder containing PDF documents")
    chunk_size: int = Field(default=1000, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks")
    max_file_size_mb: int = Field(default=50, description="Maximum file size for uploads in MB")
    
    # OCR settings
    tesseract_cmd: Optional[str] = Field(default=None, description="Path to tesseract executable")
    ocr_languages: str = Field(default="ind+eng", description="OCR languages (Indonesian + English)")
    ocr_config: str = Field(default="--oem 3 --psm 6", description="Tesseract OCR configuration")
    image_preprocessing: bool = Field(default=True, description="Enable image preprocessing for OCR")
    
    # Text processing settings
    enable_stemming: bool = Field(default=True, description="Enable text stemming")
    enable_stopword_removal: bool = Field(default=True, description="Enable stopword removal")
    min_token_length: int = Field(default=3, description="Minimum token length")
    enable_text_normalization: bool = Field(default=True, description="Enable text normalization")
    
    # Indonesian NLP settings
    use_indonesian_stemmer: bool = Field(default=True, description="Use Sastrawi Indonesian stemmer")
    use_combined_stopwords: bool = Field(default=True, description="Use combined Indonesian + English stopwords")
    
    # RAG settings
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")
    max_context_length: int = Field(default=4000, description="Maximum context length for RAG")
    
    # Multi-agent settings
    enable_hyde: bool = Field(default=True, description="Enable HyDE (Hypothetical Document Embeddings)")
    enable_query_expansion: bool = Field(default=True, description="Enable query expansion")
    max_query_expansions: int = Field(default=3, description="Maximum number of query expansions")
    
    # Performance settings
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    batch_size: int = Field(default=10, description="Batch size for processing")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Monitoring and logging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_retention_days: int = Field(default=30, description="Metrics retention period")
    
    # Security settings
    enable_auth: bool = Field(default=False, description="Enable authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    allowed_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Document categories for classification
    document_categories: List[str] = Field(
        default=[
            "undang_undang",
            "peraturan_pemerintah", 
            "keputusan_presiden",
            "peraturan_presiden",
            "peraturan_menteri",
            "keputusan_menteri",
            "peraturan_daerah",
            "dokumen_hukum"
        ],
        description="Available document categories"
    )
    
    # File processing settings
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".txt", ".docx"],
        description="Allowed file extensions for upload"
    )
    
    # Background task settings
    task_timeout_seconds: int = Field(default=3600, description="Background task timeout")
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent background tasks")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Set tesseract path if not provided
        if not self.tesseract_cmd:
            import shutil
            tesseract_path = shutil.which("tesseract")
            if tesseract_path:
                self.tesseract_cmd = tesseract_path
        
        # Create necessary directories
        os.makedirs(self.documents_folder, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Validate OCR setup
        if self.tesseract_cmd and not os.path.exists(self.tesseract_cmd):
            import warnings
            warnings.warn(f"Tesseract not found at {self.tesseract_cmd}. OCR functionality may not work.")


# Global settings instance
settings = Settings()

# Environment-specific configurations
if settings.app_debug:
    # Development settings
    settings.log_level = "DEBUG"
    settings.enable_caching = False
    settings.chunk_size = 500  # Smaller chunks for faster processing during development
    
# Production optimizations
if not settings.app_debug and os.getenv("ENVIRONMENT") == "production":
    settings.max_workers = min(8, os.cpu_count() or 4)
    settings.batch_size = 20
    settings.cache_ttl_seconds = 7200  # 2 hours