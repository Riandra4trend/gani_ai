"""
Pydantic models for Indonesian Law RAG AI Assistant.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"


class QueryIntent(str, Enum):
    """Query intent classification."""
    LEGAL_QUESTION = "legal_question"
    DOCUMENT_SEARCH = "document_search"
    REGULATION_LOOKUP = "regulation_lookup"
    GENERAL_INFO = "general_info"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    document_id: str
    title: str
    source_url: Optional[str] = None
    document_type: DocumentType
    regulation_number: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    upload_date: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = None
    language: str = "id"  # Indonesian
    
    class Config:
        use_enum_values = True


class ChunkMetadata(BaseModel):
    """Chunk metadata for vector storage."""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_pos: int
    end_pos: int
    chunk_size: int
    overlap_size: int = 0


class RetrievedDocument(BaseModel):
    """Retrieved document with relevance score."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_metadata: Optional[ChunkMetadata] = None


class HydeQuery(BaseModel):
    """HYDE (Hypothetical Document Embeddings) query model."""
    original_query: str
    hypothetical_documents: List[str]
    enhanced_query: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """User query request model."""
    query: str = Field(..., min_length=10, max_length=1000)
    query_type: Optional[QueryIntent] = None
    context: Optional[Dict[str, Any]] = None
    use_hyde: bool = True
    max_results: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    """RAG query response model."""
    query: str
    answer: str
    sources: List[RetrievedDocument]
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time: float
    hyde_info: Optional[HydeQuery] = None
    query_intent: Optional[QueryIntent] = None
    
    class Config:
        use_enum_values = True


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    title: str = Field(..., min_length=1, max_length=200)
    document_type: DocumentType
    regulation_number: Optional[str] = None
    year: Optional[int] = Field(None, ge=1945, le=2030)
    category: Optional[str] = None
    source_url: Optional[str] = None
    content: Optional[str] = None  # For direct text upload
    
    @validator('year')
    def validate_year(cls, v):
        if v and (v < 1945 or v > 2030):
            raise ValueError('Year must be between 1945 and 2030')
        return v


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str
    title: str
    status: str
    chunks_created: int
    processing_time: float
    message: str


class AgentState(BaseModel):
    """LangGraph agent state model."""
    query: str
    retrieved_docs: List[RetrievedDocument] = []
    initial_answer: Optional[str] = None
    reviewed_answer: Optional[str] = None
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    iteration_count: int = 0
    feedback: Optional[str] = None
    sources: List[RetrievedDocument] = []
    hyde_query: Optional[HydeQuery] = None
    
    class Config:
        arbitrary_types_allowed = True


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    model_status: str
    
    class Config:
        use_enum_values = True


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class MetricsResponse(BaseModel):
    """System metrics response."""
    total_documents: int
    total_chunks: int
    total_queries: int
    average_response_time: float
    uptime: float
    memory_usage: Optional[Dict[str, Any]] = None