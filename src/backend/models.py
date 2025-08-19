from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field


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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    chunk_metadata: Optional[ChunkMetadata] = None
    
    class Config:
        arbitrary_types_allowed = True


class HydeQuery(BaseModel):
    """HYDE (Hypothetical Document Embeddings) query model."""
    original_query: str
    hypothetical_documents: List[str] = Field(default_factory=list)
    enhanced_query: str
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


class QueryRequest(BaseModel):
    """User query request model."""
    query: str = Field(..., min_length=10, max_length=1000)
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    include_context: bool = True
    max_results: int = Field(default=10, ge=1, le=50)
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    """RAG query response model."""
    query: str
    answer: str
    sources: List[RetrievedDocument] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time: float
    hyde_info: Optional[HydeQuery] = None
    query_intent: Optional[QueryIntent] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    """LangGraph agent state model."""
    query: str
    chat_id: str = ""
    retrieved_docs: List[RetrievedDocument] = Field(default_factory=list)
    initial_answer: Optional[str] = None
    reviewed_answer: Optional[str] = None
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    iteration_count: int = 0
    feedback: Optional[str] = None
    sources: List[RetrievedDocument] = Field(default_factory=list)
    hyde_query: Optional[HydeQuery] = None
    
    # Context and analysis fields
    conversation_context: Optional[Dict[str, Any]] = None
    query_analysis: Optional[Dict[str, Any]] = None
    current_answer: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
    should_continue: bool = False
    update_context: bool = False
    generated_answer: Optional[str] = None

    # Frontend chat history and context
    frontend_chat_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    context_source: Optional[str] = None  # "frontend" or "internal"
    
    # Additional processing fields
    processing_start: Optional[datetime] = Field(default_factory=datetime.now)
    processing_steps: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)
    set_context_sources: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def set_context_source(self, source: str) -> None:
        """Set the context source for this state."""
        self.context_source = source
        if source not in self.set_context_sources:
            self.set_context_sources.append(source)

class ChatMessage(BaseModel):
    """Chat message model for API responses."""
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str
    chat_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True
    
    class Config:
        arbitrary_types_allowed = True


class ChatRequest(BaseModel):
    """Chat request model for conversational API."""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    include_sources: bool = True
    stream_response: bool = False
    
    @validator('message')
    def validate_message(cls, v):
        if not v or v.isspace():
            raise ValueError('Message cannot be empty')
        return v.strip()


class ChatResponse(BaseModel):
    """Chat response model for conversational API."""
    message: str
    session_id: str
    sources: List[RetrievedDocument] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Add response attribute for backward compatibility
    @property
    def response(self) -> str:
        """Backward compatibility property."""
        return self.message
    
    class Config:
        arbitrary_types_allowed = True


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


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    database_status: str
    model_status: str
    context_manager_stats: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
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
    context_stats: Optional[Dict[str, Any]] = None


class ContextSummary(BaseModel):
    """Conversation context summary model."""
    chat_id: str
    summary: str
    topics: List[str] = Field(default_factory=list)
    user_intent: str = ""
    last_updated: datetime = Field(default_factory=datetime.now)
    message_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


class WorkflowStatus(BaseModel):
    """Workflow execution status model."""
    step_name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    output_size: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class ProcessingMetrics(BaseModel):
    """Processing metrics model."""
    total_steps: int
    completed_steps: int
    failed_steps: int
    total_processing_time: float
    average_step_time: float
    workflow_status: List[WorkflowStatus] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True