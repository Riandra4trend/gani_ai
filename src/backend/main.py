import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from config import settings
from vector_store import vector_store_service
from document_processor import DocumentProcessor
from rag_agents import rag_agents

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
document_processor = DocumentProcessor()

# In-memory storage for chat sessions (replace with Redis in production)
chat_sessions: Dict[str, List[Dict]] = {}
app_metrics = {
    "start_time": datetime.now(),
    "total_chats": 0,
    "total_documents": 0,
    "successful_responses": 0,
    "failed_responses": 0,
    "documents_processed": 0
}

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: Optional[bool] = True

class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: Optional[List[Dict]] = []
    timestamp: datetime
    processing_time: float

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_updated: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_status: str
    total_documents: int

class MetricsResponse(BaseModel):
    total_chats: int
    total_documents: int
    successful_responses: int
    failed_responses: int
    uptime_seconds: float
    documents_processed: int

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    chunks_created: Optional[int] = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with automatic document loading."""
    # Startup
    logger.info("🚀 Starting Indonesian Legal RAG Assistant...")
    
    try:
        # Initialize vector store
        logger.info("📊 Initializing vector store...")
        has_data = await vector_store_service.initialize_vector_store()
        
        # Always load documents from folder on startup
        logger.info("📚 Loading documents from 'documents' folder...")
        await load_all_documents_from_folder()
        
        # Get final collection info
        collection_info = await vector_store_service.get_collection_info()
        document_count = collection_info.get('document_count', 0)
        app_metrics["total_documents"] = document_count
        
        logger.info(f"✅ Application startup complete! {document_count} documents loaded")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Indonesian Legal RAG Assistant...")


async def load_all_documents_from_folder():
    """Load all documents from the documents folder."""
    try:
        documents_folder = "documents"
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
            logger.info("📁 Created documents folder")
            return

        # Get all PDF files
        pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("⚠️ No PDF files found in documents folder")
            return

        logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
        
        # Process documents
        documents = await document_processor.load_documents_from_folder(documents_folder)
        
        if documents:
            logger.info(f"🔄 Adding {len(documents)} document chunks to vector store...")
            result = await vector_store_service.add_documents(documents)
            
            app_metrics["documents_processed"] = result.get('documents_processed', len(pdf_files))
            
            logger.info(f"✅ Successfully loaded {result['documents_added']} document chunks from {len(pdf_files)} files")
        else:
            logger.warning("⚠️ No documents were processed successfully")
            
    except Exception as e:
        logger.error(f"❌ Failed to load documents: {e}")


# Create FastAPI app
app = FastAPI(
    title="Indonesian Legal RAG Assistant",
    description="ChatGPT-like AI Assistant for Indonesian Legal Documents",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    app_metrics["failed_responses"] += 1
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)}
    )


# Main chat endpoint - ChatGPT-like interface
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for ChatGPT-like interface."""
    start_time = time.time()
    app_metrics["total_chats"] += 1
    
    try:
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if doesn't exist
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Add user message to session
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now()
        )
        chat_sessions[session_id].append(user_message.dict())
        
        logger.info(f"💬 Processing chat: {request.message[:100]}... (Session: {session_id[:8]})")
        
        # Create query request for RAG system
        from models import QueryRequest
        query_request = QueryRequest(
            query=request.message,
            include_sources=request.include_sources or True,
            max_results=5
        )
        
        # Process through RAG system
        rag_response = await rag_agents.process_query(query_request)
        
        # Add assistant response to session
        assistant_message = ChatMessage(
            role="assistant",
            content=rag_response.response,
            timestamp=datetime.now()
        )
        chat_sessions[session_id].append(assistant_message.dict())
        
        # Limit session history (keep last 20 messages)
        if len(chat_sessions[session_id]) > 20:
            chat_sessions[session_id] = chat_sessions[session_id][-20:]
        
        processing_time = time.time() - start_time
        app_metrics["successful_responses"] += 1
        
        response = ChatResponse(
            session_id=session_id,
            response=rag_response.response,
            sources=rag_response.sources if request.include_sources else [],
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        logger.info(f"✅ Chat processed in {processing_time:.2f}s (Session: {session_id[:8]})")
        return response
        
    except Exception as e:
        app_metrics["failed_responses"] += 1
        logger.error(f"❌ Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get chat history
@app.get("/api/chat/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        if session_id not in chat_sessions:
            return ChatSession(
                session_id=session_id,
                messages=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        messages = [ChatMessage(**msg) for msg in chat_sessions[session_id]]
        
        return ChatSession(
            session_id=session_id,
            messages=messages,
            created_at=messages[0].timestamp if messages else datetime.now(),
            last_updated=messages[-1].timestamp if messages else datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document upload endpoint
@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a document to the knowledge base."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        logger.info(f"📄 Starting document upload: {file.filename} (ID: {document_id[:8]})")
        
        # Read file content
        file_content = await file.read()
        
        # Process document in background
        background_tasks.add_task(
            process_uploaded_document,
            file_content,
            file.filename,
            document_id
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processing",
            message=f"Document '{file.filename}' uploaded and is being processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_uploaded_document(file_content: bytes, filename: str, document_id: str):
    """Background task to process uploaded document."""
    try:
        logger.info(f"🔄 Processing uploaded document: {filename} (ID: {document_id[:8]})")
        
        # Process through document processor
        documents = await document_processor.process_uploaded_file(
            file_content=file_content,
            filename=filename,
            title=filename.replace('.pdf', ''),
            doc_type='regulation'  # default type
        )
        
        if documents:
            # Add to vector store
            result = await vector_store_service.add_documents(documents)
            
            # Update metrics
            app_metrics["documents_processed"] += 1
            app_metrics["total_documents"] += 1
            
            logger.info(f"✅ Document processed: {filename} (ID: {document_id[:8]}), {len(documents)} chunks created")
        else:
            logger.warning(f"⚠️ No chunks created for document: {filename} (ID: {document_id[:8]})")
        
    except Exception as e:
        logger.error(f"❌ Document processing failed for {filename} (ID: {document_id[:8]}): {e}")


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        collection_info = await vector_store_service.get_collection_info()
        db_status = "healthy" if collection_info.get("status") == "initialized" else "unhealthy"
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" else "degraded",
            timestamp=datetime.now(),
            version="2.1.0",
            database_status=db_status,
            total_documents=collection_info.get("document_count", 0)
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="2.1.0",
            database_status="error",
            total_documents=0
        )


# Metrics endpoint
@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    try:
        uptime = (datetime.now() - app_metrics["start_time"]).total_seconds()
        
        return MetricsResponse(
            total_chats=app_metrics["total_chats"],
            total_documents=app_metrics["total_documents"],
            successful_responses=app_metrics["successful_responses"],
            failed_responses=app_metrics["failed_responses"],
            uptime_seconds=uptime,
            documents_processed=app_metrics["documents_processed"]
        )
    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get document collection info
@app.get("/api/documents/info")
async def get_documents_info():
    """Get information about the document collection."""
    try:
        collection_info = await vector_store_service.get_collection_info()
        return {
            "total_documents": collection_info.get("document_count", 0),
            "status": collection_info.get("status", "unknown"),
            "embedding_model": "nomic-embed-text",
            "last_updated": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reprocess all documents
@app.post("/api/documents/reprocess")
async def reprocess_documents(background_tasks: BackgroundTasks):
    """Reprocess all documents in the documents folder."""
    try:
        logger.info("🔄 Starting document reprocessing...")
        
        # Clear existing data
        await vector_store_service.clear_collection()
        
        # Reprocess in background
        background_tasks.add_task(reprocess_all_documents)
        
        return {"message": "Document reprocessing started", "status": "processing"}
        
    except Exception as e:
        logger.error(f"❌ Failed to start reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def reprocess_all_documents():
    """Background task to reprocess all documents."""
    try:
        # Reset metrics
        app_metrics["documents_processed"] = 0
        app_metrics["total_documents"] = 0
        
        # Load documents
        await load_all_documents_from_folder()
        
        logger.info("✅ Document reprocessing complete")
        
    except Exception as e:
        logger.error(f"❌ Document reprocessing failed: {e}")


if __name__ == "__main__":
    # Ensure documents folder exists
    os.makedirs("documents", exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info"
    )