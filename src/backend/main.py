# main.py code - FIXED VERSION
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
    chat_history: Optional[List[Dict[str, Any]]] = None  # Add this field for context

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
        # logger.info("📚 Loading documents from 'documents' folder...")
        # await load_all_documents_from_folder()
        
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
    allow_origins=["*"],
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


# Main chat endpoint - ChatGPT-like interface with context support
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for ChatGPT-like interface with context support."""
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
        
        # Prepare chat history for context-aware processing
        chat_history = []
        
        # Use provided chat_history from request if available
        if request.chat_history and len(request.chat_history) > 0:
            chat_history = request.chat_history
        # Otherwise, use session messages (exclude current message)
        elif len(chat_sessions[session_id]) > 1:
            # Convert session messages to the format expected by RAG agents
            chat_history = [
                {
                    "id": f"msg_{i}",
                    "content": msg["content"],
                    "role": msg["role"],
                    "timestamp": msg["timestamp"].isoformat() if isinstance(msg["timestamp"], datetime) else msg["timestamp"]
                }
                for i, msg in enumerate(chat_sessions[session_id][:-1])  # Exclude current message
            ]
        
        # Create query request for RAG system
        from models import QueryRequest
        query_request = QueryRequest(
            query=request.message,
            include_sources=request.include_sources or True,
            max_results=5
        )
        
        # Use context-aware processing if chat history exists
        if chat_history and len(chat_history) > 0:
            logger.info(f"🔄 Using context-aware processing with {len(chat_history)} previous messages")
            try:
                # Check if process_query_with_context method exists
                if hasattr(rag_agents, 'process_query_with_context'):
                    rag_response = await rag_agents.process_query_with_context(
                        request=query_request,
                        chat_id=session_id,
                        chat_history=chat_history
                    )
                else:
                    logger.warning("⚠️ Context-aware processing not available, falling back to standard processing")
                    rag_response = await rag_agents.process_query(query_request)
            except Exception as context_error:
                logger.warning(f"⚠️ Context-aware processing failed: {context_error}, falling back to standard processing")
                rag_response = await rag_agents.process_query(query_request)
        else:
            # Use standard processing for first message or when no context
            logger.info("🔄 Using standard processing for new conversation")
            rag_response = await rag_agents.process_query(query_request)
        
        # FIXED: Access the correct attribute - use 'answer' instead of 'response'
        assistant_response_content = rag_response.answer  # ✅ Fixed: was rag_response.response
        
        # Add assistant response to session
        assistant_message = ChatMessage(
            role="assistant",
            content=assistant_response_content,
            timestamp=datetime.now()
        )
        chat_sessions[session_id].append(assistant_message.dict())
        
        # Limit session history (keep last 20 messages)
        if len(chat_sessions[session_id]) > 20:
            chat_sessions[session_id] = chat_sessions[session_id][-20:]
        
        processing_time = time.time() - start_time
        app_metrics["successful_responses"] += 1
        
        # FIXED: Prepare sources properly
        formatted_sources = []
        if request.include_sources and rag_response.sources:
            for doc in rag_response.sources:
                try:
                    # Handle different document formats
                    if hasattr(doc, 'dict'):
                        # If it's a Pydantic model
                        source = doc.dict()
                    elif hasattr(doc, '__dict__'):
                        # If it's a regular object with attributes
                        source = {
                            "content": getattr(doc, 'content', getattr(doc, 'page_content', str(doc))),
                            "metadata": getattr(doc, 'metadata', {})
                        }
                    elif isinstance(doc, dict):
                        # If it's already a dict
                        source = doc
                    else:
                        # Fallback for unknown types
                        source = {
                            "content": str(doc),
                            "metadata": {}
                        }
                    formatted_sources.append(source)
                except Exception as source_error:
                    logger.warning(f"Failed to format source: {source_error}")
                    continue
        
        response = ChatResponse(
            session_id=session_id,
            response=assistant_response_content,  # ✅ Fixed: using the correct variable
            sources=formatted_sources,  # ✅ Fixed: using properly formatted sources
            timestamp=datetime.now(),
            processing_time=processing_time
        )
        
        logger.info(f"✅ Chat processed in {processing_time:.2f}s (Session: {session_id[:8]})")
        return response
        
    except Exception as e:
        app_metrics["failed_responses"] += 1
        logger.error(f"❌ Chat processing failed: {e}")
        
        # Still add error message to session for continuity
        if 'session_id' in locals() and session_id in chat_sessions:
            error_message = ChatMessage(
                role="assistant", 
                content=f"Maaf, terjadi kesalahan: {str(e)}",
                timestamp=datetime.now()
            )
            chat_sessions[session_id].append(error_message.dict())
        
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


# Clear chat context endpoint
@app.delete("/api/chat/{session_id}")
async def clear_chat_context(session_id: str):
    """Clear chat context for a specific session."""
    try:
        # Clear from in-memory storage
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        # Clear from RAG agents context manager if method exists
        if hasattr(rag_agents, 'clear_chat_context'):
            rag_agents.clear_chat_context(session_id)
        
        logger.info(f"🗑️ Cleared context for session {session_id}")
        
        return {"message": f"Chat context cleared for session {session_id}"}
        
    except Exception as e:
        logger.error(f"❌ Failed to clear chat context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get context statistics endpoint
@app.get("/api/chat/stats")
async def get_chat_stats():
    """Get chat context statistics."""
    try:
        # Get stats from RAG agents if method exists
        context_stats = {}
        if hasattr(rag_agents, 'get_context_stats'):
            context_stats = rag_agents.get_context_stats()
        
        # Add session storage stats
        session_stats = {
            "active_sessions": len(chat_sessions),
            "total_messages": sum(len(messages) for messages in chat_sessions.values()),
            "average_session_length": (
                sum(len(messages) for messages in chat_sessions.values()) / len(chat_sessions)
                if len(chat_sessions) > 0 else 0
            )
        }
        
        return {
            "session_storage": session_stats,
            "context_manager": context_stats,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get chat stats: {e}")
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
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_document,
                file_content,
                file.filename,
                document_id
            )
        else:
            # Process immediately if no background tasks available
            await process_uploaded_document(file_content, file.filename, document_id)
        
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
        if background_tasks:
            background_tasks.add_task(reprocess_all_documents)
        else:
            await reprocess_all_documents()
        
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
        # await load_all_documents_from_folder()
        
        logger.info("✅ Document reprocessing complete")
        
    except Exception as e:
        logger.error(f"❌ Document reprocessing failed: {e}")


# Additional endpoint for bulk operations (optional)
@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active chat sessions."""
    try:
        sessions_info = []
        for session_id, messages in chat_sessions.items():
            if messages:
                sessions_info.append({
                    "session_id": session_id,
                    "message_count": len(messages),
                    "last_updated": messages[-1].get("timestamp", datetime.now()),
                    "created_at": messages[0].get("timestamp", datetime.now())
                })
        
        return {
            "active_sessions": len(sessions_info),
            "sessions": sessions_info[:10],  # Return first 10 for performance
            "total_messages": sum(len(messages) for messages in chat_sessions.values())
        }
    except Exception as e:
        logger.error(f"❌ Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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