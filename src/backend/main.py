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
    chat_history: Optional[List[Dict[str, Any]]] = None

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


async def load_new_documents_from_folder():
    """Smart document loading - only process new or modified files."""
    try:
        logger.info("🔄 Checking for new documents in folder...")
        
        # Get processing stats before
        stats_before = document_processor.get_processed_files_info()
        logger.info(f"📊 Before: {stats_before['processed_files']} files processed, {stats_before['total_chunks_created']} total chunks")
        
        # Load only new documents
        new_documents = await document_processor.load_new_documents_only()
        
        if new_documents:
            logger.info(f"📚 Adding {len(new_documents)} new document chunks to vector store...")
            
            # Add to vector store
            result = await vector_store_service.add_documents(new_documents)
            
            # Update metrics
            app_metrics["documents_processed"] += len(new_documents)
            
            # Get stats after
            stats_after = document_processor.get_processed_files_info()
            logger.info(f"📊 After: {stats_after['processed_files']} files processed, {stats_after['total_chunks_created']} total chunks")
            
            return {
                "new_documents_loaded": len(new_documents),
                "processing_result": result,
                "stats": stats_after
            }
        else:
            logger.info("✅ No new documents found - all files are up to date")
            return {
                "new_documents_loaded": 0,
                "message": "No new documents to process",
                "stats": stats_before
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to load documents from folder: {e}")
        raise


async def force_reload_all_documents():
    """Force reload all documents - ignores processed files database."""
    try:
        logger.info("🔄 Force reloading ALL documents...")
        
        # Force reprocess all documents
        all_documents = await document_processor.force_reprocess_all()
        
        if all_documents:
            logger.info(f"📚 Adding {len(all_documents)} document chunks to vector store...")
            
            # Add to vector store
            result = await vector_store_service.add_documents(all_documents)
            
            # Update metrics
            app_metrics["documents_processed"] = len(all_documents)
            
            # Get final stats
            stats = document_processor.get_processed_files_info()
            logger.info(f"📊 Final: {stats['processed_files']} files processed, {stats['total_chunks_created']} total chunks")
            
            return {
                "documents_reloaded": len(all_documents),
                "processing_result": result,
                "stats": stats
            }
        else:
            logger.warning("⚠️ No documents found to reload")
            return {
                "documents_reloaded": 0,
                "message": "No documents found",
                "stats": {}
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to reload documents: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with smart document loading."""
    # Startup
    logger.info("🚀 Starting Indonesian Legal RAG Assistant...")
    
    try:
        # Initialize vector store
        logger.info("📊 Initializing vector store...")
        has_existing_data = await vector_store_service.initialize_vector_store()
        
        # Smart document loading - only process new files
        logger.info("📚 Smart loading documents from 'documents' folder...")
        load_result = await load_new_documents_from_folder()
        
        # Get final collection info
        collection_info = await vector_store_service.get_collection_info()
        document_count = collection_info.get('document_count', 0)
        app_metrics["total_documents"] = document_count
        
        logger.info(f"✅ Application startup complete!")
        logger.info(f"📊 Total documents in vector store: {document_count}")
        logger.info(f"📊 New documents processed: {load_result['new_documents_loaded']}")
        
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
                    # FIXED: Pass chat_history properly to initialize AgentState
                    rag_response = await rag_agents.process_query_with_context(
                        request=query_request,
                        chat_id=session_id,
                        chat_history=chat_history  # This should be used to set frontend_chat_history
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
        
        # Access the correct attribute
        assistant_response_content = rag_response.answer
        
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
        
        # Prepare sources properly
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
            response=assistant_response_content,
            sources=formatted_sources,
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
        processing_stats = document_processor.get_processing_stats()
        
        return {
            "total_documents": collection_info.get("document_count", 0),
            "status": collection_info.get("status", "unknown"),
            "embedding_model": "mxbai-embed-large",
            "last_updated": datetime.now(),
            "processing_stats": processing_stats
        }
    except Exception as e:
        logger.error(f"❌ Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Check for new documents endpoint
@app.post("/api/documents/check-new")
async def check_new_documents(background_tasks: BackgroundTasks):
    """Check for and process new documents in the documents folder."""
    try:
        logger.info("🔄 Manual check for new documents...")
        
        if background_tasks:
            background_tasks.add_task(load_new_documents_from_folder)
            return {"message": "Checking for new documents started", "status": "processing"}
        else:
            result = await load_new_documents_from_folder()
            return {
                "message": "New documents check completed",
                "status": "completed",
                "result": result
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to check new documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reprocess all documents endpoint
@app.post("/api/documents/reprocess")
async def reprocess_documents(background_tasks: BackgroundTasks):
    """Force reprocess all documents in the documents folder."""
    try:
        logger.info("🔄 Starting document reprocessing...")
        
        if background_tasks:
            background_tasks.add_task(force_reload_all_documents)
            return {"message": "Document reprocessing started", "status": "processing"}
        else:
            result = await force_reload_all_documents()
            return {
                "message": "Document reprocessing completed",
                "status": "completed",
                "result": result
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to start reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Clear processed files database
@app.post("/api/documents/clear-processed-db")
async def clear_processed_files_database():
    """Clear the processed files database to force reprocessing on next check."""
    try:
        document_processor.clear_processed_files_db()
        
        return {
            "message": "Processed files database cleared",
            "status": "success",
            "note": "Next document check will reprocess all files"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear processed files database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get processed files info
@app.get("/api/documents/processed-info")
async def get_processed_files_info():
    """Get information about processed files."""
    try:
        stats = document_processor.get_processed_files_info()
        return {
            "processed_files_info": stats,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get processed files info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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