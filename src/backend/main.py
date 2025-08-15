"""
Main FastAPI application for Indonesian Legal RAG Assistant.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from config import settings
from models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, DocumentUploadResponse,
    HealthCheckResponse, ErrorResponse, MetricsResponse, DocumentMetadata, DocumentType
)
from vector_store import vector_store_service
from document_processor import DocumentProcessor
from rag_agents import rag_agents
from hyde_service import hyde_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
document_processor = DocumentProcessor()
security = HTTPBearer(auto_error=False)

# Application metrics
app_metrics = {
    "start_time": datetime.now(),
    "total_queries": 0,
    "total_documents": 0,
    "total_response_time": 0.0,
    "successful_queries": 0,
    "failed_queries": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Indonesian Legal RAG Assistant...")
    
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        has_data = await vector_store_service.initialize_vector_store()
        
        if not has_data:
            logger.info("Loading initial legal documents...")
            await load_initial_documents()
        else:
            logger.info("Vector store already contains data")
        
        # Get collection info
        collection_info = await vector_store_service.get_collection_info()
        logger.info(f"Vector store ready: {collection_info.get('document_count', 0)} documents")
        
        app_metrics["total_documents"] = collection_info.get('document_count', 0)
        
        logger.info("✅ Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Indonesian Legal RAG Assistant...")


# Create FastAPI app
app = FastAPI(
    title="Indonesian Legal RAG Assistant",
    description="AI Assistant with RAG for Indonesian Compliance Law using LangChain, LangGraph, and Gemini API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Dependency for optional authentication
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Optional authentication dependency."""
    # Implement your authentication logic here if needed
    return credentials


async def load_initial_documents():
    """Load initial legal documents into vector store."""
    try:
        logger.info("Processing predefined legal documents...")
        documents = await document_processor.process_predefined_documents()
        
        if documents:
            result = await vector_store_service.add_documents(documents)
            logger.info(f"✅ Loaded {result['documents_added']} document chunks")
        else:
            logger.warning("⚠️ No documents were processed")
            
    except Exception as e:
        logger.error(f"❌ Failed to load initial documents: {e}")
        # Don't raise error - app can still function for document upload


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc)
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        collection_info = await vector_store_service.get_collection_info()
        db_status = "healthy" if collection_info.get("status") == "initialized" else "unhealthy"
        
        # Check model (simple test)
        model_status = "healthy"  # Could add actual model test
        
        return HealthCheckResponse(
            status="healthy" if db_status == "healthy" else "degraded",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status=db_status,
            model_status=model_status
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status="error",
            model_status="error"
        )


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    try:
        collection_info = await vector_store_service.get_collection_info()
        
        uptime = (datetime.now() - app_metrics["start_time"]).total_seconds()
        avg_response_time = (
            app_metrics["total_response_time"] / max(app_metrics["total_queries"], 1)
        )
        
        return MetricsResponse(
            total_documents=collection_info.get("document_count", 0),
            total_chunks=collection_info.get("document_count", 0),  # Simplified
            total_queries=app_metrics["total_queries"],
            average_response_time=avg_response_time,
            uptime=uptime
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_legal_documents(
    request: QueryRequest,
    current_user = Depends(get_current_user)
):
    """Query the legal document knowledge base."""
    start_time = time.time()
    app_metrics["total_queries"] += 1
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Process query through multi-agent RAG system
        response = await rag_agents.process_query(request)
        
        # Update metrics
        processing_time = time.time() - start_time
        app_metrics["total_response_time"] += processing_time
        app_metrics["successful_queries"] += 1
        
        logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        app_metrics["failed_queries"] += 1
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document upload endpoint
@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    title: str = Form(...),
    document_type: str = Form(...),
    regulation_number: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    category: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_current_user)
):
    """Upload a legal document to the knowledge base."""
    start_time = time.time()
    
    try:
        # Validate input
        if not file and not content:
            raise HTTPException(
                status_code=400, 
                detail="Either file upload or text content is required"
            )
        
        if not DocumentType.__members__.get(document_type.upper()):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document type. Must be one of: {list(DocumentType.__members__.keys())}"
            )
        
        doc_type = DocumentType(document_type.lower())
        
        # Extract content
        if file:
            file_content = await file.read()
            if doc_type == DocumentType.PDF:
                extracted_content = await document_processor._extract_pdf_content(file_content)
            else:
                extracted_content = file_content.decode('utf-8')
        else:
            extracted_content = content
        
        if not extracted_content or len(extracted_content.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Document content is too short (minimum 100 characters)"
            )
        
        # Create metadata
        metadata = document_processor.create_document_metadata(
            title=title,
            content=extracted_content,
            doc_type=doc_type,
            source_url=source_url,
            regulation_number=regulation_number,
            year=year,
            category=category
        )
        
        # Process document in background
        background_tasks.add_task(
            process_uploaded_document,
            extracted_content,
            metadata
        )
        
        processing_time = time.time() - start_time
        
        response = DocumentUploadResponse(
            document_id=metadata.document_id,
            title=title,
            status="processing",
            chunks_created=0,  # Will be updated in background
            processing_time=processing_time,
            message="Document uploaded successfully and is being processed"
        )
        
        logger.info(f"✅ Document upload initiated: {title}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_uploaded_document(content: str, metadata: DocumentMetadata):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Processing document: {metadata.title}")
        
        # Chunk document
        chunks = await document_processor.chunk_document(content, metadata)
        
        # Add to vector store
        result = await vector_store_service.add_documents(chunks)
        
        # Update metrics
        app_metrics["total_documents"] += 1
        
        logger.info(f"✅ Document processed: {metadata.title}, {len(chunks)} chunks created")
        
    except Exception as e:
        logger.error(f"❌ Background document processing failed: {e}")


# Direct text upload endpoint
@app.post("/upload-text", response_model=DocumentUploadResponse)
async def upload_text_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Upload a text document directly via JSON."""
    start_time = time.time()
    
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        if len(request.content.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Content is too short (minimum 100 characters)"
            )
        
        # Create metadata
        metadata = document_processor.create_document_metadata(
            title=request.title,
            content=request.content,
            doc_type=request.document_type,
            source_url=request.source_url,
            regulation_number=request.regulation_number,
            year=request.year,
            category=request.category
        )
        
        # Process document in background
        background_tasks.add_task(
            process_uploaded_document,
            request.content,
            metadata
        )
        
        processing_time = time.time() - start_time
        
        response = DocumentUploadResponse(
            document_id=metadata.document_id,
            title=request.title,
            status="processing",
            chunks_created=0,
            processing_time=processing_time,
            message="Text document uploaded successfully and is being processed"
        )
        
        logger.info(f"✅ Text document upload initiated: {request.title}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search documents endpoint
@app.get("/search-documents")
async def search_documents(
    query: str,
    limit: int = 10,
    current_user = Depends(get_current_user)
):
    """Search documents by content similarity."""
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters long"
            )
        
        results = await vector_store_service.similarity_search(
            query=query.strip(),
            k=min(limit, 50)  # Cap at 50 results
        )
        
        return {
            "query": query,
            "results": [result.dict() for result in results],
            "total_results": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collection info endpoint
@app.get("/collection-info")
async def get_collection_info(current_user = Depends(get_current_user)):
    """Get information about the document collection."""
    try:
        info = await vector_store_service.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info"
    )