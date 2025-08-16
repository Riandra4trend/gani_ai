"""
Vector store service using ChromaDB with Ollama embeddings.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings   # ✅ fixed import
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import settings
from models import RetrievedDocument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_store")


class VectorStoreService:
    """Manages vector storage and retrieval for legal documents."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize Ollama embeddings."""
        try:
            self.embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434",
                model=settings.embedding_model   # ✅ now uses config
            )
            logger.info(f"✓ Initialized Ollama embeddings with model: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            raise
    
    async def initialize_vector_store(self, force_reload: bool = False) -> bool:
        """Initialize ChromaDB vector store."""
        try:
            os.makedirs(settings.vector_store_path, exist_ok=True)
            persist_directory = settings.vector_store_path
            
            logger.info("Creating new vector store..." if force_reload else "Loading vector store...")
            self.vector_store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            
            # If directory empty or reload requested
            if force_reload or not os.listdir(persist_directory):
                logger.warning("Vector store is empty, documents must be added.")
                return False
            else:
                await self._initialize_retriever()
                return True
        
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _initialize_retriever(self):
        """Initialize retriever with reranking."""
        try:
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.retrieval_k,
                    "score_threshold": settings.similarity_threshold
                }
            )
            
            if not self.reranker:
                cross_encoder = HuggingFaceCrossEncoder(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.reranker = CrossEncoderReranker(model=cross_encoder)
            
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )
            logger.info("✓ Initialized retriever with reranking")
        
        except Exception as e:
            logger.warning(f"Failed to initialize reranker, using base retriever: {e}")
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.retrieval_k,
                    "score_threshold": settings.similarity_threshold
                }
            )
    
    async def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to vector store."""
        try:
            if not self.vector_store:
                await self.initialize_vector_store()
            
            start_time = datetime.now()
            
            doc_ids = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.vector_store.add_documents(documents)
            )
            
            await asyncio.get_event_loop().run_in_executor(None, self.vector_store.persist)
            await self._initialize_retriever()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Added {len(documents)} documents to vector store")
            return {
                "documents_added": len(documents),
                "document_ids": doc_ids,
                "processing_time": processing_time,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Perform similarity search."""
        try:
            if not self.retriever:
                raise ValueError("Vector store not initialized")
            
            k = k or settings.retrieval_k

            if filter_metadata:
                base_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k, "filter": filter_metadata}
                )
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: base_retriever.get_relevant_documents(query)
                )
            else:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.retriever.get_relevant_documents(query)
                )
            
            retrieved_docs = [
                RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=doc.metadata.get("relevance_score", 0.8)
                ) for doc in docs
            ]
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    async def get_document_count(self) -> int:
        """Get total number of documents in vector store."""
        try:
            if not self.vector_store:
                return 0
            return self.vector_store._collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection (for startup checks)."""
        try:
            if not self.vector_store:
                return {"status": "not_initialized"}
            
            collection = self.vector_store._collection
            count = collection.count()
            sample = collection.peek(limit=5)
            sample_keys = []
            if sample and sample.get("metadatas"):
                first_meta = sample["metadatas"][0] or {}
                sample_keys = list(first_meta.keys())

            return {
                "status": "initialized",
                "collection_name": settings.collection_name,
                "document_count": count,
                "embedding_model": settings.embedding_model,
                "sample_metadata_keys": sample_keys,
                "vector_store_path": settings.vector_store_path
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}


# Global vector store instance
vector_store_service = VectorStoreService()
