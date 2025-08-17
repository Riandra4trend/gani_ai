"""
Vector store service using ChromaDB with Ollama embeddings.
Aligned with main.py and document_processor.py requirements.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings  # ✅ Fixed import to match document_processor
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
                model=settings.embedding_model   # Uses config setting
            )
            logger.info(f"✅ Initialized Ollama embeddings with model: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama embeddings: {e}")
            raise
    
    async def initialize_vector_store(self, force_reload: bool = False) -> bool:
        """
        Initialize ChromaDB vector store.
        
        Returns:
            bool: True if existing data found, False if empty/new
        """
        try:
            os.makedirs(settings.vector_store_path, exist_ok=True)
            persist_directory = settings.vector_store_path
            
            logger.info("🔄 Creating new vector store..." if force_reload else "🔄 Loading vector store...")
            
            # Initialize Chroma with consistent parameters
            self.vector_store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Check if vector store has existing data
            try:
                document_count = await self.get_document_count()
                has_existing_data = document_count > 0
                
                if has_existing_data and not force_reload:
                    logger.info(f"✅ Found existing vector store with {document_count} documents")
                    await self._initialize_retriever()
                    return True
                else:
                    if force_reload:
                        logger.info("🔄 Force reload requested - vector store ready for new documents")
                    else:
                        logger.info("📊 Vector store is empty - ready for documents")
                    return False
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not check existing data: {e}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    async def _initialize_retriever(self):
        """Initialize retriever with optional reranking."""
        try:
            # Base retriever
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.retrieval_k,
                    "score_threshold": getattr(settings, 'similarity_threshold', 0.7)
                }
            )
            
            # Try to initialize reranker (optional)
            try:
                if not self.reranker:
                    cross_encoder = HuggingFaceCrossEncoder(
                        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                    self.reranker = CrossEncoderReranker(model=cross_encoder)
                
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=base_retriever
                )
                logger.info("✅ Initialized retriever with reranking")
                
            except Exception as rerank_error:
                logger.warning(f"⚠️ Failed to initialize reranker, using base retriever: {rerank_error}")
                self.retriever = base_retriever
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize retriever: {e}")
            # Fallback to basic retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": getattr(settings, 'retrieval_k', 5)}
            )
    
    async def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            Dict with processing results
        """
        try:
            if not self.vector_store:
                await self.initialize_vector_store()
            
            if not documents:
                logger.warning("⚠️ No documents provided to add")
                return {
                    "documents_added": 0,
                    "document_ids": [],
                    "processing_time": 0,
                    "status": "no_documents"
                }
            
            start_time = datetime.now()
            logger.info(f"🔄 Adding {len(documents)} documents to vector store...")
            
            # Add documents in a separate thread to avoid blocking
            doc_ids = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.vector_store.add_documents(documents)
            )
            
            
            # Reinitialize retriever with new data
            await self._initialize_retriever()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Successfully added {len(documents)} documents to vector store in {processing_time:.2f}s")
            
            return {
                "documents_added": len(documents),
                "document_ids": doc_ids,
                "processing_time": processing_time,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            raise
    
    async def similarity_search_optimized(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Optimized similarity search specifically for ChromaDB.
        This method avoids problematic parameters and uses direct ChromaDB queries.
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            k = k or getattr(settings, 'retrieval_k', 5)
            logger.info(f"🔍 Starting optimized similarity search for query: {query[:50]}... (k={k})")

            # Check if we have any documents first
            doc_count = await self.get_document_count()
            if doc_count == 0:
                logger.warning("⚠️ Vector store is empty - no documents to search")
                return []

            docs = []
            
            try:
                # Direct ChromaDB similarity search without problematic parameters
                logger.info("Using direct ChromaDB similarity search...")
                
                # Prepare search arguments
                search_args = {"k": k}
                if filter_metadata:
                    search_args["filter"] = filter_metadata
                
                # Execute search in thread pool to avoid blocking
                docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.vector_store.similarity_search(query, **search_args)
                )
                
                logger.info(f"✅ Direct ChromaDB search returned {len(docs)} documents")
                
            except Exception as search_error:
                logger.error(f"❌ Direct ChromaDB search failed: {search_error}")
                
                # Try alternative: similarity search with scores
                try:
                    logger.info("Trying similarity search with scores...")
                    docs_with_scores = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.vector_store.similarity_search_with_score(query, k=k)
                    )
                    
                    # Extract documents from (doc, score) tuples
                    docs = [doc for doc, score in docs_with_scores]
                    logger.info(f"✅ Search with scores returned {len(docs)} documents")
                    
                except Exception as score_error:
                    logger.error(f"❌ Search with scores also failed: {score_error}")
                    docs = []

            # Convert to RetrievedDocument objects with improved error handling
            retrieved_docs: List[RetrievedDocument] = []
            
            for i, doc in enumerate(docs[:k]):
                try:
                    # Extract content safely
                    content = ""
                    if hasattr(doc, 'page_content') and doc.page_content:
                        content = doc.page_content
                    elif hasattr(doc, 'content') and doc.content:
                        content = doc.content
                    elif isinstance(doc, dict):
                        content = doc.get('page_content', doc.get('content', ''))
                    else:
                        content = str(doc)
                    
                    # Skip empty documents
                    if not content or len(content.strip()) < 10:
                        logger.warning(f"⚠️ Skipping document {i} - insufficient content")
                        continue

                    # Extract metadata safely
                    metadata = {}
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                        metadata = doc.metadata
                    elif isinstance(doc, dict) and 'metadata' in doc:
                        metadata = doc['metadata'] if isinstance(doc['metadata'], dict) else {}

                    # Calculate relevance score based on position
                    relevance_score = max(0.1, 1.0 - (i * 0.15))  # Decreasing score by rank
                    
                    # Override with actual score if available
                    if 'score' in metadata:
                        try:
                            relevance_score = float(metadata['score'])
                        except (ValueError, TypeError):
                            pass

                    retrieved_doc = RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        relevance_score=relevance_score
                    )
                    retrieved_docs.append(retrieved_doc)
                    
                except Exception as doc_error:
                    logger.warning(f"⚠️ Failed to process document {i}: {doc_error}")
                    continue

            logger.info(f"✅ Successfully retrieved {len(retrieved_docs)} valid documents")
            
            # Apply simple relevance filtering if no metadata filter was used
            if not filter_metadata and len(retrieved_docs) > 1:
                # Sort by relevance score descending
                retrieved_docs.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return retrieved_docs

        except Exception as e:
            logger.error(f"❌ Optimized similarity search failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []


    async def get_document_count(self) -> int:
        """Get total number of documents in vector store."""
        try:
            if not self.vector_store:
                return 0
            
            # Use ChromaDB collection count
            collection = self.vector_store._collection
            count = collection.count()
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to get document count: {e}")
            return 0

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the vector store collection.
        Used by main.py for startup checks and status reporting.
        
        Returns:
            Dict with collection information
        """
        try:
            if not self.vector_store:
                return {
                    "status": "not_initialized",
                    "document_count": 0,
                    "collection_name": getattr(settings, 'collection_name', 'legal_docs'),
                    "embedding_model": getattr(settings, 'embedding_model', 'mxbai-embed-large'),
                    "vector_store_path": getattr(settings, 'vector_store_path', './chroma_db')
                }
            
            # Get collection details
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get sample metadata to understand document structure
            sample_metadata_keys = []
            try:
                sample = collection.peek(limit=5)
                if sample and sample.get("metadatas") and len(sample["metadatas"]) > 0:
                    first_meta = sample["metadatas"][0] or {}
                    sample_metadata_keys = list(first_meta.keys())
            except Exception as sample_error:
                logger.warning(f"⚠️ Could not get sample metadata: {sample_error}")

            return {
                "status": "initialized",
                "collection_name": getattr(settings, 'collection_name', 'legal_docs'),
                "document_count": count,
                "embedding_model": getattr(settings, 'embedding_model', 'mxbai-embed-large'),
                "sample_metadata_keys": sample_metadata_keys,
                "vector_store_path": getattr(settings, 'vector_store_path', './chroma_db'),
                "retriever_initialized": self.retriever is not None,
                "reranker_available": self.reranker is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "document_count": 0
            }

    async def clear_collection(self):
        """
        Clear all documents from the vector store collection.
        Used for force reload operations.
        """
        try:
            if not self.vector_store:
                logger.warning("⚠️ Vector store not initialized")
                return
            
            # Delete the collection
            collection = self.vector_store._collection
            collection.delete()
            
            # Reinitialize empty collection
            self.vector_store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.vector_store_path
            )
            
            # Clear retriever
            self.retriever = None
            
            logger.info("🗑️ Cleared vector store collection")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.
        
        Returns:
            Dict with health status
        """
        try:
            if not self.vector_store:
                return {"status": "unhealthy", "reason": "not_initialized"}
            
            # Try to get document count
            count = await self.get_document_count()
            
            # Try a simple similarity search
            try:
                test_docs = await self.similarity_search("test query", k=1)
                search_working = True
            except:
                search_working = False
            
            status = "healthy" if search_working or count > 0 else "degraded"
            
            return {
                "status": status,
                "document_count": count,
                "search_working": search_working,
                "retriever_available": self.retriever is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics (synchronous method).
        
        Returns:
            Dict with statistics
        """
        try:
            stats = {
                "embedding_model": getattr(settings, 'embedding_model', 'mxbai-embed-large'),
                "collection_name": getattr(settings, 'collection_name', 'legal_docs'),
                "vector_store_path": getattr(settings, 'vector_store_path', './chroma_db'),
                "initialized": self.vector_store is not None,
                "retriever_available": self.retriever is not None,
                "reranker_available": self.reranker is not None
            }
            
            if self.vector_store:
                try:
                    stats["document_count"] = self.vector_store._collection.count()
                except:
                    stats["document_count"] = 0
            else:
                stats["document_count"] = 0
                
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}


# Global vector store instance
vector_store_service = VectorStoreService()