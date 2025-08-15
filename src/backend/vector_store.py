"""
Vector store service using ChromaDB with FastEmbed embeddings.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import settings
from models import RetrievedDocument, DocumentMetadata

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages vector storage and retrieval for legal documents."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.reranker = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize FastEmbed embeddings."""
        try:
            # Initialize with proper syntax for current version
            from fastembed import TextEmbedding
            
            # Use fastembed directly instead of langchain wrapper
            self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            # Create a custom embedding function
            class CustomFastEmbedEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    return list(self.model.embed(texts))
                
                def embed_query(self, text):
                    return list(self.model.embed([text]))[0]
            
            self.embeddings = CustomFastEmbedEmbeddings(self._fastembed_model)
            logger.info(f"✓ Initialized FastEmbed with model: BAAI/bge-small-en-v1.5")
            
        except ImportError:
            logger.warning("FastEmbed not available, trying alternative approach...")
            # Fallback to sentence transformers or other embedding method
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("✓ Initialized HuggingFace embeddings as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize any embeddings: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed: {e}")
            # Try the langchain wrapper without extra parameters
            try:
                # This is the working approach for the current version
                from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
                
                # Create instance without any parameters - this should work
                import fastembed
                # Check if we can initialize fastembed directly first
                test_model = fastembed.TextEmbedding()
                del test_model
                
                # Now create the langchain wrapper
                self.embeddings = FastEmbedEmbeddings.__new__(FastEmbedEmbeddings)
                # Manually initialize without going through __init__
                self.embeddings._model = fastembed.TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                
                logger.info("✓ Initialized FastEmbedEmbeddings via workaround")
            except Exception as e2:
                logger.error(f"Workaround failed: {e2}")
                # Final fallback
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("✓ Using HuggingFace embeddings as final fallback")
                except Exception as e3:
                    logger.error(f"All embedding initialization methods failed: {e3}")
                    raise

    
    async def initialize_vector_store(self, force_reload: bool = False) -> bool:
        """Initialize ChromaDB vector store."""
        try:
            # Ensure directory exists
            os.makedirs(settings.vector_store_path, exist_ok=True)
            
            # Check if vector store exists and has data
            persist_directory = settings.vector_store_path
            
            if force_reload or not os.path.exists(persist_directory) or not os.listdir(persist_directory):
                logger.info("Creating new vector store...")
                self.vector_store = Chroma(
                    collection_name=settings.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
                return False  # Needs initial data loading
            else:
                logger.info("Loading existing vector store...")
                self.vector_store = Chroma(
                    collection_name=settings.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
                
                # Initialize retriever
                await self._initialize_retriever()
                return True  # Already has data
        
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def _initialize_retriever(self):
        """Initialize retriever with reranking."""
        try:
            # Base retriever
            base_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": settings.retrieval_k,
                    "score_threshold": settings.similarity_threshold
                }
            )
            
            # Initialize reranker
            if not self.reranker:
                cross_encoder = HuggingFaceCrossEncoder(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.reranker = CrossEncoderReranker(
                    model=cross_encoder,
                    top_k=settings.rerank_k
                )
            
            # Contextual compression retriever with reranking
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )
            
            logger.info("✓ Initialized retriever with reranking")
        
        except Exception as e:
            logger.warning(f"Failed to initialize reranker, using base retriever: {e}")
            # Fallback to base retriever
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
            
            # Add documents to vector store
            doc_ids = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.vector_store.add_documents(documents)
            )
            
            # Persist the vector store
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.vector_store.persist
            )
            
            # Initialize/update retriever
            await self._initialize_retriever()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "documents_added": len(documents),
                "document_ids": doc_ids,
                "processing_time": processing_time,
                "status": "success"
            }
            
            logger.info(f"✓ Added {len(documents)} documents to vector store")
            return result
        
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
            
            # Perform search
            search_kwargs = {"k": k}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata
            
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.retriever.get_relevant_documents(query)
            )
            
            # Convert to RetrievedDocument format
            retrieved_docs = []
            for doc in results:
                retrieved_doc = RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=doc.metadata.get("relevance_score", 0.8)
                )
                retrieved_docs.append(retrieved_doc)
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    async def hybrid_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        alpha: float = 0.5  # Weight between semantic and keyword search
    ) -> List[RetrievedDocument]:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            k = k or settings.retrieval_k
            
            # Semantic search
            semantic_results = await self.similarity_search(query, k=k*2)
            
            # Keyword search (using vector store's built-in search)
            keyword_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_store.similarity_search(
                    query, 
                    k=k*2,
                    search_type="mmr",  # Maximum Marginal Relevance
                    fetch_k=k*3
                )
            )
            
            # Convert keyword results
            keyword_docs = []
            for doc in keyword_results:
                retrieved_doc = RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=0.7  # Default score for keyword search
                )
                keyword_docs.append(retrieved_doc)
            
            # Combine and rerank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_docs, alpha
            )
            
            # Return top k results
            return combined_results[:k]
        
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic search
            return await self.similarity_search(query, k)
    
    def _combine_search_results(
        self,
        semantic_results: List[RetrievedDocument],
        keyword_results: List[RetrievedDocument],
        alpha: float = 0.5
    ) -> List[RetrievedDocument]:
        """Combine and rerank search results."""
        
        # Create a mapping of content to documents
        doc_map = {}
        
        # Add semantic results
        for doc in semantic_results:
            content_hash = hash(doc.content)
            if content_hash not in doc_map:
                doc_map[content_hash] = doc
                doc_map[content_hash].relevance_score *= alpha
            else:
                # Boost score if found in both searches
                doc_map[content_hash].relevance_score += doc.relevance_score * alpha * 0.5
        
        # Add keyword results
        for doc in keyword_results:
            content_hash = hash(doc.content)
            if content_hash not in doc_map:
                doc_map[content_hash] = doc
                doc_map[content_hash].relevance_score *= (1 - alpha)
            else:
                # Boost score if found in both searches
                doc_map[content_hash].relevance_score += doc.relevance_score * (1 - alpha) * 0.5
        
        # Sort by relevance score
        combined_results = list(doc_map.values())
        combined_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return combined_results
    
    async def get_document_count(self) -> int:
        """Get total number of documents in vector store."""
        try:
            if not self.vector_store:
                return 0
            
            collection = self.vector_store._collection
            return collection.count()
        
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store."""
        try:
            if not self.vector_store:
                return False
            
            # Delete documents
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_store.delete(ids=document_ids)
            )
            
            # Persist changes
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vector_store.persist
            )
            
            logger.info(f"✓ Deleted {len(document_ids)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[RetrievedDocument]:
        """Search documents by metadata filters."""
        try:
            if not self.vector_store:
                return []
            
            # Perform filtered search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_store.similarity_search(
                    query="",  # Empty query for metadata-only search
                    k=limit,
                    filter=metadata_filter
                )
            )
            
            # Convert to RetrievedDocument format
            retrieved_docs = []
            for doc in results:
                retrieved_doc = RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=1.0  # Perfect match for metadata search
                )
                retrieved_docs.append(retrieved_doc)
            
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        try:
            if not self.vector_store:
                return {"status": "not_initialized"}
            
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get sample documents to understand the data
            sample_docs = collection.peek(limit=5)
            
            info = {
                "status": "initialized",
                "collection_name": settings.collection_name,
                "document_count": count,
                "embedding_model": settings.embedding_model,
                "sample_metadata_keys": list(sample_docs.get("metadatas", [{}])[0].keys()) if sample_docs.get("metadatas") else [],
                "vector_store_path": settings.vector_store_path
            }
            
            return info
        
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}


# Global vector store instance
vector_store_service = VectorStoreService()