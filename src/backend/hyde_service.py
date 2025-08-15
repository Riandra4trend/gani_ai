"""
HYDE (Hypothetical Document Embeddings) service for improved retrieval.
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings, SYSTEM_PROMPTS
from models import HydeQuery

# Setup logging
logger = logging.getLogger(__name__)


class HydeService:
    """HYDE implementation for enhanced document retrieval."""
    
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=settings.gemini_api_key
        )
    
    async def generate_hypothetical_documents(
        self, 
        query: str,
        num_docs: int = 3
    ) -> HydeQuery:
        """Generate hypothetical documents for improved retrieval."""
        try:
            if not settings.hyde_enabled:
                return HydeQuery(
                    original_query=query,
                    hypothetical_documents=[],
                    enhanced_query=query,
                    confidence_score=1.0
                )
            
            # Generate hypothetical documents
            hypothetical_docs = await self._generate_hypothetical_docs(query, num_docs)
            
            # Enhance the original query
            enhanced_query = await self._enhance_query(query, hypothetical_docs)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(query, hypothetical_docs)
            
            hyde_query = HydeQuery(
                original_query=query,
                hypothetical_documents=hypothetical_docs,
                enhanced_query=enhanced_query,
                confidence_score=confidence_score
            )
            
            logger.info(f"✓ Generated HYDE query with {len(hypothetical_docs)} hypothetical documents")
            return hyde_query
        
        except Exception as e:
            logger.error(f"HYDE generation failed: {e}")
            # Fallback to original query
            return HydeQuery(
                original_query=query,
                hypothetical_documents=[],
                enhanced_query=query,
                confidence_score=0.5
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_hypothetical_docs(self, query: str, num_docs: int) -> List[str]:
        """Generate hypothetical legal documents."""
        
        prompt = f"""Anda adalah ahli hukum Indonesia yang akan membuat dokumen hipotetis untuk membantu pencarian informasi hukum.

Berdasarkan pertanyaan: "{query}"

Buatlah {num_docs} paragraf hipotetis yang mungkin menjawab pertanyaan tersebut. Setiap paragraf harus:
1. Merujuk pada peraturan hukum Indonesia yang relevan
2. Menggunakan terminologi hukum yang tepat
3. Memberikan konteks yang spesifik
4. Bersifat informatif dan akurat

Format: Berikan setiap paragraf dalam baris terpisah yang dimulai dengan "DOKUMEN:"

Contoh:
DOKUMEN: Menurut UUD 1945 Pasal 28...
DOKUMEN: Berdasarkan UU No. 6 Tahun 2023...

Mulai:"""

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            # Parse the response
            hypothetical_docs = []
            for line in response.split('\n'):
                if line.strip().startswith('DOKUMEN:'):
                    doc = line.replace('DOKUMEN:', '').strip()
                    if doc and len(doc) > 50:  # Ensure meaningful content
                        hypothetical_docs.append(doc)
            
            # Fallback if parsing fails
            if not hypothetical_docs:
                hypothetical_docs = [response.strip()]
            
            return hypothetical_docs[:num_docs]
        
        except Exception as e:
            logger.error(f"Failed to generate hypothetical documents: {e}")
            return []
    
    async def _enhance_query(self, original_query: str, hypothetical_docs: List[str]) -> str:
        """Enhance the original query based on hypothetical documents."""
        
        if not hypothetical_docs:
            return original_query
        
        try:
            # Create context from hypothetical documents
            context = "\n\n".join(hypothetical_docs[:2])  # Use top 2 docs
            
            prompt = f"""Berdasarkan pertanyaan asli dan konteks dokumen hipotetis, buat query pencarian yang lebih baik.

Pertanyaan Asli: "{original_query}"

Konteks Dokumen:
{context}

Buat query pencarian yang:
1. Lebih spesifik dan terarah
2. Menggunakan terminologi hukum yang tepat
3. Mencakup konsep kunci dari dokumen hipotetis
4. Tetap fokus pada pertanyaan asli

Query yang Ditingkatkan:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            enhanced_query = response.strip()
            
            # Validate enhanced query
            if len(enhanced_query) < 10 or len(enhanced_query) > 500:
                return original_query
            
            return enhanced_query
        
        except Exception as e:
            logger.error(f"Failed to enhance query: {e}")
            return original_query
    
    def _calculate_confidence_score(self, query: str, hypothetical_docs: List[str]) -> float:
        """Calculate confidence score for HYDE generation."""
        
        if not hypothetical_docs:
            return 0.3
        
        # Factors for confidence calculation
        factors = []
        
        # 1. Number of generated documents
        doc_factor = min(len(hypothetical_docs) / 3, 1.0)
        factors.append(doc_factor)
        
        # 2. Average length of hypothetical documents
        avg_length = sum(len(doc) for doc in hypothetical_docs) / len(hypothetical_docs)
        length_factor = min(avg_length / 200, 1.0)  # 200 chars as baseline
        factors.append(length_factor)
        
        # 3. Presence of legal terms
        legal_terms = [
            'undang-undang', 'peraturan', 'pasal', 'ayat', 'uu', 'uud',
            'pemerintah', 'hukum', 'ketentuan', 'sanksi', 'norma'
        ]
        
        legal_term_count = 0
        total_words = 0
        
        for doc in hypothetical_docs:
            doc_lower = doc.lower()
            total_words += len(doc.split())
            for term in legal_terms:
                legal_term_count += doc_lower.count(term)
        
        legal_factor = min(legal_term_count / max(total_words / 50, 1), 1.0)
        factors.append(legal_factor)
        
        # 4. Query length factor (longer queries often work better with HYDE)
        query_factor = min(len(query.split()) / 10, 1.0)
        factors.append(query_factor)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.3, 0.1]
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return max(0.1, min(confidence, 1.0))
    
    async def adaptive_hyde(
        self,
        query: str,
        retrieval_results: List[Any],
        threshold: float = 0.6
    ) -> Optional[HydeQuery]:
        """Adaptive HYDE that generates hypothetical docs based on initial retrieval quality."""
        
        # Calculate retrieval quality
        if not retrieval_results:
            quality_score = 0.0
        else:
            # Use relevance scores from retrieval results
            scores = [getattr(result, 'relevance_score', 0.5) for result in retrieval_results]
            quality_score = sum(scores) / len(scores) if scores else 0.0
        
        logger.info(f"Initial retrieval quality: {quality_score:.2f}")
        
        # Apply HYDE if retrieval quality is below threshold
        if quality_score < threshold:
            logger.info("Applying HYDE due to low retrieval quality")
            return await self.generate_hypothetical_documents(query)
        
        logger.info("Skipping HYDE due to good retrieval quality")
        return None
    
    async def multi_perspective_hyde(self, query: str) -> HydeQuery:
        """Generate hypothetical documents from multiple legal perspectives."""
        
        perspectives = [
            "Perspektif Hukum Pidana",
            "Perspektif Hukum Perdata", 
            "Perspektif Hukum Tata Negara",
            "Perspektif Hukum Administrasi"
        ]
        
        all_hypothetical_docs = []
        
        for perspective in perspectives:
            perspective_query = f"Dari {perspective}: {query}"
            try:
                docs = await self._generate_hypothetical_docs(perspective_query, 1)
                if docs:
                    all_hypothetical_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to generate from {perspective}: {e}")
                continue
        
        # Enhance query with multi-perspective context
        enhanced_query = await self._enhance_query(query, all_hypothetical_docs)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence_score(query, all_hypothetical_docs)
        
        return HydeQuery(
            original_query=query,
            hypothetical_documents=all_hypothetical_docs,
            enhanced_query=enhanced_query,
            confidence_score=confidence_score
        )


# Global HYDE service instance
hyde_service = HydeService()