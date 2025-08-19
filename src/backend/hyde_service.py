import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import re

from langchain_google_genai import GoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings, SYSTEM_PROMPTS
from models import HydeQuery

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedHydeService:
    """Enhanced HYDE implementation with better legal document generation."""
    
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Low temperature for consistency
            google_api_key=settings.gemini_api_key
        )
        
        # Legal document templates and patterns
        self.legal_patterns = {
            "law_reference": r"(?:uu|undang-undang)\s+(?:no\.|nomor\s+)?(\d+)\s+tahun\s+(\d{4})",
            "article": r"pasal\s+(\d+)",
            "institution": ["kpk", "komisi pemberantasan korupsi", "kepolisian", "kejaksaan"]
        }
    
    async def generate_hypothetical_documents(
        self, 
        query: str,
        num_docs: int = 3,
        query_context: Optional[Dict[str, Any]] = None
    ) -> HydeQuery:
        """Generate enhanced hypothetical documents for improved retrieval."""
        try:
            if not settings.hyde_enabled:
                return self._create_fallback_hyde(query)
            
            # Analyze query for legal entities
            legal_analysis = self._analyze_legal_query(query)
            
            # Generate hypothetical documents with enhanced prompting
            hypothetical_docs = await self._generate_enhanced_hypothetical_docs(
                query, num_docs, legal_analysis, query_context
            )
            
            # Create multiple enhanced queries
            enhanced_queries = await self._generate_multiple_enhanced_queries(
                query, hypothetical_docs, legal_analysis
            )
            
            # Select best enhanced query
            best_enhanced_query = self._select_best_enhanced_query(
                query, enhanced_queries, legal_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_enhanced_confidence_score(
                query, hypothetical_docs, legal_analysis
            )
            
            hyde_query = HydeQuery(
                original_query=query,
                hypothetical_documents=hypothetical_docs,
                enhanced_query=best_enhanced_query,
                confidence_score=confidence_score
            )
            
            logger.info(f"✓ Enhanced HYDE: {len(hypothetical_docs)} docs, confidence: {confidence_score:.2f}")
            return hyde_query
        
        except Exception as e:
            logger.error(f"Enhanced HYDE generation failed: {e}")
            return self._create_fallback_hyde(query)
    
    def _create_fallback_hyde(self, query: str) -> HydeQuery:
        """Create fallback HYDE when generation fails."""
        return HydeQuery(
            original_query=query,
            hypothetical_documents=[],
            enhanced_query=query,
            confidence_score=0.5
        )
    
    def _analyze_legal_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for legal entities and context."""
        query_lower = query.lower()
        
        analysis = {
            "laws": [],
            "articles": [],
            "institutions": [],
            "legal_concepts": [],
            "query_type": "general",
            "specificity": "medium"
        }
        
        # Extract law references
        law_matches = re.findall(self.legal_patterns["law_reference"], query_lower)
        analysis["laws"] = [f"{num} tahun {year}" for num, year in law_matches]
        
        # Extract article references
        article_matches = re.findall(self.legal_patterns["article"], query_lower)
        analysis["articles"] = article_matches
        
        # Extract institutions
        for institution in self.legal_patterns["institution"]:
            if institution in query_lower:
                analysis["institutions"].append(institution)
        
        # Identify legal concepts
        legal_concepts = [
            "korupsi", "pidana", "perdata", "tata negara", "administrasi",
            "hukum acara", "sanksi", "denda", "penjara", "rehabilitasi",
            "kompensasi", "ganti rugi", "pemberantasan", "penyelidikan",
            "penyidikan", "penuntutan", "pemeriksaan"
        ]
        
        for concept in legal_concepts:
            if concept in query_lower:
                analysis["legal_concepts"].append(concept)
        
        # Determine query type
        if "jelaskan" in query_lower or "uraikan" in query_lower:
            analysis["query_type"] = "explanation"
        elif "apa itu" in query_lower or "pengertian" in query_lower:
            analysis["query_type"] = "definition"
        elif "bagaimana" in query_lower or "cara" in query_lower:
            analysis["query_type"] = "procedure"
        elif analysis["laws"] or analysis["articles"]:
            analysis["query_type"] = "specific_legal"
        
        # Assess specificity
        entity_count = len(analysis["laws"]) + len(analysis["articles"]) + len(analysis["institutions"])
        if entity_count >= 2:
            analysis["specificity"] = "high"
        elif entity_count == 1:
            analysis["specificity"] = "medium"
        else:
            analysis["specificity"] = "low"
        
        return analysis
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_enhanced_hypothetical_docs(
        self, 
        query: str, 
        num_docs: int,
        legal_analysis: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate enhanced hypothetical legal documents."""
        
        # Build context-aware prompt
        prompt = self._build_enhanced_hyde_prompt(query, legal_analysis, query_context, num_docs)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            # Parse and validate hypothetical documents
            hypothetical_docs = self._parse_hypothetical_documents(response, legal_analysis)
            
            # Ensure we have quality documents
            if len(hypothetical_docs) < num_docs:
                # Generate additional docs if needed
                additional_docs = await self._generate_fallback_documents(
                    query, legal_analysis, num_docs - len(hypothetical_docs)
                )
                hypothetical_docs.extend(additional_docs)
            
            return hypothetical_docs[:num_docs]
        
        except Exception as e:
            logger.error(f"Failed to generate enhanced hypothetical documents: {e}")
            return await self._generate_fallback_documents(query, legal_analysis, num_docs)
    
    def _build_enhanced_hyde_prompt(
        self, 
        query: str, 
        legal_analysis: Dict[str, Any],
        query_context: Optional[Dict[str, Any]],
        num_docs: int
    ) -> str:
        """Build enhanced prompt for HYDE generation."""
        
        query_type = legal_analysis.get("query_type", "general")
        specificity = legal_analysis.get("specificity", "medium")
        
        # Base prompt with legal expertise
        base_prompt = f"""Anda adalah ahli hukum Indonesia yang akan membuat {num_docs} paragraf hipotetis yang menjawab pertanyaan hukum berikut: "{query}"

KONTEKS LEGAL:"""
        
        # Add legal context
        if legal_analysis.get("laws"):
            base_prompt += f"\n- Peraturan terkait: {', '.join(['UU ' + law for law in legal_analysis['laws']])}"
        
        if legal_analysis.get("articles"):
            base_prompt += f"\n- Pasal terkait: {', '.join(['Pasal ' + art for art in legal_analysis['articles']])}"
        
        if legal_analysis.get("institutions"):
            base_prompt += f"\n- Lembaga terkait: {', '.join(legal_analysis['institutions'])}"
        
        if legal_analysis.get("legal_concepts"):
            base_prompt += f"\n- Konsep hukum: {', '.join(legal_analysis['legal_concepts'])}"
        
        # Add conversation context if available
        if query_context:
            base_prompt += f"\n- Konteks percakapan: {query_context.get('summary', '')}"
        
        # Type-specific instructions
        if query_type == "explanation":
            instructions = """
TUGAS: Buat paragraf yang memberikan penjelasan komprehensif dan detail tentang topik yang ditanyakan.

SETIAP PARAGRAF HARUS:
1. Memberikan penjelasan mendalam tentang aspek tertentu dari topik
2. Merujuk pada ketentuan hukum yang spesifik dan relevan
3. Menggunakan terminologi hukum Indonesia yang tepat
4. Menyebutkan dasar hukum yang konkret (UU, Pasal, dll)
5. Memberikan konteks implementasi atau penerapan
6. Bersifat informatif dan faktual sesuai peraturan Indonesia"""
        
        elif query_type == "definition":
            instructions = """
TUGAS: Buat paragraf yang memberikan definisi dan penjelasan istilah hukum.

SETIAP PARAGRAF HARUS:
1. Memberikan definisi yang jelas dan komprehensif
2. Merujuk pada definisi resmi dalam peraturan perundang-undangan
3. Menjelaskan konteks penggunaan istilah tersebut
4. Menyebutkan dasar hukum yang mendefinisikan istilah
5. Memberikan contoh penerapan jika relevan"""
        
        elif query_type == "procedure":
            instructions = """
TUGAS: Buat paragraf yang menjelaskan prosedur atau tata cara hukum.

SETIAP PARAGRAF HARUS:
1. Menjelaskan langkah-langkah prosedur secara sistematis
2. Menyebutkan dasar hukum yang mengatur prosedur
3. Menjelaskan persyaratan dan ketentuan yang berlaku
4. Menyebutkan pihak-pihak yang terlibat dan kewenangannya
5. Memberikan informasi tentang jangka waktu dan sanksi"""
        
        else:  # general or specific_legal
            instructions = """
TUGAS: Buat paragraf yang menjawab pertanyaan hukum secara komprehensif.

SETIAP PARAGRAF HARUS:
1. Memberikan informasi yang akurat berdasarkan hukum Indonesia
2. Merujuk pada peraturan perundang-undangan yang relevan
3. Menggunakan terminologi hukum yang tepat
4. Menyebutkan pasal, ayat, atau ketentuan spesifik
5. Memberikan konteks hukum yang diperlukan
6. Bersifat informatif dan dapat dijadikan referensi"""
        
        # Output format
        format_instructions = f"""
FORMAT OUTPUT:
Berikan {num_docs} paragraf yang terpisah, masing-masing dimulai dengan "DOKUMEN:" diikuti paragraf yang substantial (minimal 100 kata).

Contoh:
DOKUMEN: Menurut UU No. 30 Tahun 2002 tentang Komisi Pemberantasan Tindak Pidana Korupsi, [penjelasan detail...]
DOKUMEN: Berdasarkan ketentuan Pasal 6 UU No. 31 Tahun 1999 jo. UU No. 20 Tahun 2001, [penjelasan detail...]

PENTING: Setiap paragraf harus substantial, informatif, dan merujuk pada ketentuan hukum yang spesifik."""
        
        return base_prompt + "\n" + instructions + "\n" + format_instructions
    
    def _parse_hypothetical_documents(self, response: str, legal_analysis: Dict[str, Any]) -> List[str]:
        """Parse and validate hypothetical documents from response."""
        hypothetical_docs = []
        
        # Split response into potential documents
        lines = response.split('\n')
        current_doc = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('DOKUMEN:'):
                # Save previous document if exists
                if current_doc:
                    doc_content = current_doc.replace('DOKUMEN:', '').strip()
                    if self._validate_hypothetical_document(doc_content, legal_analysis):
                        hypothetical_docs.append(doc_content)
                
                # Start new document
                current_doc = line
            elif current_doc and line:
                current_doc += " " + line
        
        # Don't forget the last document
        if current_doc:
            doc_content = current_doc.replace('DOKUMEN:', '').strip()
            if self._validate_hypothetical_document(doc_content, legal_analysis):
                hypothetical_docs.append(doc_content)
        
        # Fallback: if no proper documents found, try to extract from raw response
        if not hypothetical_docs:
            # Split by paragraphs and validate
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                if self._validate_hypothetical_document(paragraph, legal_analysis):
                    hypothetical_docs.append(paragraph)
        
        return hypothetical_docs
    
    def _validate_hypothetical_document(self, doc: str, legal_analysis: Dict[str, Any]) -> bool:
        """Validate quality of hypothetical document."""
        
        # Minimum length check
        if len(doc) < 50:
            return False
        
        # Legal content indicators
        legal_indicators = [
            "undang-undang", "uu", "pasal", "ayat", "ketentuan", "peraturan",
            "hukum", "sanksi", "pidana", "perdata", "kpk", "komisi"
        ]
        
        doc_lower = doc.lower()
        indicator_count = sum(1 for indicator in legal_indicators if indicator in doc_lower)
        
        # Should have at least 2 legal indicators
        if indicator_count < 2:
            return False
        
        # Check for specific legal entities if mentioned in query
        if legal_analysis.get("laws"):
            # At least one law should be mentioned
            law_mentioned = any(law.split()[0] in doc_lower for law in legal_analysis["laws"])
            if not law_mentioned and legal_analysis.get("specificity") == "high":
                return False
        
        return True
    
    async def _generate_fallback_documents(
        self, 
        query: str, 
        legal_analysis: Dict[str, Any], 
        num_docs: int
    ) -> List[str]:
        """Generate fallback documents using simpler approach."""
        
        fallback_docs = []
        
        # Template-based generation
        templates = [
            f"Berdasarkan ketentuan hukum Indonesia, {query.lower()} diatur dalam berbagai peraturan perundang-undangan yang berlaku.",
            f"Dalam konteks hukum Indonesia, {query.lower()} memiliki dasar hukum yang jelas dalam sistem peraturan perundang-undangan.",
            f"Menurut prinsip-prinsip hukum Indonesia, {query.lower()} merupakan hal yang diatur secara komprehensif dalam berbagai ketentuan."
        ]
        
        # Add legal entities to templates if available
        if legal_analysis.get("laws"):
            law_ref = legal_analysis["laws"][0]
            templates.append(f"UU No. {law_ref} mengatur secara detail mengenai {query.lower()} beserta implementasinya.")
        
        if legal_analysis.get("legal_concepts"):
            concept = legal_analysis["legal_concepts"][0]
            templates.append(f"Dalam rangka {concept}, {query.lower()} memiliki peran penting sesuai ketentuan yang berlaku.")
        
        return templates[:num_docs]
    
    async def _generate_multiple_enhanced_queries(
        self, 
        original_query: str, 
        hypothetical_docs: List[str],
        legal_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate multiple enhanced query variations."""
        
        enhanced_queries = [original_query]  # Always include original
        
        if not hypothetical_docs:
            return enhanced_queries
        
        try:
            # Strategy 1: Keyword extraction and enhancement
            keyword_enhanced = self._create_keyword_enhanced_query(
                original_query, hypothetical_docs, legal_analysis
            )
            enhanced_queries.append(keyword_enhanced)
            
            # Strategy 2: Legal entity focused
            entity_enhanced = self._create_entity_enhanced_query(
                original_query, legal_analysis
            )
            enhanced_queries.append(entity_enhanced)
            
            # Strategy 3: Context-based enhancement using LLM
            if len(hypothetical_docs) >= 2:
                context_enhanced = await self._create_context_enhanced_query(
                    original_query, hypothetical_docs[:2]
                )
                enhanced_queries.append(context_enhanced)
            
            # Remove duplicates and empty queries
            enhanced_queries = list(set([q for q in enhanced_queries if q.strip()]))
            
            return enhanced_queries
        
        except Exception as e:
            logger.error(f"Failed to generate multiple enhanced queries: {e}")
            return [original_query]
    
    def _create_keyword_enhanced_query(
        self, 
        query: str, 
        hypothetical_docs: List[str],
        legal_analysis: Dict[str, Any]
    ) -> str:
        """Create keyword-enhanced query."""
        
        # Extract important legal terms from hypothetical documents
        important_terms = set()
        
        legal_keywords = [
            "undang-undang", "peraturan", "pasal", "ayat", "ketentuan",
            "kpk", "komisi", "pemberantasan", "korupsi", "pidana",
            "sanksi", "denda", "penjara", "rehabilitasi", "kompensasi"
        ]
        
        for doc in hypothetical_docs:
            doc_lower = doc.lower()
            for keyword in legal_keywords:
                if keyword in doc_lower:
                    important_terms.add(keyword)
        
        # Add legal entities from analysis
        if legal_analysis.get("laws"):
            important_terms.update([f"uu {law}" for law in legal_analysis["laws"][:2]])
        
        if legal_analysis.get("articles"):
            important_terms.update([f"pasal {art}" for art in legal_analysis["articles"][:2]])
        
        # Combine with original query
        additional_terms = " ".join(list(important_terms)[:5])  # Top 5 terms
        return f"{query} {additional_terms}".strip()
    
    def _create_entity_enhanced_query(self, query: str, legal_analysis: Dict[str, Any]) -> str:
        """Create entity-focused enhanced query."""
        
        entity_parts = [query]
        
        # Add law references
        if legal_analysis.get("laws"):
            law_parts = [f"UU {law}" for law in legal_analysis["laws"]]
            entity_parts.extend(law_parts[:2])  # Top 2 laws
        
        # Add article references
        if legal_analysis.get("articles"):
            article_parts = [f"pasal {art}" for art in legal_analysis["articles"]]
            entity_parts.extend(article_parts[:2])  # Top 2 articles
        
        # Add institutions
        if legal_analysis.get("institutions"):
            entity_parts.extend(legal_analysis["institutions"][:1])  # Top 1 institution
        
        # Add legal concepts
        if legal_analysis.get("legal_concepts"):
            entity_parts.extend(legal_analysis["legal_concepts"][:2])  # Top 2 concepts
        
        return " ".join(entity_parts)
    
    async def _create_context_enhanced_query(self, query: str, top_docs: List[str]) -> str:
        """Create context-enhanced query using LLM."""
        
        context = "\n\n".join(top_docs[:2])
        
        prompt = f"""Berdasarkan pertanyaan dan konteks dokumen hipotetis, buat query pencarian yang lebih efektif.

Pertanyaan: "{query}"

Konteks Dokumen:
{context}

Buat query pencarian yang:
1. Lebih spesifik dan terarah
2. Menggunakan terminologi hukum yang tepat dari konteks
3. Mencakup konsep kunci yang relevan
4. Tetap fokus pada pertanyaan asli
5. Panjang maksimal 15 kata

Query Yang Ditingkatkan:"""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            enhanced_query = response.strip()
            
            # Validate enhanced query
            if 10 <= len(enhanced_query) <= 200 and enhanced_query != query:
                return enhanced_query
            else:
                return query
        
        except Exception as e:
            logger.error(f"Failed to create context-enhanced query: {e}")
            return query
    
    def _select_best_enhanced_query(
        self, 
        original_query: str, 
        enhanced_queries: List[str],
        legal_analysis: Dict[str, Any]
    ) -> str:
        """Select the best enhanced query from candidates."""
        
        if len(enhanced_queries) <= 1:
            return original_query
        
        # Scoring criteria
        scored_queries = []
        
        for query in enhanced_queries:
            score = 0
            query_lower = query.lower()
            
            # Length score (prefer moderate length)
            word_count = len(query.split())
            if 8 <= word_count <= 15:
                score += 2
            elif 5 <= word_count <= 20:
                score += 1
            
            # Legal entity score
            for law in legal_analysis.get("laws", []):
                if law in query_lower:
                    score += 3
            
            for article in legal_analysis.get("articles", []):
                if f"pasal {article}" in query_lower:
                    score += 2
            
            # Legal concept score
            for concept in legal_analysis.get("legal_concepts", []):
                if concept in query_lower:
                    score += 1
            
            # Avoid over-stuffing penalty
            if word_count > 20:
                score -= 2
            
            scored_queries.append((query, score))
        
        # Sort by score and return best
        scored_queries.sort(key=lambda x: x[1], reverse=True)
        
        best_query = scored_queries[0][0]
        logger.info(f"Selected enhanced query with score {scored_queries[0][1]}: {best_query[:50]}...")
        
        return best_query
    
    def _calculate_enhanced_confidence_score(
        self, 
        query: str, 
        hypothetical_docs: List[str],
        legal_analysis: Dict[str, Any]
    ) -> float:
        """Calculate enhanced confidence score."""
        
        factors = []
        
        # 1. Document quality factor
        if hypothetical_docs:
            avg_length = sum(len(doc) for doc in hypothetical_docs) / len(hypothetical_docs)
            doc_quality = min(avg_length / 150, 1.0)  # 150 chars as baseline
            factors.append(("doc_quality", doc_quality, 0.3))
        else:
            factors.append(("doc_quality", 0.2, 0.3))
        
        # 2. Legal entity specificity factor
        entity_count = (len(legal_analysis.get("laws", [])) + 
                       len(legal_analysis.get("articles", [])) + 
                       len(legal_analysis.get("institutions", [])))
        
        if entity_count >= 2:
            specificity_factor = 0.9
        elif entity_count == 1:
            specificity_factor = 0.7
        else:
            specificity_factor = 0.5
        
        factors.append(("specificity", specificity_factor, 0.25))
        
        # 3. Legal concept coverage factor
        concept_count = len(legal_analysis.get("legal_concepts", []))
        concept_factor = min(concept_count / 3, 1.0)
        factors.append(("concept_coverage", concept_factor, 0.2))
        
        # 4. Query complexity factor
        word_count = len(query.split())
        if 8 <= word_count <= 15:
            complexity_factor = 0.9
        elif 5 <= word_count <= 20:
            complexity_factor = 0.7
        else:
            complexity_factor = 0.5
        
        factors.append(("complexity", complexity_factor, 0.15))
        
        # 5. Document count factor
        doc_count_factor = min(len(hypothetical_docs) / 3, 1.0)
        factors.append(("doc_count", doc_count_factor, 0.1))
        
        # Calculate weighted score
        weighted_score = sum(factor * weight for _, factor, weight in factors)
        
        # Ensure score is between 0.1 and 1.0
        final_score = max(0.1, min(weighted_score, 1.0))
        
        logger.debug(f"HYDE confidence factors: {[(name, f'{factor:.2f}') for name, factor, _ in factors]}")
        logger.debug(f"Final HYDE confidence: {final_score:.2f}")
        
        return final_score
    
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
            scores = [getattr(result, 'relevance_score', 0.5) for result in retrieval_results]
            quality_score = sum(scores) / len(scores) if scores else 0.0
        
        logger.info(f"Initial retrieval quality: {quality_score:.2f}")
        
        # Apply HYDE if retrieval quality is below threshold
        if quality_score < threshold:
            logger.info("Applying adaptive HYDE due to low retrieval quality")
            return await self.generate_hypothetical_documents(query)
        
        logger.info("Skipping HYDE due to good retrieval quality")
        return None
    
    async def multi_perspective_hyde(self, query: str) -> HydeQuery:
        """Generate hypothetical documents from multiple legal perspectives."""
        
        perspectives = [
            ("Hukum Pidana", "aspek pidana dan sanksi"),
            ("Hukum Administrasi", "prosedur dan tata cara"),
            ("Hukum Tata Negara", "kelembagaan dan kewenangan"),
            ("Implementasi Praktis", "penerapan dan pelaksanaan")
        ]
        
        all_hypothetical_docs = []
        legal_analysis = self._analyze_legal_query(query)
        
        for perspective_name, perspective_focus in perspectives:
            perspective_query = f"{query} dari perspektif {perspective_focus}"
            try:
                docs = await self._generate_enhanced_hypothetical_docs(
                    perspective_query, 1, legal_analysis
                )
                if docs:
                    # Add perspective context to document
                    enhanced_doc = f"[{perspective_name}] {docs[0]}"
                    all_hypothetical_docs.append(enhanced_doc)
            except Exception as e:
                logger.warning(f"Failed to generate from {perspective_name}: {e}")
                continue
        
        # Generate enhanced queries
        enhanced_queries = await self._generate_multiple_enhanced_queries(
            query, all_hypothetical_docs, legal_analysis
        )
        
        # Select best enhanced query
        best_enhanced_query = self._select_best_enhanced_query(
            query, enhanced_queries, legal_analysis
        )
        
        # Calculate confidence
        confidence_score = self._calculate_enhanced_confidence_score(
            query, all_hypothetical_docs, legal_analysis
        )
        
        return HydeQuery(
            original_query=query,
            hypothetical_documents=all_hypothetical_docs,
            enhanced_query=best_enhanced_query,
            confidence_score=confidence_score
        )


# Global enhanced HYDE service instance
hyde_service = EnhancedHydeService()