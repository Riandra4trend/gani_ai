"""
LangGraph Multi-Agent RAG System for Indonesian Legal Assistant.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from langgraph.graph import Graph, END
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from config import settings, SYSTEM_PROMPTS
from models import AgentState, QueryRequest, QueryResponse, RetrievedDocument
from vector_store import vector_store_service
from hyde_service import hyde_service

# Setup logging
logger = logging.getLogger(__name__)


class LegalRAGAgents:
    """Multi-agent RAG system for Indonesian legal queries."""
    
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            google_api_key=settings.gemini_api_key
        )
        
        self.graph = None
        self._build_agent_graph()
    
    def _build_agent_graph(self):
        """Build the LangGraph multi-agent workflow."""
        
        # Create workflow graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("query_analyzer", self.analyze_query)
        workflow.add_node("hyde_generator", self.generate_hyde)
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("answer_generator", self.generate_answer)
        workflow.add_node("document_reviewer", self.review_answer)
        workflow.add_node("quality_controller", self.quality_control)
        
        # Add edges
        workflow.add_edge("query_analyzer", "hyde_generator")
        workflow.add_edge("hyde_generator", "retriever")
        workflow.add_edge("retriever", "answer_generator")
        workflow.add_edge("answer_generator", "document_reviewer")
        workflow.add_edge("document_reviewer", "quality_controller")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "quality_controller",
            self.should_continue,
            {
                "continue": "answer_generator",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("query_analyzer")
        
        # Compile the graph
        self.graph = workflow.compile()
        
        logger.info("✓ Built multi-agent RAG workflow")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a legal query through the multi-agent system."""
        start_time = datetime.now()
        
        try:
            # Initialize agent state
            initial_state = AgentState(
                query=request.query,
                retrieved_docs=[],
                iteration_count=0
            )
            
            # Run the multi-agent workflow
            final_state = await self._run_workflow(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response
            response = QueryResponse(
                query=request.query,
                answer=final_state.final_answer or "Maaf, tidak dapat memberikan jawaban yang memadai.",
                sources=final_state.sources,
                confidence_score=final_state.confidence_score,
                processing_time=processing_time,
                hyde_info=final_state.hyde_query,
                query_intent=None  # Could be enhanced with intent classification
            )
            
            logger.info(f"✓ Processed query in {processing_time:.2f}s with confidence {final_state.confidence_score:.2f}")
            return response
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return error response
            return QueryResponse(
                query=request.query,
                answer=f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    async def _run_workflow(self, initial_state: AgentState) -> AgentState:
        """Run the multi-agent workflow."""
        try:
            # Execute the graph workflow
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.graph.invoke(initial_state.dict())
            )
            
            # Convert back to AgentState
            final_state = AgentState(**result)
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Return state with error
            initial_state.final_answer = f"Terjadi kesalahan dalam pemrosesan: {str(e)}"
            initial_state.confidence_score = 0.0
            return initial_state
    
    async def analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and classify the user query."""
        try:
            query = state["query"]
            
            # For now, simple analysis - can be enhanced with classification
            state["query_analysis"] = {
                "type": "legal_question",
                "complexity": "medium",
                "requires_hyde": len(query.split()) > 5
            }
            
            logger.info(f"Analyzed query: {query[:50]}...")
            return state
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return state
    
    async def generate_hyde(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HYDE query if beneficial."""
        try:
            query = state["query"]
            analysis = state.get("query_analysis", {})
            
            if analysis.get("requires_hyde", True) and settings.hyde_enabled:
                hyde_query = await hyde_service.generate_hypothetical_documents(query)
                state["hyde_query"] = hyde_query.dict()
                logger.info("Generated HYDE query")
            else:
                state["hyde_query"] = None
                logger.info("Skipped HYDE generation")
            
            return state
        
        except Exception as e:
            logger.error(f"HYDE generation failed: {e}")
            state["hyde_query"] = None
            return state
    
    async def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents from vector store."""
        try:
            query = state["query"]
            hyde_info = state.get("hyde_query")
            
            # Use enhanced query if HYDE was generated
            if hyde_info and hyde_info.get("enhanced_query"):
                search_query = hyde_info["enhanced_query"]
                logger.info("Using HYDE-enhanced query for retrieval")
            else:
                search_query = query
            
            # Retrieve documents using hybrid search
            retrieved_docs = await vector_store_service.hybrid_search(
                search_query,
                k=settings.retrieval_k
            )
            
            state["retrieved_docs"] = [doc.dict() for doc in retrieved_docs]
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            return state
        
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_docs"] = []
            return state
    
    async def generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer based on retrieved documents."""
        try:
            query = state["query"]
            retrieved_docs = state.get("retrieved_docs", [])
            iteration = state.get("iteration_count", 0)
            
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Generate answer
            answer = await self._generate_legal_answer(query, context)
            
            if iteration == 0:
                state["initial_answer"] = answer
            else:
                state["reviewed_answer"] = answer
            
            state["current_answer"] = answer
            logger.info("Generated legal answer")
            
            return state
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            state["current_answer"] = "Maaf, tidak dapat menghasilkan jawaban."
            return state
    
    async def review_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Review the generated answer for accuracy and completeness."""
        try:
            current_answer = state.get("current_answer", "")
            retrieved_docs = state.get("retrieved_docs", [])
            query = state["query"]
            
            # Prepare review context
            context = self._prepare_context(retrieved_docs)
            
            # Review the answer
            review_result = await self._review_legal_answer(query, current_answer, context)
            
            state["review_result"] = review_result
            state["feedback"] = review_result.get("feedback", "")
            
            logger.info("Completed answer review")
            return state
        
        except Exception as e:
            logger.error(f"Answer review failed: {e}")
            state["review_result"] = {"score": 5, "feedback": "Review tidak tersedia"}
            return state
    
    async def quality_control(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Final quality control and decision making."""
        try:
            current_answer = state.get("current_answer", "")
            review_result = state.get("review_result", {})
            iteration = state.get("iteration_count", 0)
            
            review_score = review_result.get("score", 5)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(review_score, iteration)
            
            # Decide if answer is acceptable or needs iteration
            if review_score >= 7 or iteration >= settings.max_iterations - 1:
                # Accept the answer
                state["final_answer"] = current_answer
                state["confidence_score"] = confidence
                state["should_continue"] = False
                
                # Prepare sources
                retrieved_docs = state.get("retrieved_docs", [])
                state["sources"] = retrieved_docs
                
                logger.info(f"Final answer accepted with score {review_score}")
            else:
                # Iterate for improvement
                state["iteration_count"] = iteration + 1
                state["should_continue"] = True
                logger.info(f"Answer needs improvement, iteration {iteration + 1}")
            
            return state
        
        except Exception as e:
            logger.error(f"Quality control failed: {e}")
            state["final_answer"] = state.get("current_answer", "Terjadi kesalahan")
            state["confidence_score"] = 0.3
            state["should_continue"] = False
            return state
    
    def should_continue(self, state: Dict[str, Any]) -> str:
        """Decide whether to continue iteration or end."""
        return "continue" if state.get("should_continue", False) else "end"
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        if not retrieved_docs:
            return "Tidak ada dokumen relevan ditemukan."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 documents
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            source_info = ""
            if metadata.get("title"):
                source_info += f"Sumber: {metadata['title']}"
            if metadata.get("regulation_number"):
                source_info += f" ({metadata['regulation_number']})"
            
            context_part = f"[Dokumen {i}]\n"
            if source_info:
                context_part += f"{source_info}\n"
            context_part += f"{content}\n"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    async def _generate_legal_answer(self, query: str, context: str) -> str:
        """Generate legal answer using LLM."""
        
        system_prompt = SYSTEM_PROMPTS["legal_assistant"]
        
        user_prompt = f"""Pertanyaan: {query}

Konteks Dokumen Hukum:
{context}

Berdasarkan konteks dokumen hukum di atas, berikan jawaban yang:
1. Akurat dan berdasarkan peraturan yang tersedia
2. Jelas dan mudah dipahami
3. Mencantumkan referensi peraturan yang relevan
4. Memberikan konteks hukum yang diperlukan
5. Menggunakan bahasa Indonesia yang baik dan benar

Jika informasi tidak tersedia dalam dokumen, nyatakan dengan jelas dan sarankan konsultasi dengan ahli hukum.

Jawaban:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(messages)
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Legal answer generation failed: {e}")
            return "Maaf, tidak dapat menghasilkan jawaban yang memadai."
    
    async def _review_legal_answer(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Review the generated legal answer."""
        
        system_prompt = SYSTEM_PROMPTS["document_reviewer"]
        
        user_prompt = f"""Pertanyaan: {query}

Jawaban yang Diberikan:
{answer}

Konteks Dokumen:
{context}

Evaluasi jawaban berdasarkan:
1. Keakuratan informasi hukum (1-10)
2. Kelengkapan jawaban (1-10)
3. Kejelasan penjelasan (1-10)
4. Kesesuaian dengan pertanyaan (1-10)
5. Penggunaan referensi yang tepat (1-10)

Berikan evaluasi dalam format:
SKOR: [rata-rata skor 1-10]
FEEDBACK: [saran perbaikan spesifik]"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(messages)
            )
            
            # Parse review response
            review_text = response.strip()
            score = 5  # default score
            feedback = "Tidak ada feedback spesifik"
            
            lines = review_text.split('\n')
            for line in lines:
                if line.startswith('SKOR:'):
                    try:
                        score_text = line.replace('SKOR:', '').strip()
                        score = float(score_text)
                    except ValueError:
                        pass
                elif line.startswith('FEEDBACK:'):
                    feedback = line.replace('FEEDBACK:', '').strip()
            
            return {
                "score": min(max(score, 1), 10),  # Clamp between 1-10
                "feedback": feedback,
                "raw_review": review_text
            }
        
        except Exception as e:
            logger.error(f"Answer review failed: {e}")
            return {
                "score": 5,
                "feedback": f"Review gagal: {str(e)}",
                "raw_review": ""
            }
    
    def _calculate_confidence_score(self, review_score: float, iteration: int) -> float:
        """Calculate overall confidence score."""
        
        # Base confidence from review score (0-1 scale)
        base_confidence = review_score / 10.0
        
        # Penalty for multiple iterations
        iteration_penalty = iteration * 0.1
        
        # Final confidence
        confidence = max(0.1, min(base_confidence - iteration_penalty, 1.0))
        
        return round(confidence, 2)


# Global RAG agents instance
rag_agents = LegalRAGAgents()