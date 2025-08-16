"""
LangGraph Multi-Agent RAG System for Indonesian Legal Assistant with In-Memory Context.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
from collections import deque
from dataclasses import dataclass, asdict
import hashlib

from langgraph.graph import Graph, END
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from config import settings, SYSTEM_PROMPTS
from models import AgentState, QueryRequest, QueryResponse, RetrievedDocument
from vector_store import vector_store_service
from hyde_service import hyde_service

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    id: str
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    
    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ConversationContext:
    """Represents conversation context and summary."""
    chat_id: str
    summary: str
    topics: List[str]
    user_intent: str
    last_updated: datetime
    message_count: int
    
    def to_dict(self):
        return asdict(self)


class InMemoryContextManager:
    """Manages conversation context and history in memory."""
    
    def __init__(self, max_history_per_chat: int = 100, context_window: int = 10):
        # Chat history storage: chat_id -> deque of ChatMessage
        self.chat_histories: Dict[str, deque] = {}
        
        # Context summaries: chat_id -> ConversationContext
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Configuration
        self.max_history_per_chat = max_history_per_chat
        self.context_window = context_window  # Number of recent messages to consider for context
        
        logger.info("✓ Initialized in-memory context manager")
    
    def add_message(self, chat_id: str, message: ChatMessage) -> None:
        """Add a message to chat history."""
        if chat_id not in self.chat_histories:
            self.chat_histories[chat_id] = deque(maxlen=self.max_history_per_chat)
        
        self.chat_histories[chat_id].append(message)
        logger.debug(f"Added message to chat {chat_id}: {message.content[:50]}...")
    
    def get_recent_messages(self, chat_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get recent messages from chat history."""
        if chat_id not in self.chat_histories:
            return []
        
        messages = list(self.chat_histories[chat_id])
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_context(self, chat_id: str) -> Optional[ConversationContext]:
        """Get conversation context summary."""
        return self.conversation_contexts.get(chat_id)
    
    def update_conversation_context(self, chat_id: str, context: ConversationContext) -> None:
        """Update conversation context."""
        self.conversation_contexts[chat_id] = context
        logger.debug(f"Updated context for chat {chat_id}")
    
    def clear_chat_history(self, chat_id: str) -> None:
        """Clear history for a specific chat."""
        if chat_id in self.chat_histories:
            del self.chat_histories[chat_id]
        if chat_id in self.conversation_contexts:
            del self.conversation_contexts[chat_id]
        logger.info(f"Cleared history for chat {chat_id}")
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about stored chats."""
        return {
            "total_chats": len(self.chat_histories),
            "total_messages": sum(len(history) for history in self.chat_histories.values()),
            "contexts_stored": len(self.conversation_contexts)
        }


class LegalRAGAgents:
    """Multi-agent RAG system for Indonesian legal queries with in-memory context."""
    
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            google_api_key=settings.gemini_api_key
        )
        
        # Initialize context manager
        self.context_manager = InMemoryContextManager()
        
        self.graph = None
        self._build_agent_graph()
    
    def _build_agent_graph(self):
        """Build the LangGraph multi-agent workflow."""
        
        # Create workflow graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("context_analyzer", self.analyze_context)
        workflow.add_node("query_analyzer", self.analyze_query)
        workflow.add_node("hyde_generator", self.generate_hyde)
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("answer_generator", self.generate_answer)
        workflow.add_node("document_reviewer", self.review_answer)
        workflow.add_node("quality_controller", self.quality_control)
        workflow.add_node("context_updater", self.update_context)
        
        # Add edges
        workflow.add_edge("context_analyzer", "query_analyzer")
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
                "update_context": "context_updater"
            }
        )
        
        workflow.add_edge("context_updater", END)
        
        # Set entry point
        workflow.set_entry_point("context_analyzer")
        
        # Compile the graph
        self.graph = workflow.compile()
        
        logger.info("✓ Built multi-agent RAG workflow with context management")
    
    async def process_query_with_context(self, request: QueryRequest, chat_id: str, 
                                       chat_history: List[Dict[str, Any]] = None) -> QueryResponse:
        """Process a legal query with conversation context."""
        start_time = datetime.now()
        
        try:
            # Convert chat history to internal format
            if chat_history:
                self._update_chat_history(chat_id, chat_history)
            
            # Add current query to history
            user_message = ChatMessage(
                id=f"msg_{int(start_time.timestamp())}",
                content=request.query,
                role="user",
                timestamp=start_time
            )
            self.context_manager.add_message(chat_id, user_message)
            
            # Initialize agent state with context
            initial_state = AgentState(
                query=request.query,
                retrieved_docs=[],
                iteration_count=0,
                chat_id=chat_id
            )
            
            # Run the multi-agent workflow
            final_state = await self._run_workflow(initial_state)
            
            # Add assistant response to history
            if final_state.final_answer:
                assistant_message = ChatMessage(
                    id=f"msg_{int(datetime.now().timestamp())}",
                    content=final_state.final_answer,
                    role="assistant",
                    timestamp=datetime.now()
                )
                self.context_manager.add_message(chat_id, assistant_message)
            
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
                query_intent=None
            )
            
            logger.info(f"✓ Processed query with context in {processing_time:.2f}s")
            return response
        
        except Exception as e:
            logger.error(f"Query processing with context failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResponse(
                query=request.query,
                answer=f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    def _update_chat_history(self, chat_id: str, chat_history: List[Dict[str, Any]]) -> None:
        """Update internal chat history from external format."""
        for msg_data in chat_history:
            message = ChatMessage(
                id=msg_data.get("id", f"msg_{int(datetime.now().timestamp())}"),
                content=msg_data.get("content", ""),
                role=msg_data.get("role", "user"),
                timestamp=datetime.fromisoformat(msg_data.get("timestamp", datetime.now().isoformat()))
            )
            # Only add if not already in history (to avoid duplicates)
            recent_messages = self.context_manager.get_recent_messages(chat_id, 5)
            if not any(m.id == message.id for m in recent_messages):
                self.context_manager.add_message(chat_id, message)
    
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
            initial_state.final_answer = f"Terjadi kesalahan dalam pemrosesan: {str(e)}"
            initial_state.confidence_score = 0.0
            return initial_state
    
    async def analyze_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation context and generate summary."""
        try:
            chat_id = state.get("chat_id")
            if not chat_id:
                logger.warning("No chat_id provided for context analysis")
                state["conversation_context"] = None
                return state
            
            # Get recent messages
            recent_messages = self.context_manager.get_recent_messages(
                chat_id, self.context_manager.context_window
            )
            
            if len(recent_messages) <= 1:  # Only current message
                state["conversation_context"] = None
                return state
            
            # Generate context summary
            context_summary = await self._generate_context_summary(recent_messages)
            state["conversation_context"] = context_summary
            
            logger.info("Generated conversation context")
            return state
        
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            state["conversation_context"] = None
            return state
    
    async def _generate_context_summary(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Generate a summary of conversation context."""
        
        # Prepare conversation history text
        conversation_text = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages[:-1]  # Exclude current message
        ])
        
        system_prompt = """Anda adalah asisten yang bertugas menganalisis konteks percakapan untuk memberikan ringkasan yang membantu.
        
Tugas Anda:
1. Buat ringkasan singkat tentang topik yang dibahas
2. Identifikasi intent/maksud utama pengguna
3. Tentukan topik-topik kunci yang muncul
4. Berikan konteks yang relevan untuk pertanyaan selanjutnya

Format respons dalam JSON:
{
    "summary": "Ringkasan percakapan dalam 2-3 kalimat",
    "user_intent": "Intent/maksud utama pengguna",
    "topics": ["topik1", "topik2", "topik3"],
    "context_relevance": "Bagaimana konteks ini relevan untuk pertanyaan hukum"
}"""
        
        user_prompt = f"""Analisis percakapan berikut dan berikan konteks yang membantu:

{conversation_text}

Berikan analisis dalam format JSON yang diminta."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(messages)
            )
            
            # Parse JSON response
            try:
                context_data = json.loads(response.strip())
                return context_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "summary": "Percakapan sedang berlangsung tentang topik hukum",
                    "user_intent": "Mencari informasi hukum",
                    "topics": ["hukum"],
                    "context_relevance": "Konteks percakapan sebelumnya dapat membantu"
                }
        
        except Exception as e:
            logger.error(f"Context summary generation failed: {e}")
            return {
                "summary": "Tidak dapat menganalisis konteks percakapan",
                "user_intent": "Tidak diketahui",
                "topics": [],
                "context_relevance": "Konteks tidak tersedia"
            }
    
    async def analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and classify the user query with context."""
        try:
            query = state["query"]
            conversation_context = state.get("conversation_context")
            
            # Enhanced analysis with context
            analysis = {
                "type": "legal_question",
                "complexity": "medium",
                "requires_hyde": len(query.split()) > 5,
                "has_context": conversation_context is not None
            }
            
            if conversation_context:
                analysis["context_topics"] = conversation_context.get("topics", [])
                analysis["user_intent"] = conversation_context.get("user_intent", "")
            
            state["query_analysis"] = analysis
            logger.info(f"Analyzed query with context: {query[:50]}...")
            return state
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return state
    
    async def generate_hyde(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HYDE query if beneficial."""
        try:
            query = state["query"]
            analysis = state.get("query_analysis", {})
            conversation_context = state.get("conversation_context")
            
            if analysis.get("requires_hyde", True) and settings.hyde_enabled:
                # Enhanced HYDE with context
                context_info = ""
                if conversation_context:
                    context_info = f"\nKonteks percakapan: {conversation_context.get('summary', '')}"
                
                enhanced_query = query + context_info
                hyde_query = await hyde_service.generate_hypothetical_documents(enhanced_query)
                state["hyde_query"] = hyde_query.dict()
                logger.info("Generated context-aware HYDE query")
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
            conversation_context = state.get("conversation_context")
            
            # Use enhanced query if HYDE was generated
            if hyde_info and hyde_info.get("enhanced_query"):
                search_query = hyde_info["enhanced_query"]
                logger.info("Using HYDE-enhanced query for retrieval")
            else:
                search_query = query
                # Add context topics to search if available
                if conversation_context and conversation_context.get("topics"):
                    context_terms = " ".join(conversation_context["topics"])
                    search_query = f"{query} {context_terms}"
            
            # Retrieve documents using hybrid search
            retrieved_docs = await vector_store_service.hybrid_search(
                search_query,
                k=settings.retrieval_k
            )
            
            state["retrieved_docs"] = [doc.dict() for doc in retrieved_docs]
            logger.info(f"Retrieved {len(retrieved_docs)} documents with context")
            
            return state
        
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_docs"] = []
            return state
    
    async def generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer based on retrieved documents and context."""
        try:
            query = state["query"]
            retrieved_docs = state.get("retrieved_docs", [])
            conversation_context = state.get("conversation_context")
            iteration = state.get("iteration_count", 0)
            
            # Prepare context from retrieved documents
            document_context = self._prepare_context(retrieved_docs)
            
            # Generate answer with conversation context
            answer = await self._generate_legal_answer_with_context(
                query, document_context, conversation_context
            )
            
            if iteration == 0:
                state["initial_answer"] = answer
            else:
                state["reviewed_answer"] = answer
            
            state["current_answer"] = answer
            logger.info("Generated context-aware legal answer")
            
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
            conversation_context = state.get("conversation_context")
            
            # Prepare review context
            context = self._prepare_context(retrieved_docs)
            
            # Review the answer with conversation context
            review_result = await self._review_legal_answer(query, current_answer, context)
            
            state["review_result"] = review_result
            state["feedback"] = review_result.get("feedback", "")
            
            logger.info("Completed context-aware answer review")
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
                # Accept the answer and update context
                state["final_answer"] = current_answer
                state["confidence_score"] = confidence
                state["should_continue"] = False
                state["update_context"] = True
                
                # Prepare sources
                retrieved_docs = state.get("retrieved_docs", [])
                state["sources"] = retrieved_docs
                
                logger.info(f"Final answer accepted with score {review_score}")
            else:
                # Iterate for improvement
                state["iteration_count"] = iteration + 1
                state["should_continue"] = True
                state["update_context"] = False
                logger.info(f"Answer needs improvement, iteration {iteration + 1}")
            
            return state
        
        except Exception as e:
            logger.error(f"Quality control failed: {e}")
            state["final_answer"] = state.get("current_answer", "Terjadi kesalahan")
            state["confidence_score"] = 0.3
            state["should_continue"] = False
            state["update_context"] = True
            return state
    
    async def update_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update conversation context after successful answer generation."""
        try:
            chat_id = state.get("chat_id")
            if not chat_id:
                return state
            
            # Get current conversation context
            current_context = self.context_manager.get_conversation_context(chat_id)
            
            # Create updated context
            recent_messages = self.context_manager.get_recent_messages(chat_id)
            
            if len(recent_messages) >= 2:  # Have at least question and answer
                context_summary = await self._generate_context_summary(recent_messages)
                
                updated_context = ConversationContext(
                    chat_id=chat_id,
                    summary=context_summary.get("summary", ""),
                    topics=context_summary.get("topics", []),
                    user_intent=context_summary.get("user_intent", ""),
                    last_updated=datetime.now(),
                    message_count=len(recent_messages)
                )
                
                self.context_manager.update_conversation_context(chat_id, updated_context)
                logger.info(f"Updated conversation context for chat {chat_id}")
            
            return state
        
        except Exception as e:
            logger.error(f"Context update failed: {e}")
            return state
    
    def should_continue(self, state: Dict[str, Any]) -> str:
        """Decide whether to continue iteration or end."""
        if state.get("should_continue", False):
            return "continue"
        elif state.get("update_context", False):
            return "update_context"
        else:
            return "update_context"  # Always update context at the end
    
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
    
    async def _generate_legal_answer_with_context(self, query: str, document_context: str, 
                                                conversation_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate legal answer using LLM with conversation context."""
        
        system_prompt = SYSTEM_PROMPTS["legal_assistant"]
        
        # Add conversation context to system prompt if available
        if conversation_context:
            context_addition = f"""
            
KONTEKS PERCAKAPAN SEBELUMNYA:
- Ringkasan: {conversation_context.get('summary', 'Tidak tersedia')}
- Intent pengguna: {conversation_context.get('user_intent', 'Tidak diketahui')}
- Topik yang dibahas: {', '.join(conversation_context.get('topics', []))}
- Relevansi: {conversation_context.get('context_relevance', 'Tidak tersedia')}

Gunakan konteks percakapan ini untuk memberikan jawaban yang lebih relevan dan konsisten dengan diskusi sebelumnya."""
            
            system_prompt += context_addition
        
        user_prompt = f"""Pertanyaan: {query}

Konteks Dokumen Hukum:
{document_context}

Berdasarkan konteks dokumen hukum dan percakapan sebelumnya (jika ada), berikan jawaban yang:
1. Akurat dan berdasarkan peraturan yang tersedia
2. Konsisten dengan konteks percakapan sebelumnya
3. Jelas dan mudah dipahami
4. Mencantumkan referensi peraturan yang relevan
5. Memberikan konteks hukum yang diperlukan
6. Menggunakan bahasa Indonesia yang baik dan benar

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
6. Konsistensi dengan konteks percakapan (1-10)

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
    
    # Context management methods
    def get_chat_history(self, chat_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat history for a specific chat."""
        messages = self.context_manager.get_recent_messages(chat_id, limit)
        return [msg.to_dict() for msg in messages]
    
    def clear_chat_context(self, chat_id: str) -> None:
        """Clear context for a specific chat."""
        self.context_manager.clear_chat_history(chat_id)
        logger.info(f"Cleared context for chat {chat_id}")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context management statistics."""
        return self.context_manager.get_chat_stats()


# Global RAG agents instance
rag_agents = LegalRAGAgents()