import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import re
from langgraph.graph import Graph, END

from langgraph.graph import StateGraph, END
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
    
    def load_chat_history(self, chat_id: str, chat_history: List[Dict[str, Any]]) -> None:
        """Load chat history from external source (e.g., frontend)."""
        if chat_id not in self.chat_histories:
            self.chat_histories[chat_id] = deque(maxlen=self.max_history_per_chat)
        
        # Clear existing history for this chat
        self.chat_histories[chat_id].clear()
        
        # Load messages from external history
        for msg_data in chat_history:
            try:
                # Handle different timestamp formats
                timestamp_str = msg_data.get("timestamp", datetime.now().isoformat())
                if isinstance(timestamp_str, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                message = ChatMessage(
                    id=msg_data.get("id", f"msg_{int(datetime.now().timestamp())}"),
                    content=msg_data.get("content", ""),
                    role=msg_data.get("role", "user"),
                    timestamp=timestamp
                )
                
                self.chat_histories[chat_id].append(message)
            except Exception as e:
                logger.warning(f"Failed to load chat history message: {e}")
                continue
        
        logger.info(f"Loaded {len(chat_history)} messages for chat {chat_id}")

class LegalRAGAgents:
    """Multi-agent RAG system for Indonesian legal queries with enhanced retrieval."""
    
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            google_api_key=settings.gemini_api_key
        )
        
        # Initialize context manager
        self.context_manager = InMemoryContextManager()
        
        self.graph = None
        self.workflow = None
        self._build_agent_graph()
        
    
    def _build_agent_graph(self):
        """Build the LangGraph multi-agent workflow."""
        
        # Create StateGraph workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("context_analyzer", self.analyze_context)
        workflow.add_node("query_analyzer", self.analyze_query)
        workflow.add_node("smart_retriever", self.smart_retrieve_documents)  # Enhanced retrieval
        workflow.add_node("answer_generator", self.generate_answer)
        workflow.add_node("document_reviewer", self.review_answer)
        workflow.add_node("quality_controller", self.quality_control)
        workflow.add_node("context_updater", self.update_context)
        
        # Add edges - Simplified workflow for better performance
        workflow.add_edge("context_analyzer", "query_analyzer")
        workflow.add_edge("query_analyzer", "smart_retriever")
        workflow.add_edge("smart_retriever", "answer_generator")
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
        
        logger.info("✓ Built enhanced multi-agent RAG workflow")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a legal query without conversation context (standard processing)."""
        start_time = datetime.now()
        
        try:
            # Generate a temporary chat_id for this single query
            temp_chat_id = f"temp_{int(start_time.timestamp())}"
            
            logger.info(f"🔄 Processing query without context: {request.query[:100]}...")
            
            # Initialize agent state without context
            initial_state = AgentState(
                query=request.query,
                retrieved_docs=[],
                iteration_count=0,
                chat_id=temp_chat_id
            )
            
            # Run the multi-agent workflow
            final_state = await self._run_workflow(initial_state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Map the internal intent to valid QueryResponse intent
            analysis_intent = final_state.query_analysis.get('intent', 'general_inquiry') if final_state.query_analysis else 'general_inquiry'
            valid_intent = self._map_analysis_intent_to_response_intent(analysis_intent)
            
            # Create response
            response = QueryResponse(
                query=request.query,
                answer=final_state.final_answer or "Maaf, tidak dapat memberikan jawaban yang memadai.",
                sources=final_state.sources or [],
                confidence_score=final_state.confidence_score or 0.0,
                processing_time=processing_time,
                hyde_info=final_state.hyde_query,  # This should now be a dict or None
                query_intent=valid_intent  # Use mapped intent
            )
            
            # Clean up temporary chat context
            self.context_manager.clear_chat_history(temp_chat_id)
            
            logger.info(f"✓ Processed query without context in {processing_time:.2f}s")
            return response
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResponse(
                query=request.query,
                answer=f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                query_intent="legal_question"  # Use valid enum value
            )
    
    async def update_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update conversation context after successful answer generation."""
        try:
            chat_id = state.get("chat_id")
            if not chat_id or chat_id.startswith("temp_"):
                # Skip context update for temporary chats
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
    
    async def process_query_with_context(self, request: QueryRequest, chat_id: str, chat_history: List[Dict[str, Any]]) -> QueryResponse:
        """Process query with context from frontend chat history."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query with context: {len(chat_history)} messages from frontend")
            
            # Load chat history into context manager
            self.context_manager.load_chat_history(chat_id, chat_history)
            
            # FIXED: Properly initialize AgentState with frontend_chat_history
            state = AgentState(
                query=request.query,
                chat_id=chat_id,
                frontend_chat_history=chat_history,  # Set this field properly
                context_source="frontend"  # Set context source directly
            )
            
            logger.info(f"Initialized state with {len(chat_history)} frontend messages")
            
            # Run the workflow with the properly initialized state
            final_state = await self._run_workflow(state)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Map the internal intent to valid QueryResponse intent
            analysis_intent = final_state.query_analysis.get('intent', 'general_inquiry') if final_state.query_analysis else 'general_inquiry'
            valid_intent = self._map_analysis_intent_to_response_intent(analysis_intent)
            
            # Return the response
            return QueryResponse(
                query=request.query,
                answer=final_state.final_answer or "Maaf, tidak dapat memberikan jawaban yang memadai.",
                sources=final_state.sources or [],
                confidence_score=final_state.confidence_score or 0.0,
                processing_time=processing_time,
                hyde_info=final_state.hyde_query,
                query_intent=valid_intent
            )
            
        except Exception as e:
            logger.error(f"Context-aware query processing failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Fallback to standard processing
            logger.info("Falling back to standard processing without context")
            return await self.process_query(request)
    
    def _update_chat_history(self, chat_id: str, chat_history: List[Dict[str, Any]]) -> None:
        """Update internal chat history from external format."""
        self.context_manager.load_chat_history(chat_id, chat_history)
    
    async def _run_workflow(self, initial_state: AgentState) -> AgentState:
        """Run the multi-agent workflow asynchronously."""
        try:
            logger.info("Starting workflow execution...")
            
            # FIXED: Use self.graph instead of self.workflow
            # Use async invoke for proper async execution
            result = await self.graph.ainvoke(initial_state)
            
            # Convert result to AgentState if needed
            if isinstance(result, dict):
                final_state = AgentState(**result)
            else:
                final_state = result
                
            logger.info("Workflow execution completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Return a failed state
            initial_state.final_answer = f"Terjadi kesalahan dalam pemrosesan: {str(e)}"
            initial_state.confidence_score = 0.0
            initial_state.sources = []
            return initial_state
    
    async def analyze_context(self, state: AgentState) -> AgentState:
        """Analyze conversation context from frontend chat history."""
        try:
            chat_id = state.chat_id
            if not chat_id:
                logger.warning("No chat_id provided for context analysis")
                state.conversation_context = None
                return state
            
            # FIXED: Check if frontend chat history is available in state
            frontend_chat_history = state.frontend_chat_history
            
            if frontend_chat_history and len(frontend_chat_history) > 0:
                # Use frontend chat history directly
                logger.info(f"Using frontend chat history: {len(frontend_chat_history)} messages")
                
                # Convert frontend messages to internal format
                recent_messages = []
                for msg_data in frontend_chat_history[-self.context_manager.context_window:]:
                    try:
                        # Handle different timestamp formats
                        timestamp_str = msg_data.get("timestamp", datetime.now().isoformat())
                        if isinstance(timestamp_str, str):
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            except ValueError:
                                timestamp = datetime.now()
                        else:
                            timestamp = datetime.now()
                        
                        message = ChatMessage(
                            id=msg_data.get("id", f"msg_{int(datetime.now().timestamp())}"),
                            content=msg_data.get("content", ""),
                            role=msg_data.get("role", "user"),
                            timestamp=timestamp
                        )
                        recent_messages.append(message)
                    except Exception as e:
                        logger.warning(f"Failed to process frontend message: {e}")
                        continue
            else:
                # Fallback to internal context manager
                logger.info("No frontend chat history, using internal context manager")
                recent_messages = self.context_manager.get_recent_messages(
                    chat_id, self.context_manager.context_window
                )
            
            if len(recent_messages) <= 1:  # Only current message
                state.conversation_context = None
                return state
            
            # Generate context summary
            context_summary = await self._generate_context_summary(recent_messages)
            state.conversation_context = context_summary
            
            logger.info(f"Generated conversation context from {len(recent_messages)} messages")
            return state
        
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            state.conversation_context = None
            return state
    
    async def analyze_query(self, state: AgentState) -> AgentState:
        """Enhanced query analysis with legal domain understanding."""
        try:
            query = state.query
            conversation_context = state.conversation_context
            
            # Extract legal entities and concepts
            legal_entities = self._extract_legal_entities(query)
            
            # Analyze query complexity and type
            analysis = {
                "type": self._classify_query_type(query),
                "complexity": self._assess_complexity(query),
                "legal_entities": legal_entities,
                "requires_hyde": self._should_use_hyde(query, legal_entities),
                "has_context": conversation_context is not None,
                "intent": self._extract_intent(query)
            }
            
            if conversation_context:
                analysis["context_topics"] = conversation_context.get("topics", [])
                analysis["user_intent"] = conversation_context.get("user_intent", "")
            
            state.query_analysis = analysis
            logger.info(f"Enhanced query analysis: {analysis['type']} - {analysis['complexity']}")
            return state
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state.query_analysis = {
                "type": "legal_question", 
                "complexity": "medium",
                "intent": "general_inquiry"
            }
            return state
    
    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal entities from query."""
        entities = {
            "laws": [],
            "articles": [],
            "institutions": [],
            "legal_terms": []
        }
        
        query_lower = query.lower()
        
        # Extract law references (UU, Undang-Undang, dll)
        law_patterns = [
            r'uu\s+(?:no\.|nomor\s+)?(\d+)\s+tahun\s+(\d{4})',
            r'undang-undang\s+(?:no\.|nomor\s+)?(\d+)\s+tahun\s+(\d{4})',
            r'peraturan\s+pemerintah\s+(?:no\.|nomor\s+)?(\d+)\s+tahun\s+(\d{4})',
            r'uud\s+(\d{4})'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, query_lower)
            entities["laws"].extend([f"{m[0]} tahun {m[1]}" if len(m) > 1 else m[0] for m in matches])
        
        # Extract article references
        article_patterns = [
            r'pasal\s+(\d+)',
            r'ayat\s+(\d+)',
            r'huruf\s+([a-z])',
            r'angka\s+(\d+)'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, query_lower)
            entities["articles"].extend(matches)
        
        # Extract institutions
        institutions = [
            "kpk", "komisi pemberantasan korupsi", "kepolisian", "kejaksaan",
            "mahkamah agung", "ma", "mahkamah konstitusi", "mk", "dpr",
            "mpr", "dpd", "bpk", "ombudsman", "komnas ham"
        ]
        
        for institution in institutions:
            if institution in query_lower:
                entities["institutions"].append(institution)
        
        # Extract common legal terms
        legal_terms = [
            "korupsi", "pidana", "perdata", "tata usaha negara", "administrasi",
            "hukum acara", "hukum materiil", "sanksi", "denda", "penjara",
            "rehabilitasi", "kompensasi", "ganti rugi"
        ]
        
        for term in legal_terms:
            if term in query_lower:  # FIXED: Changed from content_lower to query_lower
                entities["legal_terms"].append(term)
        
        return entities
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of legal query."""
        query_lower = query.lower()
        
        # Check for specific query types
        if any(word in query_lower for word in ["apa itu", "definisi", "pengertian", "arti"]):
            return "definition_query"
        elif any(word in query_lower for word in ["berapa", "jumlah", "besaran", "tarif"]):
            return "quantitative_query"
        elif any(word in query_lower for word in ["jelaskan", "uraikan", "paparkan", "rincian"]):
            return "explanation_query"
        elif any(word in query_lower for word in ["bagaimana", "cara", "prosedur"]):
            return "procedure_query"  # FIXED: Added procedure_query classification
        else:
            return "general_inquiry"
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        legal_entity_count = sum(len(entities) for entities in self._extract_legal_entities(query).values())
        
        if word_count < 5 and legal_entity_count < 2:
            return "simple"
        elif word_count < 15 and legal_entity_count < 5:
            return "medium"
        else:
            return "complex"
    
    def _should_use_hyde(self, query: str, legal_entities: Dict[str, List[str]]) -> bool:
        """Determine if HYDE should be used based on query characteristics."""
        # Use HYDE for complex queries with multiple legal entities
        entity_count = sum(len(entities) for entities in legal_entities.values())
        word_count = len(query.split())
        
        # HYDE is beneficial for:
        # 1. Complex queries (many words)
        # 2. Queries with few or no specific legal references
        # 3. General explanations
        return (word_count > 8 and entity_count < 3) or ("jelaskan" in query.lower())
    
    def _extract_intent(self, query: str) -> str:
        """Extract user intent from query - returns valid enum values."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["apa", "definisi", "pengertian"]):
            return "legal_question"
        elif any(word in query_lower for word in ["bagaimana", "cara", "prosedur"]):
            return "legal_question"
        elif any(word in query_lower for word in ["jelaskan", "uraikan", "rincian"]):
            return "legal_question"
        elif any(word in query_lower for word in ["siapa", "lembaga"]):
            return "general_info"
        elif any(word in query_lower for word in ["uu", "undang-undang", "peraturan", "pasal"]):
            return "regulation_lookup"
        elif any(word in query_lower for word in ["dokumen", "file", "teks"]):
            return "document_search"
        else:
            return "legal_question"  # Default to legal_question instead of general_inquiry

    def _map_analysis_intent_to_response_intent(self, analysis_intent: str) -> str:
        """Map internal analysis intent to QueryResponse enum values."""
        intent_mapping = {
            "seek_definition": "legal_question",
            "seek_procedure": "legal_question", 
            "seek_explanation": "legal_question",
            "seek_entity_info": "general_info",
            "general_inquiry": "legal_question",  # Map general_inquiry to legal_question
            "document_lookup": "document_search",
            "regulation_reference": "regulation_lookup"
        }
        
        return intent_mapping.get(analysis_intent, "legal_question")
    
    async def smart_retrieve_documents(self, state: AgentState) -> AgentState:
        """Enhanced document retrieval with multiple strategies."""
        try:
            query = state.query
            analysis = state.query_analysis or {}
            conversation_context = state.conversation_context
            
            # Strategy 1: Direct retrieval with original query
            direct_docs = await self._direct_retrieval(query, analysis)
            
            # Strategy 2: HYDE-enhanced retrieval (if beneficial)
            hyde_docs = []
            hyde_result = None
            if analysis.get("requires_hyde", False) and settings.hyde_enabled:
                hyde_query_result = await self._hyde_retrieval(query, conversation_context)
                if hyde_query_result:
                    # Extract the HydeQuery object and documents separately
                    hyde_result = hyde_query_result.get("hyde_query")
                    hyde_docs = hyde_query_result.get("retrieved_docs", [])
                    
                    # Convert HydeQuery object to dict format that AgentState expects
                    if hyde_result:
                        state.hyde_query = {
                            "original_query": hyde_result.original_query,
                            "enhanced_query": hyde_result.enhanced_query,
                            "hypothetical_documents": hyde_result.hypothetical_documents,
                            "confidence_score": hyde_result.confidence_score
                        }
            
            # Strategy 3: Entity-based retrieval
            entity_docs = await self._entity_based_retrieval(query, analysis.get("legal_entities", {}))
            
            # Combine and rank documents
            all_docs = self._combine_and_rank_documents(direct_docs, hyde_docs, entity_docs)
            
            # Apply relevance filtering
            filtered_docs = self._filter_by_relevance(all_docs, query, threshold=0.3)
            
            state.retrieved_docs = filtered_docs[:settings.retrieval_k]
            
            logger.info(f"Smart retrieval: {len(direct_docs)} direct, {len(hyde_docs)} HYDE, "
                    f"{len(entity_docs)} entity-based -> {len(filtered_docs)} final")
            
            return state
        
        except Exception as e:
            logger.error(f"Smart document retrieval failed: {e}")
            state.retrieved_docs = []
            return state
    
    async def _direct_retrieval(self, query: str, analysis: Dict[str, Any]) -> List[RetrievedDocument]:
        """Direct retrieval using original query."""
        try:
            # Enhance query based on analysis
            enhanced_query = query
            
            # Add legal context if query type suggests it
            if analysis.get("type") == "explanation_query":
                enhanced_query += " penjelasan detail ketentuan hukum"
            elif analysis.get("type") == "procedure_query":
                enhanced_query += " prosedur tata cara"
            
            docs = await vector_store_service.similarity_search_optimized(
                enhanced_query,
                k=settings.retrieval_k * 2  # Get more docs for ranking
            )
            
            return docs
        
        except Exception as e:
            logger.error(f"Direct retrieval failed: {e}")
            return []
    
    async def _hyde_retrieval(self, query: str, conversation_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """HYDE-enhanced retrieval."""
        try:
            # Generate HYDE query with context
            context_info = ""
            if conversation_context:
                context_info = f" Konteks: {conversation_context.get('summary', '')}"
            
            enhanced_query = query + context_info
            hyde_result = await hyde_service.generate_hypothetical_documents(enhanced_query)
            
            if hyde_result and hyde_result.enhanced_query:
                # Retrieve using enhanced query
                docs = await vector_store_service.similarity_search_optimized(
                    hyde_result.enhanced_query,
                    k=settings.retrieval_k
                )
                
                return {
                    "hyde_query": hyde_result,
                    "retrieved_docs": docs
                }
            
            return None
        
        except Exception as e:
            logger.error(f"HYDE retrieval failed: {e}")
            return None
    
    async def _entity_based_retrieval(self, query: str, legal_entities: Dict[str, List[str]]) -> List[RetrievedDocument]:
        """Retrieval based on extracted legal entities."""
        try:
            if not any(legal_entities.values()):
                return []
            
            # Build entity-based query
            entity_terms = []
            
            # Add law references
            for law in legal_entities.get("laws", []):
                entity_terms.append(f"UU {law}")
            
            # Add articles
            for article in legal_entities.get("articles", []):
                entity_terms.append(f"pasal {article}")
            
            # Add institutions
            entity_terms.extend(legal_entities.get("institutions", []))
            
            # Add legal terms
            entity_terms.extend(legal_entities.get("legal_terms", []))
            
            if not entity_terms:
                return []
            
            # Create entity-focused query
            entity_query = " ".join(entity_terms[:5])  # Top 5 most relevant terms
            
            docs = await vector_store_service.similarity_search_optimized(
                entity_query,
                k=settings.retrieval_k
            )
            
            return docs
        
        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return []
    
    def _combine_and_rank_documents(self, *doc_lists) -> List[RetrievedDocument]:
        """Combine multiple document lists and rank by relevance."""
        all_docs = []
        seen_contents = set()
        
        # Assign bonus scores based on retrieval method
        method_bonuses = [1.0, 0.8, 0.6]  # Direct, HYDE, Entity-based
        
        for i, doc_list in enumerate(doc_lists):
            bonus = method_bonuses[i] if i < len(method_bonuses) else 0.4
            
            for doc in doc_list:
                # Avoid duplicates based on content
                content_hash = hashlib.md5(doc.content[:500].encode()).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    
                    # Apply bonus to relevance score
                    doc.relevance_score = (doc.relevance_score or 0.5) * bonus
                    all_docs.append(doc)
        
        # Sort by relevance score
        all_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_docs
    
    def _filter_by_relevance(self, docs: List[RetrievedDocument], query: str, threshold: float = 0.3) -> List[RetrievedDocument]:
        """Filter documents by relevance threshold."""
        filtered_docs = []
        
        for doc in docs:
            # Basic relevance check
            if doc.relevance_score and doc.relevance_score >= threshold:
                filtered_docs.append(doc)
            # Content-based relevance for docs without scores
            elif self._is_content_relevant(doc.content, query):
                doc.relevance_score = 0.5  # Assign default relevance
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _is_content_relevant(self, content: str, query: str) -> bool:
        """Check if content is relevant to query using keyword matching."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Extract key terms from query
        query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term in content_lower)
        relevance_ratio = matches / len(query_terms) if query_terms else 0
        
        return relevance_ratio >= 0.3  # At least 30% term overlap
    
    async def generate_answer(self, state: AgentState) -> AgentState:
        """Generate comprehensive legal answer with enhanced prompting."""
        try:
            query = state.query
            retrieved_docs = state.retrieved_docs or []
            conversation_context = state.conversation_context
            analysis = state.query_analysis or {}
            iteration = state.iteration_count
            
            # Prepare enhanced document context
            document_context = self._prepare_enhanced_context(retrieved_docs, query, analysis)
            
            # Generate answer with optimized prompting
            answer = await self._generate_comprehensive_legal_answer(
                query, document_context, conversation_context, analysis
            )
            
            if iteration == 0:
                state.initial_answer = answer
            else:
                state.reviewed_answer = answer
            
            state.generated_answer = answer
            logger.info("Generated comprehensive legal answer")
            
            return state
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            state.generated_answer = "Maaf, tidak dapat menghasilkan jawaban yang memadai."
            return state
    
    async def _generate_context_summary(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Generate context summary from chat messages."""
        try:
            if not messages:
                return None
            
            # Extract conversation content
            conversation_text = "\n".join([
                f"{msg.role}: {msg.content}" for msg in messages[-10:]  # Last 10 messages
            ])
            
            # Generate summary using LLM
            summary_prompt = f"""
            Analisis percakapan berikut dan buat ringkasan konteks dalam bahasa Indonesia:

            Percakapan:
            {conversation_text}

            Berikan ringkasan dalam format JSON dengan field:
            - summary: ringkasan singkat percakapan
            - topics: list topik utama yang dibahas
            - user_intent: intensi pengguna secara keseluruhan
            """
            
            # This would need to be implemented with your LLM service
            # For now, return a basic summary
            topics = []
            user_intent = "mencari informasi hukum"
            
            # Extract topics from message content
            all_content = " ".join([msg.content for msg in messages])
            if "kpk" in all_content.lower():
                topics.append("KPK")
            if "korupsi" in all_content.lower():
                topics.append("korupsi")
            if "penyidikan" in all_content.lower():
                topics.append("penyidikan")
            
            return {
                "summary": f"Diskusi tentang {', '.join(topics[:3]) if topics else 'hukum Indonesia'}",
                "topics": topics,
                "user_intent": user_intent
            }
        
        except Exception as e:
            logger.error(f"Context summary generation failed: {e}")
            return None
    
    def _prepare_enhanced_context(self, docs: List[RetrievedDocument], query: str, analysis: Dict[str, Any]) -> str:
        """Prepare enhanced document context for answer generation."""
        if not docs:
            return "Tidak ada dokumen relevan ditemukan."
        
        context_parts = []
        
        for i, doc in enumerate(docs[:5]):  # Top 5 documents
            doc_text = doc.content[:1500]  # Limit document length
            relevance = doc.relevance_score or 0.0
            
            context_parts.append(
                f"Dokumen {i+1} (Relevansi: {relevance:.2f}):\n{doc_text}\n"
            )
        
        return "\n".join(context_parts)
    
    async def _generate_comprehensive_legal_answer(self, query: str, document_context: str, 
                                                 conversation_context: Optional[Dict[str, Any]], 
                                                 analysis: Dict[str, Any]) -> str:
        """Generate comprehensive legal answer using LLM."""
        try:
            # Build context-aware prompt
            context_info = ""
            if conversation_context:
                context_info = f"Konteks percakapan: {conversation_context.get('summary', '')}\n"
            
            query_type = analysis.get("type", "general_inquiry")
            complexity = analysis.get("complexity", "medium")
            
            prompt = f"""
            Anda adalah asisten hukum Indonesia yang ahli. Jawab pertanyaan berikut berdasarkan dokumen yang tersedia.

            {context_info}
            Pertanyaan: {query}
            Jenis pertanyaan: {query_type}
            Kompleksitas: {complexity}

            Dokumen referensi:
            {document_context}

            Berikan jawaban yang:
            1. Komprehensif dan akurat berdasarkan dokumen
            2. Menggunakan bahasa Indonesia yang jelas
            3. Menyertakan referensi pasal/undang-undang jika relevan
            4. Memberikan penjelasan praktis jika memungkinkan
            5. Mempertimbangkan konteks percakapan sebelumnya

            Jawaban:
            """
            
            # Generate answer using LLM
            response = await self.llm.ainvoke(prompt)
            
            return response.content if hasattr(response, 'content') else str(response)
        
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return "Maaf, terjadi kesalahan dalam menghasilkan jawaban."
    
    async def review_answer(self, state: AgentState) -> AgentState:
        """Review and improve the generated answer."""
        try:
            answer = state.generated_answer
            if not answer:
                return state
            
            # Simple review - check answer quality
            quality_score = self._assess_answer_quality(answer, state.query)
            
            state.review_result = {
                "quality_score": quality_score,
                "needs_improvement": quality_score < 7.0,
                "feedback": "Jawaban memadai" if quality_score >= 7.0 else "Perlu perbaikan"
            }
            
            logger.info(f"Enhanced review completed - Score: {quality_score}")
            return state
        
        except Exception as e:
            logger.error(f"Answer review failed: {e}")
            state.review_result = {"quality_score": 5.0, "needs_improvement": True}
            return state
    
    def _assess_answer_quality(self, answer: str, query: str) -> float:
        """Assess the quality of generated answer."""
        score = 5.0  # Base score
        
        # Length check
        if len(answer) > 100:
            score += 1.0
        if len(answer) > 300:
            score += 1.0
        
        # Query term coverage
        query_terms = query.lower().split()
        answer_lower = answer.lower()
        covered_terms = sum(1 for term in query_terms if term in answer_lower)
        coverage_ratio = covered_terms / len(query_terms) if query_terms else 0
        score += coverage_ratio * 2.0
        
        # Indonesian legal context
        if any(term in answer_lower for term in ["pasal", "undang-undang", "peraturan"]):
            score += 1.0
        
        return min(score, 10.0)
    
    async def quality_control(self, state: AgentState) -> AgentState:
        """Quality control for the generated answer."""
        try:
            review_result = state.review_result or {}
            quality_score = review_result.get("quality_score", 5.0)
            
            # Determine if answer should be accepted or improved
            if quality_score >= 7.0 and state.iteration_count < 2:
                state.should_continue = False
                state.update_context = True
                state.final_answer = state.generated_answer
                state.confidence_score = min(quality_score / 10.0, 1.0)
                state.sources = state.retrieved_docs
                logger.info(f"Answer accepted - Score: {quality_score}, Confidence: {state.confidence_score:.2f}")
            else:
                state.should_continue = True if state.iteration_count < 2 else False
                state.update_context = not state.should_continue
                if not state.should_continue:
                    state.final_answer = state.generated_answer or "Jawaban tidak dapat dihasilkan dengan baik."
                    state.confidence_score = max(quality_score / 10.0, 0.1)
                    state.sources = state.retrieved_docs
                
                logger.info(f"Answer needs improvement - Score: {quality_score}, Continue: {state.should_continue}")
            
            return state
        
        except Exception as e:
            logger.error(f"Quality control failed: {e}")
            state.should_continue = False
            state.update_context = True
            state.final_answer = state.generated_answer or "Terjadi kesalahan dalam kontrol kualitas."
            return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine workflow continuation."""
        if state.should_continue:
            state.iteration_count += 1
            return "continue"
        elif state.update_context:
            return "update_context"
        else:
            return "update_context"  # Default endpoint
    
    def _prepare_context(self, retrieved_docs: List[RetrievedDocument]) -> str:
        """Prepare context from retrieved documents."""
        if not retrieved_docs:
            return "Tidak ada dokumen relevan ditemukan."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            content = self._extract_document_content(doc)
            metadata = self._extract_document_metadata(doc)
            
            source_info = self._format_source_info(metadata)
            
            context_part = f"[Dokumen {i}]\n"
            if source_info:
                context_part += f"{source_info}\n"
            context_part += f"{content}\n"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    # Context management methods
    def get_chat_history(self, chat_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get chat history for a specific chat."""
        messages = self.context_manager.get_recent_messages(chat_id, limit)
        return [msg.to_dict() for msg in messages]
    
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context statistics including frontend usage."""
        try:
            internal_stats = self.context_manager.get_chat_stats()
            
            return {
                "internal_context_manager": internal_stats,
                "frontend_integration": {
                    "supports_frontend_context": True,
                    "context_source": "frontend_chat_history_preferred",
                    "fallback": "internal_context_manager"
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get context stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


    def clear_chat_context(self, chat_id: str) -> None:
        """Clear chat context for a specific session."""
        try:
            self.context_manager.clear_chat_history(chat_id)
            logger.info(f"Cleared context for chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to clear context for chat {chat_id}: {e}")


    async def _generate_context_summary(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Generate a summary of conversation context."""
        
        # Prepare conversation history text (exclude current message)
        conversation_text = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in messages[:-1]  # Exclude current message
        ])
        
        system_prompt = """Analisis percakapan untuk konteks hukum yang relevan.

    Tugas:
    1. Ringkas topik utama dalam 1-2 kalimat
    2. Identifikasi intent pengguna 
    3. Tentukan kata kunci hukum penting
    4. Berikan konteks untuk pertanyaan lanjutan

    Format JSON:
    {
        "summary": "ringkasan singkat",
        "user_intent": "intent utama", 
        "topics": ["topik1", "topik2"],
        "context_relevance": "relevansi konteks",
        "legal_context": "konteks hukum spesifik",
        "conversation_flow": "alur percakapan"
    }"""
        
        user_prompt = f"""Analisis percakapan hukum:

    {conversation_text}

    CURRENT MESSAGE COUNT: {len(messages)}
    SOURCE: Frontend chat history

    Berikan analisis dalam format JSON."""
        
        try:
            messages_for_llm = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(messages_for_llm)
            )
            
            # Parse JSON response
            try:
                context_data = json.loads(response.strip())
                
                # Add metadata about source
                context_data["source"] = "frontend_chat_history"
                context_data["message_count"] = len(messages)
                context_data["generated_at"] = datetime.now().isoformat()
                
                return context_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "summary": "Percakapan tentang topik hukum",
                    "user_intent": "Mencari informasi hukum",
                    "topics": ["hukum"],
                    "context_relevance": "Konteks dari frontend chat history",
                    "legal_context": "Topik hukum umum",
                    "conversation_flow": "Dialog berkelanjutan",
                    "source": "frontend_chat_history_fallback",
                    "message_count": len(messages),
                    "generated_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Context summary generation failed: {e}")
            return {
                "summary": "Tidak dapat menganalisis konteks",
                "user_intent": "Tidak diketahui",
                "topics": [],
                "context_relevance": "Konteks tidak tersedia",
                "legal_context": "Tidak dapat dianalisis",
                "conversation_flow": "Error dalam analisis",
                "source": "error",
                "message_count": len(messages),
                "generated_at": datetime.now().isoformat(),
                "error": str(e)
            }


# Global RAG agents instance
rag_agents = LegalRAGAgents()