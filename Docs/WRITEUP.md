# Indonesian Legal RAG System Documentation

## Overview
This system is a sophisticated **Retrieval-Augmented Generation (RAG)** application designed specifically for Indonesian legal documents. It combines document processing, vector storage, conversational AI, Advance Context Analyzer and advanced retrieval techniques to provide accurate legal information.

[Watch the video]([https://github.com/USER/REPO/raw/main/path/to/video.mp4](https://drive.google.com/file/d/1OLYzd2LCwjkSy39JVjcpEK4dBrkqvrjE/view?usp=sharing))

## Desain System

### Document Processor to DB Vector
![Document Process System](https://raw.githubusercontent.com/Riandra4trend/gani_ai/main/desain%20system/Document%20Process%20System.jpg)

### RAG Agents
![Document Process System](https://github.com/Riandra4trend/gani_ai/blob/main/desain%20system/RAG%20Agent%20System.jpg)


## System Architecture

### Core Components
1. **Document Processor** - Handles PDF processing and OCR
2. **Vector Store Service** - Manages document embeddings and similarity search
3. **HYDE Service** - Enhances queries using hypothetical document generation
4. **Context Conversation Inmemory** - Give Context Conversation Sumamary from Chat histroy from website storage with Chat ID
5. **RAG Agents** - Multi-agent workflow for intelligent query processing
6. **Data Models** - Pydantic models for type safety and validation

---

## 1. Document Processor (`document_processor.py`)

### Purpose
Processes Indonesian legal PDF documents, extracts text content, and prepares them for vector storage with Indonesian NLP processing.

### Key Features

#### PDF Text Extraction
```python
async def _extract_pdf_content(self, pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)  # PyMuPDF for PDF reading
    text_content = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        
        # If no text found, use OCR
        if not page_text.strip():
            page_text = await self._ocr_page(page)
```
- **PyMuPDF (fitz)** extracts text from PDF pages
- **OCR fallback** using Tesseract for scanned documents
- **Page-by-page processing** ensures complete document coverage

#### Indonesian NLP Pipeline
```python
def _setup_nlp_components(self):
    # Indonesian stemmer - reduces words to root form
    stemmer_factory = StemmerFactory()
    self.stemmer = stemmer_factory.create_stemmer()
    
    # Indonesian stopword remover
    stopword_factory = StopWordRemoverFactory()
    self.stopword_remover = stopword_factory.create_stop_word_remover()
```
- **Sastrawi Stemmer** - Converts Indonesian words to root forms (e.g., "mengatur" → "atur")
- **Stopword Removal** - Removes common Indonesian words that don't add meaning
- **Custom Legal Stopwords** - Removes legal document formatting words

#### Text Preprocessing
```python
def _preprocess_text(self, text: str) -> str:
    text = text.lower()  # Normalize case
    
    if self.stopword_remover:
        text = self.stopword_remover.remove(text)  # Remove stopwords
    
    tokens = word_tokenize(text, language='indonesian')  # Tokenize
    
    if self.stemmer:
        tokens = [self.stemmer.stem(token) for token in tokens if len(token) > 2]
```
- **Tokenization** breaks text into individual words
- **Stemming** normalizes word variations
- **Filtering** removes short and irrelevant tokens

#### Smart File Tracking
```python
def _is_file_processed(self, file_path: Path) -> bool:
    current_hash = self._get_file_hash(file_path)
    return stored_info.get('hash') == current_hash
```
- **File hashing** detects document changes
- **Incremental processing** only processes new/modified files
- **JSON tracking database** maintains processing history

---

## 2. Vector Store Service (`vector_store.py`)

### Purpose
Manages vector embeddings storage and similarity search using ChromaDB and Ollama embeddings.

### Key Components

#### Ollama Embeddings Integration
```python
def _initialize_embeddings(self):
    self.embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model=settings.embedding_model
    )
```
- **Local Ollama server** provides embedding generation
- **mxbai-embed-large model** creates high-quality vector representations
- **Self-hosted solution** ensures data privacy

#### ChromaDB Vector Storage
```python
self.vector_store = Chroma(
    collection_name=settings.collection_name,
    embedding_function=self.embeddings,
    persist_directory=persist_directory
)
```
- **ChromaDB** stores document vectors locally
- **Persistent storage** maintains embeddings across restarts
- **Collection-based organization** for different document types

#### Optimized Similarity Search
```python
async def similarity_search_optimized(self, query: str, k: Optional[int] = None) -> List[RetrievedDocument]:
    # Direct ChromaDB similarity search
    docs = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: self.vector_store.similarity_search(query, **search_args)
    )
```
- **Asynchronous execution** prevents blocking during search
- **Error handling** with multiple search strategies
- **Relevance scoring** ranks documents by similarity

---

## 3. HYDE Service (`hyde_service.py`)
# Analisis Sistem HYDE Legal RAG

## HYDE Theory Overview

**HYDE (Hypothetical Document Embeddings)** adalah teknik query expansion yang menggunakan LLM untuk generate dokumen hipotetis yang menjawab query, kemudian menggunakan embedding dokumen hipotetis tersebut untuk mencari dokumen relevan di vector store.

## Function-by-Function Analysis

### 1. **Legal Query Analyzer** (`_analyze_legal_query`)

**Input**: 
- `query: str` - Raw user query dalam Bahasa Indonesia

**Process**:
- **Regex Pattern Matching** untuk legal entities:
  ```python
  "law_reference": r"(?:uu|undang-undang)\s+(?:no\.|nomor\s+)?(\d+)\s+tahun\s+(\d{4})"
  "article": r"pasal\s+(\d+)"
  ```
- **Institution Detection** via keyword matching
- **Legal Concept Extraction** dari predefined vocabulary
- **Query Type Classification**: explanation/definition/procedure/specific_legal
- **Specificity Assessment**: high/medium/low berdasarkan entity count

**Output**: 
```python
{
    "laws": ["30 tahun 2002"],
    "articles": ["6", "12"], 
    "institutions": ["kpk"],
    "legal_concepts": ["korupsi", "pidana"],
    "query_type": "explanation",
    "specificity": "high"
}
```

**Theory**: **Domain-Specific NER** untuk legal entities Indonesia + **Query Understanding** untuk strategy selection

---

### 2. **Enhanced Hypothetical Document Generator** (`_generate_enhanced_hypothetical_docs`)

**Input**: 
- `query: str` + `legal_analysis: Dict` + `query_context: Optional[Dict]`

**Process**:
- **Type-Specific Prompt Engineering**:
  - **Explanation queries**: Focus pada penjelasan komprehensif
  - **Definition queries**: Focus pada definisi resmi + contoh
  - **Procedure queries**: Focus pada langkah-langkah sistematis
- **Legal Context Integration** dari chat history
- **LLM Generation** dengan temperature 0.1 untuk consistency
- **Document Parsing** dengan format "DOKUMEN:" delimiter
- **Quality Validation** per document

**Output**: 
```python
List[str] = [
    "Menurut UU No. 30 Tahun 2002, KPK memiliki kewenangan...",
    "Berdasarkan Pasal 6 UU No. 31 Tahun 1999...",
    "Dalam konteks pemberantasan korupsi..."
]
```

**Theory**: **Conditional Text Generation** + **Template-based Prompting** + **Legal Domain Knowledge Injection**

---

### 3. **Multi-Strategy Query Enhancement** (`_generate_multiple_enhanced_queries`)

**Input**: 
- `original_query: str` + `hypothetical_docs: List[str]` + `legal_analysis: Dict`

**Process**:
- **Strategy 1 - Keyword Enhancement**: Extract legal keywords dari hypothetical docs
- **Strategy 2 - Entity Enhancement**: Combine dengan extracted legal entities
- **Strategy 3 - Context Enhancement**: LLM-generated query enhancement
- **Deduplication** dan validation

**Output**: 
```python
List[str] = [
    "kewenangan KPK pemberantasan korupsi",  # Original
    "kpk korupsi pidana undang-undang pasal sanksi",  # Keyword-enhanced
    "UU 30 tahun 2002 pasal 6 kpk",  # Entity-enhanced  
    "kewenangan komisi pemberantasan korupsi berdasarkan ketentuan hukum"  # Context-enhanced
]
```

**Theory**: **Query Expansion** + **Multi-Strategy Optimization** + **Ensemble Query Generation**

---

### 4. **Best Query Selector** (`_select_best_enhanced_query`)

**Input**: 
- `enhanced_queries: List[str]` + `legal_analysis: Dict`

**Process**:
- **Multi-Criteria Scoring**:
  - Length optimization (8-15 words = +2 points)
  - Legal entity presence (+3 points per law, +2 per article)
  - Legal concept coverage (+1 point per concept)
  - Over-stuffing penalty (-2 points if >20 words)
- **Ranking & Selection** berdasarkan highest score

**Output**: 
- `best_enhanced_query: str` dengan highest relevance score

**Theory**: **Multi-Objective Optimization** + **Heuristic Scoring** + **Query Quality Assessment**

---

### 5. **Confidence Score Calculator** (`_calculate_enhanced_confidence_score`)

**Input**: 
- `query: str` + `hypothetical_docs: List[str]` + `legal_analysis: Dict`

**Process**:
- **Weighted Factor Calculation**:
  - **Document Quality** (30%): avg_length/150 
  - **Entity Specificity** (25%): entity_count scoring
  - **Concept Coverage** (20%): concept_count/3
  - **Query Complexity** (15%): optimal word_count range
  - **Document Count** (10%): doc_count/3
- **Weighted Sum** dengan normalization 0.1-1.0

**Output**: 
- `confidence_score: float` (0.1 - 1.0)

**Theory**: **Multi-Factor Confidence Estimation** + **Weighted Scoring System** + **Quality Metrics Aggregation**

---

### 6. **Document Validator** (`_validate_hypothetical_document`)

**Input**: 
- `doc: str` + `legal_analysis: Dict`

**Process**:
- **Length Validation**: minimum 50 characters
- **Legal Indicator Check**: minimum 2 legal terms required
- **Entity Consistency**: validate entity mentions vs query analysis
- **High Specificity Validation**: strict checking untuk query dengan entitas spesifik

**Output**: 
- `bool` - valid/invalid document

**Theory**: **Content Quality Assurance** + **Domain-Specific Validation** + **Consistency Checking**

---

### 7. **Adaptive HYDE** (`adaptive_hyde`)

**Input**: 
- `query: str` + `retrieval_results: List[Any]` + `threshold: float = 0.6`

**Process**:
- **Retrieval Quality Assessment**: calculate average relevance_score
- **Adaptive Decision**: apply HYDE only if quality < threshold
- **Resource Optimization**: skip HYDE untuk good initial retrieval

**Output**: 
- `Optional[HydeQuery]` - HYDE result atau None

**Theory**: **Adaptive Processing** + **Resource Optimization** + **Quality-based Decision Making**

---

### 8. **Multi-Perspective HYDE** (`multi_perspective_hyde`)

**Input**: 
- `query: str`

**Process**:
- **Multiple Legal Perspectives**:
  - Hukum Pidana (sanksi & criminal aspects)
  - Hukum Administrasi (prosedur & tata cara)  
  - Hukum Tata Negara (kelembagaan & kewenangan)
  - Implementasi Praktis (penerapan real-world)
- **Perspective-Specific Generation** untuk setiap sudut pandang
- **Context Labeling** dengan [Perspective] tags
- **Comprehensive Enhancement** dari multiple viewpoints

**Output**: 
```python
HydeQuery(
    hypothetical_documents=[
        "[Hukum Pidana] Berdasarkan UU Tipikor, sanksi...",
        "[Hukum Administrasi] Prosedur penyidikan KPK...", 
        "[Hukum Tata Negara] Kewenangan kelembagaan...",
        "[Implementasi Praktis] Dalam praktek..."
    ]
)
```

**Theory**: **Multi-Perspective Analysis** + **Comprehensive Coverage** + **Legal Domain Decomposition**

---

## Best Practices Implementation

### 1. **Intelligent HYDE Activation**
```python
# HYDE digunakan hanya ketika:
(word_count > 8 AND entity_count < 3) OR "jelaskan" in query
# → Complex queries tanpa specific entities
```

**Rationale**: HYDE paling efektif untuk query abstrak/general, tidak perlu untuk query dengan entitas spesifik.

### 2. **Legal Domain Specialization**
- **Custom Regex Patterns** untuk legal entities Indonesia
- **Legal Vocabulary Database** untuk concept extraction  
- **Type-Specific Prompting** sesuai jenis pertanyaan hukum
- **Indonesian Legal Format Recognition** (UU, Pasal, Ayat, dll)

### 3. **Quality Assurance Pipeline**
```python
Generation → Parsing → Validation → Fallback (if needed)
```
- **Multi-Level Validation**: length, content, entity consistency
- **Graceful Degradation** dengan template-based fallbacks
- **Retry Mechanism** dengan exponential backoff

### 4. **Multi-Strategy Query Enhancement**
- **Keyword-based**: Extract terms dari hypothetical docs
- **Entity-based**: Combine dengan legal entities
- **Context-based**: LLM-generated enhancement
- **Ensemble Selection**: Score-based best query selection

### 5. **Confidence Modeling**
- **Multi-Factor Scoring** dengan weighted components
- **Quality Indicators**: doc quality, specificity, coverage, complexity
- **Uncertainty Quantification** untuk downstream decision making

### 6. **Resource Optimization**
- **Adaptive Processing**: HYDE hanya jika diperlukan
- **Temperature Control**: 0.1 untuk consistency vs creativity
- **Bounded Generation**: limit jumlah docs dan iterations
- **Async Processing** untuk non-blocking operations

### 7. **Error Resilience**
- **Comprehensive Error Handling** di setiap level
- **Fallback Mechanisms**: template-based docs jika generation gagal
- **Retry Strategy** dengan tenacity decorator
- **Graceful Degradation**: return original query jika enhancement gagal

## HYDE Effectiveness Factors

### **High Effectiveness** (confidence > 0.8):
- Query complex tanpa entitas spesifik
- Multiple hypothetical docs generated successfully
- High legal concept coverage
- Optimal query length (8-15 words)

### **Medium Effectiveness** (confidence 0.5-0.8):
- Some legal entities present
- Partial hypothetical doc generation
- Moderate concept coverage

### **Low Effectiveness** (confidence < 0.5):
- Query sangat spesifik dengan banyak entitas
- Generation gagal atau doc quality rendah
- Over-specific atau under-specific queries

---

## Advanced HYDE Features

### **Adaptive HYDE**
```python
if initial_retrieval_quality < 0.6:
    apply_hyde()  # Only when needed
else:
    skip_hyde()   # Save resources
```

### **Multi-Perspective HYDE**
```python
perspectives = [
    ("Hukum Pidana", "sanksi dan aspek criminal"),
    ("Hukum Administrasi", "prosedur dan tata cara"),
    ("Hukum Tata Negara", "kelembagaan dan kewenangan"),  
    ("Implementasi Praktis", "penerapan real-world")
]
```

**Benefit**: Comprehensive coverage dari berbagai sudut pandang hukum untuk query kompleks.

---

## System Strengths

1. **Domain-Aware HYDE**: Specialized untuk hukum Indonesia
2. **Multi-Strategy Enhancement**: Multiple query enhancement approaches
3. **Quality-Driven Processing**: Validation dan confidence scoring
4. **Resource Optimization**: Adaptive usage berdasarkan need
5. **Error Resilience**: Comprehensive fallback mechanisms
6. **Legal Entity Intelligence**: Sophisticated legal NER
7. **Context Integration**: Conversation-aware enhancement

---

## 4. RAG Agents (`rag_agents.py`)

### Purpose
Orchestrates a multi-agent workflow using **LangGraph** to process legal queries intelligently with conversation context.

## Overview RAG Agent
Legal RAG System adalah sistem multi-agent yang dirancang khusus untuk menjawab pertanyaan hukum Indonesia menggunakan:
- **RAG (Retrieval-Augmented Generation)**: Kombinasi pencarian dokumen + generasi jawaban
- **HYDE (Hypothetical Document Embeddings)**: Teknik untuk meningkatkan relevansi pencarian
- **Context Management**: Pengelolaan konteks percakapan untuk continuity
- **Multi-Agent Workflow**: Beberapa agent yang bekerja secara sequential

### Multi-Agent Workflow

#### LangGraph State Machine
```python
def _build_agent_graph(self):
    workflow = StateGraph(AgentState)
    
    # Add specialized agents
    workflow.add_node("context_analyzer", self.analyze_context)
    workflow.add_node("query_analyzer", self.analyze_query)
    workflow.add_node("smart_retriever", self.smart_retrieve_documents)
    workflow.add_node("answer_generator", self.generate_answer)
    workflow.add_node("document_reviewer", self.review_answer)
    workflow.add_node("quality_controller", self.quality_control)
```
- **State-based workflow** ensures consistent processing
- **Specialized agents** handle specific tasks
- **Conditional branching** adapts to query complexity

## Komponen Utama

### 1. Context Management
```python
class InMemoryContextManager:
    # Menyimpan riwayat chat per session
    chat_histories: Dict[str, deque]  # chat_id -> messages
    conversation_contexts: Dict[str, ConversationContext]  # chat_id -> context summary
```

**Fungsi:**
- Menyimpan chat history per session
- Membuat ringkasan konteks percakapan
- Mengelola frontend chat history integration

### 2. Multi-Agent Workflow (LangGraph)
Workflow terdiri dari 7 agent yang bekerja secara berurutan:

{link flow}

### 3. Smart Retrieval System
Menggunakan 3 strategi retrieval:
- **Direct Retrieval**: Pencarian langsung dengan query asli
- **HYDE Retrieval**: Pencarian dengan hypothetical documents
- **Entity-based Retrieval**: Pencarian berdasarkan entitas hukum

---

## Flow Sistem Lengkap

### Input Process
**Input ada 2 jenis:**

#### A. Query Tanpa Context (Standard)
```python
QueryRequest:
    - query: "Apa itu korupsi menurut UU?"
    - processing_type: "standard"
```

#### B. Query Dengan Context (Conversational)
```python
QueryRequest + Context:
    - query: "Bagaimana sanksi untuk kasus tersebut?"
    - chat_id: "chat_123"
    - chat_history: [
        {"role": "user", "content": "Apa itu korupsi?", "timestamp": "..."},
        {"role": "assistant", "content": "Korupsi adalah...", "timestamp": "..."}
    ]
```

### Processing Flow

#### Stage 1: Context Analyzer
**Input:** AgentState dengan query dan optional chat_history
**Process:**
```python
async def analyze_context(self, state: AgentState):
    # Jika ada frontend_chat_history
    if state.frontend_chat_history:
        # Convert ke internal ChatMessage format
        recent_messages = convert_frontend_to_internal(frontend_chat_history)
        
        # Generate context summary dari percakapan
        context_summary = await self._generate_context_summary(recent_messages)
        state.conversation_context = context_summary
```

**Output:** State dengan conversation_context
```python
conversation_context = {
    "summary": "Diskusi tentang korupsi, KPK",
    "topics": ["korupsi", "KPK", "penyidikan"],
    "user_intent": "mencari informasi sanksi",
    "source": "frontend_chat_history"
}
```

#### Stage 2: Query Analyzer  
**Input:** State dengan query dan context
**Process:**
```python
async def analyze_query(self, state: AgentState):
    # Extract legal entities
    legal_entities = self._extract_legal_entities(query)
    # {
    #     "laws": ["UU 31 tahun 1999"],
    #     "articles": ["pasal 2", "pasal 3"], 
    #     "institutions": ["KPK"],
    #     "legal_terms": ["korupsi", "pidana"]
    # }
    
    analysis = {
        "type": "explanation_query",  # definition_query, procedure_query, etc
        "complexity": "medium",       # simple, medium, complex
        "legal_entities": legal_entities,
        "requires_hyde": True,        # Based on complexity
        "intent": "legal_question"
    }
```

**Output:** State dengan query_analysis

#### Stage 3: Smart Retriever (Multi-Strategy)
**Input:** State dengan analyzed query
**Process:**

##### Strategy 1: Direct Retrieval
```python
async def _direct_retrieval(self, query, analysis):
    enhanced_query = query + " penjelasan detail ketentuan hukum"  # Based on query type
    docs = await vector_store_service.similarity_search_optimized(enhanced_query)
    return docs
```

##### Strategy 2: HYDE Retrieval (Jika diperlukan)
```python
async def _hyde_retrieval(self, query, conversation_context):
    # Add context ke query
    context_info = f" Konteks: {conversation_context.get('summary', '')}"
    enhanced_query = query + context_info
    
    # Generate hypothetical documents
    hyde_result = await hyde_service.generate_hypothetical_documents(enhanced_query)
    # HydeQuery {
    #     original_query: "Bagaimana sanksi korupsi?",
    #     enhanced_query: "sanksi pidana korupsi UU tipikor denda penjara",
    #     hypothetical_documents: ["Sanksi korupsi meliputi...", "Pidana pokok dan tambahan..."],
    #     confidence_score: 0.85
    # }
    
    # Search using enhanced query
    docs = await vector_store_service.similarity_search_optimized(hyde_result.enhanced_query)
    return {"hyde_query": hyde_result, "retrieved_docs": docs}
```

##### Strategy 3: Entity-based Retrieval
```python
async def _entity_based_retrieval(self, query, legal_entities):
    # Build query dari entities
    entity_terms = []
    entity_terms.extend([f"UU {law}" for law in legal_entities["laws"]])
    entity_terms.extend([f"pasal {art}" for art in legal_entities["articles"]])
    entity_terms.extend(legal_entities["institutions"])
    
    entity_query = " ".join(entity_terms[:5])
    docs = await vector_store_service.similarity_search_optimized(entity_query)
    return docs
```

##### Combine & Rank Documents
```python
def _combine_and_rank_documents(self, direct_docs, hyde_docs, entity_docs):
    # Assign bonus scores: Direct=1.0, HYDE=0.8, Entity=0.6
    # Remove duplicates based on content hash
    # Sort by relevance score
    # Filter by threshold (0.3)
    return filtered_ranked_docs
```

**Output:** State dengan retrieved_docs (top documents)

#### Stage 4: Answer Generator
**Input:** State dengan retrieved documents dan context
**Process:**
```python
async def generate_answer(self, state: AgentState):
    # Prepare enhanced context
    document_context = self._prepare_enhanced_context(retrieved_docs, query, analysis)
    
    # Build comprehensive prompt
    context_info = f"Konteks percakapan: {conversation_context.get('summary', '')}"
    
    prompt = f"""
    Anda adalah asisten hukum Indonesia yang ahli.
    
    {context_info}
    Pertanyaan: {query}
    Jenis pertanyaan: {query_type}
    
    Dokumen referensi:
    {document_context}
    
    Berikan jawaban yang:
    1. Komprehensif berdasarkan dokumen
    2. Bahasa Indonesia yang jelas  
    3. Referensi pasal/UU jika relevan
    4. Penjelasan praktis
    5. Mempertimbangkan konteks percakapan
    """
    
    answer = await self.llm.ainvoke(prompt)
```

**Output:** State dengan generated_answer

#### Stage 5: Document Reviewer
**Input:** Generated answer
**Process:**
```python
def _assess_answer_quality(self, answer, query):
    score = 5.0
    
    # Length check
    if len(answer) > 100: score += 1.0
    if len(answer) > 300: score += 1.0
    
    # Query term coverage
    covered_terms = count_matching_terms(query, answer)
    score += (covered_terms / total_query_terms) * 2.0
    
    # Legal context presence
    if has_legal_terms(answer): score += 1.0
    
    return min(score, 10.0)
```

**Output:** State dengan review_result dan quality_score

#### Stage 6: Quality Controller
**Input:** Review result dan quality score
**Process:**
```python
async def quality_control(self, state: AgentState):
    quality_score = state.review_result.get("quality_score", 5.0)
    
    if quality_score >= 7.0 and state.iteration_count < 2:
        # Accept answer
        state.should_continue = False
        state.final_answer = state.generated_answer
        state.confidence_score = quality_score / 10.0
    else:
        # Retry or finalize
        state.should_continue = True if state.iteration_count < 2 else False
```

**Decision Points:**
- `quality_score >= 7.0` → Accept answer → "update_context"
- `quality_score < 7.0 & iteration < 2` → Retry → "continue" 
- `iteration >= 2` → Finalize → "update_context"

#### Stage 7: Context Updater
**Input:** Final state dengan accepted answer
**Process:**
```python
async def update_context(self, state: AgentState):
    # Skip jika temporary chat
    if chat_id.startswith("temp_"): return state
    
    # Generate updated context summary
    recent_messages = get_recent_messages(chat_id)
    context_summary = await self._generate_context_summary(recent_messages)
    
    # Update conversation context
    updated_context = ConversationContext(
        chat_id=chat_id,
        summary=context_summary["summary"],
        topics=context_summary["topics"],
        user_intent=context_summary["user_intent"],
        last_updated=datetime.now(),
        message_count=len(recent_messages)
    )
```

**Output:** Updated conversation context untuk chat berikutnya

---

## Final Output

### QueryResponse Object
```python
QueryResponse(
    query="Bagaimana sanksi korupsi?",
    answer="Sanksi korupsi berdasarkan UU No. 31 Tahun 1999...",
    sources=[
        RetrievedDocument(content="...", source="UU 31/1999", relevance_score=0.89),
        RetrievedDocument(content="...", source="UU 20/2001", relevance_score=0.75)
    ],
    confidence_score=0.85,
    processing_time=2.3,
    hyde_info={
        "original_query": "Bagaimana sanksi korupsi?",
        "enhanced_query": "sanksi pidana korupsi denda penjara UU tipikor",
        "hypothetical_documents": ["Sanksi korupsi meliputi..."],
        "confidence_score": 0.85
    },
    query_intent="legal_question"
)
```

---

## Context Integration Detail

### Frontend Chat History Integration
```python
# Method 1: Process dengan context dari frontend
async def process_query_with_context(self, request, chat_id, chat_history):
    # Load frontend chat history
    self.context_manager.load_chat_history(chat_id, chat_history)
    
    # Initialize state dengan frontend_chat_history
    state = AgentState(
        query=request.query,
        chat_id=chat_id,
        frontend_chat_history=chat_history,  # Key: data dari frontend
        context_source="frontend"
    )
```

### Context Analysis Process
```python
async def analyze_context(self, state: AgentState):
    frontend_chat_history = state.frontend_chat_history
    
    if frontend_chat_history:
        # Convert frontend format ke internal ChatMessage
        recent_messages = []
        for msg_data in frontend_chat_history[-10:]:  # Last 10 messages
            message = ChatMessage(
                id=msg_data["id"],
                content=msg_data["content"], 
                role=msg_data["role"],       # "user" or "assistant"
                timestamp=parse_timestamp(msg_data["timestamp"])
            )
            recent_messages.append(message)
        
        # Generate context summary
        context_summary = await self._generate_context_summary(recent_messages)
        state.conversation_context = context_summary
```

---

## HYDE (Hypothetical Document Embeddings) Detail

### Cara Kerja HYDE
```python
# 1. Input: Original query + context
query = "Bagaimana sanksi korupsi?"
context = "Konteks: Diskusi tentang korupsi, KPK"

# 2. Generate hypothetical documents
hyde_result = await hyde_service.generate_hypothetical_documents(query + context)

# 3. HYDE generates "ideal" documents
hypothetical_docs = [
    "Sanksi korupsi berdasarkan UU No. 31 Tahun 1999 meliputi pidana penjara minimal 4 tahun...",
    "Denda korupsi dapat mencapai Rp 1 miliar atau 20x kerugian negara...",
    "Pidana tambahan berupa pencabutan hak politik dan perampasan aset..."
]

# 4. Create enhanced query from hypothetical docs
enhanced_query = "sanksi pidana korupsi denda penjara UU tipikor pencabutan hak"

# 5. Search menggunakan enhanced query
final_docs = await vector_store_service.similarity_search_optimized(enhanced_query)
```

### Kapan HYDE Digunakan
```python
def _should_use_hyde(self, query, legal_entities):
    entity_count = sum(len(entities) for entities in legal_entities.values())
    word_count = len(query.split())
    
    # HYDE beneficial untuk:
    return (word_count > 8 and entity_count < 3) or ("jelaskan" in query.lower())
    # 1. Query kompleks dengan sedikit entitas spesifik
    # 2. Query eksplanatori (jelaskan, uraikan)
```

---

## Final LLM Input Construction

### Prompt Building untuk LLM Terakhir
```python
async def _generate_comprehensive_legal_answer(self, query, document_context, conversation_context, analysis):
    # 1. Context dari percakapan sebelumnya
    context_info = ""
    if conversation_context:
        context_info = f"Konteks percakapan: {conversation_context.get('summary', '')}\n"
    
    # 2. Query analysis info
    query_type = analysis.get("type", "general_inquiry")     # "explanation_query"
    complexity = analysis.get("complexity", "medium")        # "medium"
    
    # 3. Document context (dari retrieval)
    document_context = """
    Dokumen 1 (Relevansi: 0.89):
    UU No. 31 Tahun 1999 tentang Korupsi - Pasal 2: Setiap orang yang melawan hukum...
    
    Dokumen 2 (Relevansi: 0.75): 
    UU No. 20 Tahun 2001 - Pasal 18: Pidana pokok berupa penjara seumur hidup...
    """
    
    # 4. Final prompt construction
    prompt = f"""
    Anda adalah asisten hukum Indonesia yang ahli. Jawab pertanyaan berikut berdasarkan dokumen yang tersedia.

    {context_info}  // Konteks percakapan: Diskusi tentang korupsi, KPK
    Pertanyaan: {query}  // Bagaimana sanksi korupsi?
    Jenis pertanyaan: {query_type}  // explanation_query
    Kompleksitas: {complexity}  // medium

    Dokumen referensi:
    {document_context}  // Retrieved documents

    Berikan jawaban yang:
    1. Komprehensif dan akurat berdasarkan dokumen
    2. Menggunakan bahasa Indonesia yang jelas
    3. Menyertakan referensi pasal/undang-undang jika relevan
    4. Memberikan penjelasan praktis jika memungkinkan
    5. Mempertimbangkan konteks percakapan sebelumnya

    Jawaban:
    """
```

---

## Case Examples

### Case 1: Query Pertama (Tanpa Context)
**Input:**
```
Query: "Apa itu korupsi menurut hukum Indonesia?"
Context: None
```

**Flow:**
1. **Context Analyzer**: No context → conversation_context = None
2. **Query Analyzer**: 
   - Type: "definition_query"
   - Entities: {"legal_terms": ["korupsi"]}
   - Requires HYDE: True (explanation query)
3. **Smart Retriever**:
   - Direct: Search "apa itu korupsi hukum Indonesia"
   - HYDE: Generate hypothetical → "korupsi definisi UU tipikor melawan hukum"
   - Entity: Search "korupsi"
   - Combine & rank documents
4. **Answer Generator**: Generate comprehensive definition
5. **Quality Control**: Accept answer (score: 8.2)
6. **Context Update**: Create new conversation context

**Output:**
```
Answer: "Korupsi menurut UU No. 31 Tahun 1999 adalah perbuatan melawan hukum dengan tujuan memperkaya diri..."
Confidence: 0.82
Sources: [UU 31/1999 Pasal 2, UU 20/2001, ...]
```

### Case 2: Query Follow-up (Dengan Context)
**Input:**
```
Query: "Bagaimana sanksinya?"
Chat_ID: "chat_123"
Chat_History: [
    {"role": "user", "content": "Apa itu korupsi?"},
    {"role": "assistant", "content": "Korupsi adalah..."}
]
```

**Flow:**
1. **Context Analyzer**: 
   - Load frontend chat history
   - Generate context: "Diskusi tentang korupsi, definisi dan ketentuan"
2. **Query Analyzer**:
   - Type: "procedure_query" 
   - Context-aware: "sanksi" merujuk ke "korupsi" dari context
3. **Smart Retriever**:
   - Direct: "sanksi korupsi pidana denda"
   - HYDE: Enhanced dengan context → "sanksi pidana korupsi UU tipikor penjara denda"
   - Entity: Search based on "korupsi" dari context
4. **Answer Generator**: 
   - Use context: "Berdasarkan pembahasan korupsi sebelumnya..."
   - Generate sanksi explanation
5. **Quality Control**: Accept
6. **Context Update**: Update context dengan topik "sanksi"

**Output:**
```
Answer: "Berdasarkan pembahasan korupsi sebelumnya, sanksi korupsi meliputi: 1. Pidana penjara minimal 4 tahun..."
Confidence: 0.87
Context-aware: true
```

### Case 3: Query Kompleks (Multiple Entities)
**Input:**
```
Query: "Jelaskan proses penyidikan korupsi oleh KPK berdasarkan UU No. 30 Tahun 2002"
```

**Flow:**
1. **Query Analyzer**:
   - Type: "explanation_query"
   - Complexity: "complex"
   - Entities: {
     "laws": ["UU 30 tahun 2002"],
     "institutions": ["KPK"], 
     "legal_terms": ["korupsi", "penyidikan"]
   }
   - Requires HYDE: True
2. **Smart Retriever**:
   - Direct: Original query
   - HYDE: "prosedur penyidikan korupsi KPK tahapan wewenang UU KPK"
   - Entity: "UU 30 tahun 2002 KPK penyidikan korupsi"
3. **Answer Generation**: Comprehensive procedural explanation

---

## Key Features

### 1. Context Continuity
- Frontend chat history integration
- Automatic context summarization
- Reference ke percakapan sebelumnya

### 2. Multi-Strategy Retrieval
- **Direct**: Query asli
- **HYDE**: Enhanced dengan hypothetical documents  
- **Entity-based**: Fokus pada entitas hukum

### 3. Quality Assurance
- Answer quality scoring
- Automatic retry mechanism
- Confidence scoring

### 4. Legal Domain Optimization
- Legal entity extraction
- Indonesian law-specific processing
- Citation formatting

### 5. Performance Features
- Async processing
- Document deduplication
- Relevance filtering
- Iteration limits (max 2)

---

## Error Handling & Fallbacks

1. **Context Analysis Fails**: Fallback to no-context processing
2. **HYDE Fails**: Use direct retrieval only
3. **LLM Fails**: Return error message with sources
4. **No Documents Found**: Generate answer with disclaimer
5. **Quality Too Low**: Accept answer after 2 iterations

Sistem ini memberikan jawaban hukum yang akurat dengan mempertimbangkan konteks percakapan dan menggunakan multiple retrieval strategies untuk memastikan relevansi dokumen yang optimal.

## 5. Data Models (`models.py`)

### Purpose
Defines **Pydantic models** for type safety, validation, and API contracts.

### Key Models

#### AgentState - Workflow State Management
```python
class AgentState(BaseModel):
    query: str
    chat_id: str = ""
    retrieved_docs: List[RetrievedDocument] = Field(default_factory=list)
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    frontend_chat_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
```
- **Workflow state container** passes data between agents
- **Frontend integration** accepts external chat history
- **Progress tracking** monitors processing steps

#### QueryRequest/Response - API Contracts
```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=1000)
    session_id: Optional[str] = None
    include_context: bool = True
```
- **Input validation** ensures query quality
- **Session management** enables conversation tracking
- **Response formatting** standardizes outputs

#### DocumentMetadata - Document Information
```python
class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    document_type: DocumentType
    regulation_number: Optional[str] = None
    year: Optional[int] = None
```
- **Document tracking** maintains source information
- **Legal metadata** stores regulation numbers and years
- **Type safety** prevents data corruption

---

## System Flow

### 1. Document Processing Flow
```
PDF File → Text Extraction (PyMuPDF/OCR) → Indonesian NLP → Text Chunking → Vector Embeddings → ChromaDB Storage
```

### 2. Query Processing Flow
```
User Query → Context Analysis → Query Analysis → Smart Retrieval (Direct + HYDE + Entity) → Answer Generation → Quality Control → Response
```

### 3. Context Management Flow
```
Frontend Chat History → Context Conversion → Summary Generation → Context-Aware Processing → Updated Context Storage
```

## Key Technical Decisions

### Why Ollama for Embeddings?
- **Local deployment** ensures data privacy
- **High-quality embeddings** with mxbai-embed-large model
- **No external API costs** for embedding generation

### Why ChromaDB for Vector Storage?
- **Local storage** maintains data control
- **Efficient similarity search** with built-in optimizations
- **Persistent storage** survives application restarts

### Why HYDE for Query Enhancement?
- **Semantic gap bridging** between queries and documents
- **Legal domain adaptation** generates domain-specific hypothetical content
- **Improved retrieval accuracy** especially for complex legal queries

### Why LangGraph for Workflow?
- **Structured agent orchestration** ensures consistent processing
- **State management** maintains data flow between agents
- **Conditional logic** adapts workflow based on query characteristics

## Input/Output Specifications

### Document Processing
**Input:**
- PDF files in `./documents` folder
- File metadata (title, regulation number, year)

**Output:**
- Processed document chunks with embeddings
- Document metadata with processing statistics
- Vector storage ready for retrieval

### Query Processing
**Input:**
```python
QueryRequest(
    query="Jelaskan tugas dan wewenang KPK dalam pemberantasan korupsi",
    session_id="user_123",
    include_context=True
)
```

**Output:**
```python
QueryResponse(
    query="Jelaskan tugas dan wewenang KPK dalam pemberantasan korupsi",
    answer="KPK memiliki tugas utama dalam pemberantasan korupsi berdasarkan UU No. 30 Tahun 2002...",
    sources=[RetrievedDocument(...)],
    confidence_score=0.85,
    processing_time=2.34
)
```

### Chat with Context
**Input:**
```python
ChatRequest(
    message="Apa sanksi untuk pelaku korupsi?",
    session_id="session_456",
    chat_history=[
        {"role": "user", "content": "Jelaskan tentang KPK"},
        {"role": "assistant", "content": "KPK adalah lembaga..."}
    ]
)
```

**Output:**
```python
ChatResponse(
    message="Berdasarkan diskusi sebelumnya tentang KPK, sanksi untuk pelaku korupsi diatur dalam...",
    sources=[...],
    confidence_score=0.92
)
```

## Critical Features Explained

### 1. Incremental Document Processing
```python
def _is_file_processed(self, file_path: Path) -> bool:
    current_hash = self._get_file_hash(file_path)
    return stored_info.get('hash') == current_hash
```
**Purpose:** Only processes new or modified documents, saving computational resources.

### 2. OCR Fallback for Scanned Documents
```python
if not page_text.strip():
    page_text = await self._ocr_page(page)
```
**Purpose:** Handles scanned legal documents that don't have extractable text.

### 3. Multi-Strategy Document Retrieval
```python
# Strategy 1: Direct retrieval
direct_docs = await self._direct_retrieval(query, analysis)

# Strategy 2: HYDE-enhanced retrieval
hyde_docs = []
if analysis.get("requires_hyde", False):
    hyde_query_result = await self._hyde_retrieval(query, conversation_context)

# Strategy 3: Entity-based retrieval
entity_docs = await self._entity_based_retrieval(query, legal_entities)
```
**Purpose:** Maximizes retrieval coverage by using different approaches for different query types.

### 4. Conversation Context Integration
```python
def load_chat_history(self, chat_id: str, chat_history: List[Dict[str, Any]]) -> None:
    for msg_data in chat_history:
        message = ChatMessage(
            id=msg_data.get("id"),
            content=msg_data.get("content"),
            role=msg_data.get("role"),
            timestamp=datetime.fromisoformat(timestamp_str)
        )
```
**Purpose:** Maintains conversation continuity and provides context-aware responses.

### 5. Quality Control System
```python
def _assess_answer_quality(self, answer: str, query: str) -> float:
    score = 5.0  # Base score
    
    # Length check
    if len(answer) > 300:
        score += 1.0
    
    # Query term coverage
    covered_terms = sum(1 for term in query_terms if term in answer_lower)
    coverage_ratio = covered_terms / len(query_terms)
    score += coverage_ratio * 2.0
```
**Purpose:** Ensures answer quality and triggers re-processing if needed.

## System Benefits

### For Indonesian Legal Domain
- **Language-specific processing** handles Indonesian legal terminology
- **Legal entity recognition** identifies specific laws and articles
- **Regulation-aware formatting** understands Indonesian legal document structure

### For Performance
- **Incremental processing** reduces computational overhead
- **Async operations** improve responsiveness
- **Local deployment** eliminates API latency

### For Accuracy
- **Multi-agent validation** ensures answer quality
- **HYDE enhancement** improves semantic matching
- **Context awareness** provides more relevant responses

### For User Experience
- **Conversation continuity** maintains chat context
- **Source attribution** provides document references
- **Confidence scoring** indicates answer reliability

## Configuration and Deployment

### Environment Requirements
- **Ollama server** running locally on port 11434
- **Python dependencies** including PyMuPDF, Tesseract, LangChain
- **Indonesian NLP models** (Sastrawi)
- **Google Gemini API** for answer generation

### Storage Requirements
- **ChromaDB** for vector storage
- **Local file system** for PDF documents
- **JSON files** for processing metadata

This system represents a comprehensive approach to legal document processing and querying, specifically designed for Indonesian legal domain with advanced NLP and AI capabilities.


# Frontend Flow Aplikasi Indonesian Legal Assistant

## 1. Struktur Data dan State Management

### Interface Utama
```typescript
interface Message {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
  sources?: Array<{...}>         // Sumber referensi dari backend
  processing_time?: number       // Waktu proses AI
}

interface Chat {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  session_id?: string           // ID sesi dari backend
}

interface Document {
  id: string
  name: string
  size: number
  type: string
  content: string
  uploadedAt: Date
  status?: "processing" | "completed" | "error"
  file?: File                   // File asli untuk upload
}
```

### State Management
- **chats**: Array semua percakapan
- **currentChatId**: ID chat yang sedang aktif
- **documents**: Array dokumen yang diupload
- **isMessageLoading**: Status loading saat mengirim pesan
- **metrics**: Statistik aplikasi (total chats, docs, success rate)

## 2. Storage & Persistence

### Local Storage
```typescript
// Load chats dari localStorage saat aplikasi dimuat
useEffect(() => {
  const loadedChats = loadChatsFromStorage()
  setChats(loadedChats)
  setIsLoading(false)
}, [])

// Save chats ke localStorage setiap kali ada perubahan
useEffect(() => {
  if (!isLoading) {
    saveChatsToStorage(chats)
  }
}, [chats, isLoading])
```

### Backend Session Management
- Setiap chat memiliki `session_id` dari backend
- Chat history disimpan di backend dan bisa dimuat ulang
- Local storage hanya untuk UI state, data lengkap ada di backend

## 3. Flow Mengirim Pesan

### Step 1: Input Processing
```
User mengetik pesan di ChatInput
↓
Optional: User attach files (PDF, DOC, TXT, dll)
↓
User klik Send atau Enter
```

### Step 2: Message Creation
```typescript
const userMessage: Message = {
  id: `${Date.now()}_user`,
  content,
  role: "user", 
  timestamp: new Date(),
}

// Tambah ke state immediately untuk UX yang responsif
setChats(prev => prev.map(chat => 
  chat.id === chatId 
    ? { ...chat, messages: [...chat.messages, userMessage] }
    : chat
))
```

### Step 3: File Upload (jika ada attachment)
```
Jika ada files attached:
  ↓
  Buat FormData untuk setiap file
  ↓
  POST /api/upload untuk upload ke backend
  ↓
  Backend memproses file → chunks untuk RAG
  ↓
  Dapat document_id dari backend
  ↓
  Update pesan dengan info attachment
```

### Step 4: API Request ke Backend
```typescript
const requestBody = {
  message: content,
  session_id: sessionId,
  include_sources: true,
  chat_history: chatHistory.slice(-10)  // 10 pesan terakhir untuk context
}

POST /api/chat
↓
Backend memproses dengan RAG + AI
↓
Response: {
  session_id,
  response,
  sources,
  timestamp,
  processing_time
}
```

### Step 5: Response Processing
```
Backend response diterima
↓
Buat assistantMessage object
↓
Update chat state dengan response
↓
Update session_id di chat object
↓
Refresh metrics
```

## 4. Document Management Flow

### Upload Process
```
User select files di DocumentUpload component
↓
Files divalidasi (type, size)
↓
Buat Document object dengan status "processing"
↓
FormData dibuat dengan file
↓
POST /api/upload ke backend
↓
Backend:
  - Extract text dari file
  - Split ke chunks
  - Store di vector database
  - Return document_id
↓
Update document status ke "completed"
```

### Document Integration
- Dokumen yang diupload jadi bagian dari knowledge base
- Saat user bertanya, sistem RAG search di dokumen
- Sources dari dokumen ditampilkan di response

## 5. Chat History Management

### Local vs Backend Storage
```
Local Storage (localStorage):
  - UI state dan basic chat info
  - Cepat untuk loading initial UI
  - Backup jika backend tidak available

Backend Storage:
  - Complete chat history dengan session_id
  - Sources dan metadata lengkap
  - Authoritative source of truth
```

### History Loading Flow
```
User buka chat lama
↓
Cek: apakah chat punya session_id tapi messages kosong?
↓
Jika ya: loadChatHistoryFromBackend()
↓
GET /api/chat/{session_id}
↓
Convert backend format ke frontend Message[]
↓
Update chat state dengan loaded messages
```

## 6. Search & Filter

### Chat Search
```
User ketik di search box
↓
searchChats() function dipanggil
↓
Search di:
  - Chat titles
  - Message content
  - Timestamps
↓
Return filtered results dengan highlights
```

### Auto-suggestions
```
getSearchSuggestions() analyze:
  - Recent chat titles
  - Common phrases dari messages
  - Generate relevant suggestions
```

## 7. Error Handling & UX

### Optimistic Updates
- User message ditampilkan langsung sebelum API call
- Loading state ditampilkan selama proses
- Error handling jika API call gagal

### Graceful Degradation
- Jika backend down, chat tetap bisa dibuka dari localStorage
- Upload gagal tidak crash app
- Error messages user-friendly

## 8. Component Architecture

```
HomePage (Main Container)
├── Sidebar
│   ├── Chat list dengan search
│   └── Document management
├── ChatMessage (Message display)
├── ChatInput (Input dengan file attach)
└── DocumentUpload (File upload UI)
```

## 9. API Integration Points

### Endpoints Used
- `POST /api/chat` - Send message & get AI response
- `POST /api/upload` - Upload documents
- `GET /api/metrics` - App statistics
- `GET /api/chat/{session_id}` - Load chat history
- `POST /api/documents/reprocess` - Reprocess all docs
- `GET /api/documents/info` - Document statistics

### Data Flow Pattern
```
Frontend State → API Call → Backend Processing → Response → Update State → UI Update
```

## 10. Key Features

1. **Real-time Chat**: Instant UI updates dengan backend sync
2. **RAG Integration**: Document upload → Vector search → Contextual responses
3. **Session Persistence**: Chat history tersimpan di backend
4. **File Attachments**: Multi-file upload dengan progress tracking
5. **Search Functionality**: Full-text search across all chats
6. **Metrics Dashboard**: Real-time app statistics
7. **Responsive Design**: Mobile-friendly dengan collapsible sidebar

## 11. Performance Optimizations

- **Lazy Loading**: Chat history dimuat on-demand
- **Optimistic Updates**: UI responsif sebelum backend response
- **Debounced Search**: Search suggestions dengan delay
- **Memory Management**: Limit chat history yang dikirim ke backend (10 pesan terakhir)
- **File Validation**: Check file type/size sebelum upload
