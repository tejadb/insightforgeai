# InsightForge AI - Complete Architecture & Implementation Plan

## 📋 Executive Summary

InsightForge AI is being rebuilt as a **conversational document intelligence system** with three core capabilities:
1. **Generate Summary** - Create document briefs on-demand
2. **Generate Q&A** - Generate practice interview questions
3. **RAG Chat** - Answer questions about documents using vector search

The system uses **LangChain Agents** for intelligent routing, with an option for **direct routing** to bypass the agent for specific endpoints.

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                          │
│  - Chat Interface                                            │
│  - Document Upload                                           │
│  - Summary/Q&A Display                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP/WebSocket
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI GATEWAY                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Router Layer                                         │   │
│  │  - POST /chat (Agent or Direct)                       │   │
│  │  - POST /generate-summary (Direct)                    │   │
│  │  - POST /generate-qa (Direct)                         │   │
│  │  - POST /upload (Triggers indexing)                   │   │
│  │  - GET /health                                         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   AGENT     │ │   DIRECT    │ │   RAG       │
│  (Router)   │ │  (Bypass)   │ │  (Vector)   │
└─────────────┘ └─────────────┘ └─────────────┘
```

### Component Flow

```
User Request
    ↓
FastAPI Router
    ↓
    ├─→ Direct Route? → Execute directly (summary/qa)
    │
    └─→ Agent Route? → LangChain Agent
                          ↓
                    ┌─────┴─────┐
                    │           │
              Intent:      Intent:
              Summary      Q&A
                    │           │
                    └─────┬─────┘
                          ↓
                    Execute Tool
                          ↓
                    Return Result
```

---

## 📊 Database Schema

### Sessions Table

```sql
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);
```

**Purpose**: Track conversation sessions per user

### Messages Table

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
);
```

**Purpose**: Store individual messages in conversations

### Documents Table (Existing: db9Documents)

```sql
-- Already exists, we use:
- id (document_id)
- user_id
- title
- kind
- storage_path
- summary_short (for generated summaries)
- summary_json (for generated Q&A and metadata)
- uploaded_at
```

### Vector Store (Supabase pgvector)

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR NOT NULL,
    user_id UUID NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    embedding vector(1536),  -- OpenAI embedding dimension
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_document_id (document_id),
    INDEX idx_user_id (user_id),
    INDEX idx_embedding USING ivfflat (embedding vector_cosine_ops)
);
```

**Purpose**: Store document chunks with embeddings for RAG search

---

## 🔧 Component Breakdown

### 1. FastAPI Router (Entry Point)

**Location**: `agents/insight_forge_ai/router.py`

**Responsibilities**:
- Authentication (X-Service-Secret)
- Request validation
- Route to agent or direct handler
- Error handling
- Response formatting

**Endpoints**:

```python
# Main chat endpoint (uses agent)
POST /api/v1/insight-forge/chat
Body: {
    "session_id": "uuid",
    "user_id": "uuid",
    "message": "Create summary of my ML assignment",
    "use_agent": true  # Optional, default true
}

# Direct summary generation (bypasses agent)
POST /api/v1/insight-forge/generate-summary
Body: {
    "document_id": "123",
    "user_id": "uuid",
    "session_id": "uuid"  # Optional, for logging
}

# Direct Q&A generation (bypasses agent)
POST /api/v1/insight-forge/generate-qa
Body: {
    "document_id": "123",
    "user_id": "uuid",
    "session_id": "uuid"  # Optional
}

# Document upload (triggers indexing)
POST /api/v1/insight-forge/upload
Body: {
    "document_id": "123",
    "user_id": "uuid",
    "file_path": "..."
}

# Health check
GET /api/v1/insight-forge/health
```

**Code Structure**:

```python
from fastapi import APIRouter, HTTPException, Depends
from .schemas import ChatRequest, GenerateSummaryRequest, GenerateQARequest
from .agent.agent_router import route_with_agent
from .handlers.direct_handlers import generate_summary_direct, generate_qa_direct

router = APIRouter(prefix="/api/v1/insight-forge", tags=["InsightForge AI"])

@router.post("/chat")
async def chat(request: ChatRequest):
    if request.use_agent:
        return await route_with_agent(request)
    else:
        # Direct routing logic
        return await handle_direct_chat(request)

@router.post("/generate-summary")
async def generate_summary(request: GenerateSummaryRequest):
    return await generate_summary_direct(request)

@router.post("/generate-qa")
async def generate_qa(request: GenerateQARequest):
    return await generate_qa_direct(request)
```

---

### 2. LangChain Agent Router

**Location**: `agents/insight_forge_ai/agent/agent_router.py`

**Purpose**: Intelligent routing using LangChain agent to determine user intent

**Architecture**:

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

class AgentRouter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = self._create_tools()
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def _create_tools(self):
        return [
            Tool(
                name="generate_summary",
                func=self._generate_summary_tool,
                description="Use this when user asks to create a summary, brief, or overview of a document. Input: document_id"
            ),
            Tool(
                name="generate_qa",
                func=self._generate_qa_tool,
                description="Use this when user asks to create questions, Q&A, practice questions, or interview questions. Input: document_id"
            ),
            Tool(
                name="answer_question",
                func=self._answer_question_tool,
                description="Use this when user asks a question about their documents. Input: question text"
            )
        ]
    
    async def route(self, message: str, user_id: str, session_id: str):
        """
        Route user message using agent
        """
        # Load conversation history
        history = await load_conversation_history(session_id)
        
        # Build context
        context = f"""
        User ID: {user_id}
        Session ID: {session_id}
        
        Previous conversation:
        {format_history(history)}
        
        Current message: {message}
        
        Available actions:
        1. generate_summary - Create document summary
        2. generate_qa - Create practice Q&A
        3. answer_question - Answer questions about documents
        """
        
        # Agent decides what to do
        response = await self.agent.arun(context)
        
        # Save to conversation
        await save_message(session_id, "user", message)
        await save_message(session_id, "assistant", response)
        
        return response
```

**Agent Decision Flow**:

```
User: "Create summary of my ML assignment"
    ↓
Agent analyzes intent
    ↓
Agent calls: generate_summary(document_id="123")
    ↓
Tool executes → Returns summary
    ↓
Agent formats response → Returns to user
```

---

### 3. Direct Handlers (Bypass Agent)

**Location**: `agents/insight_forge_ai/handlers/direct_handlers.py`

**Purpose**: Direct execution without agent routing (faster, simpler)

#### 3.1 Generate Summary Handler

```python
async def generate_summary_direct(request: GenerateSummaryRequest):
    """
    Direct summary generation without agent
    """
    # 1. Validate document access
    document = await repository.get_document(request.document_id, request.user_id)
    
    # 2. Check if already generated
    if document.get('summary_short'):
        return {"summary": document['summary_short'], "from_cache": True}
    
    # 3. Get document content (from vector store or file)
    content = await get_document_content(document['storage_path'])
    
    # 4. Generate summary using LLM
    summary = await llm_client.generate_summary(
        content=content,
        document_title=document['title'],
        document_type=document['kind']
    )
    
    # 5. Save to database
    await repository.update_document(
        request.document_id,
        {"summary_short": summary}
    )
    
    # 6. Log to session if provided
    if request.session_id:
        await save_message(
            request.session_id,
            "assistant",
            f"Generated summary for {document['title']}"
        )
    
    return {
        "success": True,
        "summary": summary,
        "document_id": request.document_id
    }
```

#### 3.2 Generate Q&A Handler

```python
async def generate_qa_direct(request: GenerateQARequest):
    """
    Direct Q&A generation without agent
    """
    # 1. Validate document access
    document = await repository.get_document(request.document_id, request.user_id)
    
    # 2. Check if already generated
    summary_json = document.get('summary_json', {})
    if summary_json.get('practice_qa'):
        return {"qa": summary_json['practice_qa'], "from_cache": True}
    
    # 3. Get document content
    content = await get_document_content(document['storage_path'])
    
    # 4. Generate Q&A using LLM
    qa = await llm_client.generate_qa(
        content=content,
        document_title=document['title'],
        document_type=document['kind']
    )
    
    # 5. Save to database
    await repository.update_document(
        request.document_id,
        {
            "summary_json": {
                **summary_json,
                "practice_qa": qa,
                "qa_generated_at": datetime.now().isoformat()
            }
        }
    )
    
    # 6. Log to session if provided
    if request.session_id:
        await save_message(
            request.session_id,
            "assistant",
            f"Generated Q&A for {document['title']}"
        )
    
    return {
        "success": True,
        "qa": qa,
        "document_id": request.document_id
    }
```

---

### 4. RAG System (Vector Search)

**Location**: `agents/insight_forge_ai/rag/`

**Purpose**: Semantic search across user documents

#### 4.1 Document Indexer

**File**: `rag/document_indexer.py`

```python
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import supabase

class DocumentIndexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.db = get_supabase_client()
    
    async def index_document(
        self, 
        document_id: str, 
        user_id: str, 
        file_path: str
    ):
        """
        Process document: Load → Chunk → Embed → Store
        """
        # 1. Load document
        loader = self._get_loader(file_path)
        documents = loader.load()
        
        # 2. Add metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "document_id": document_id,
                "user_id": user_id,
                "page_number": i + 1
            })
        
        # 3. Chunk
        chunks = self.splitter.split_documents(documents)
        
        # 4. Generate embeddings (batch)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = await self.embeddings.aembed_documents(texts)
        
        # 5. Store in vector DB
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await self.db.table("document_chunks").insert({
                "document_id": document_id,
                "user_id": user_id,
                "chunk_text": chunk.page_content,
                "chunk_index": i,
                "page_number": chunk.metadata.get("page_number", 0),
                "embedding": embedding,
                "metadata": chunk.metadata
            }).execute()
        
        return {"chunks_created": len(chunks), "status": "indexed"}
    
    def _get_loader(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".docx":
            return Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
```

#### 4.2 Document Retriever

**File**: `rag/document_retriever.py`

```python
from langchain_openai import OpenAIEmbeddings
import supabase

class DocumentRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.db = get_supabase_client()
    
    async def search(
        self, 
        query: str, 
        user_id: str, 
        top_k: int = 5,
        document_ids: List[str] = None
    ) -> List[Dict]:
        """
        Semantic search across user's documents
        """
        # 1. Embed query
        query_embedding = await self.embeddings.aembed_query(query)
        
        # 2. Build filter
        filter_dict = {"user_id": {"eq": user_id}}
        if document_ids:
            filter_dict["document_id"] = {"in": document_ids}
        
        # 3. Vector search using pgvector
        results = await self.db.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.7,
                "match_count": top_k,
                "user_id": user_id
            }
        ).execute()
        
        return [
            {
                "text": result["chunk_text"],
                "document_id": result["document_id"],
                "page_number": result["page_number"],
                "score": result.get("similarity", 0)
            }
            for result in results.data
        ]
```

**SQL Function for Vector Search**:

```sql
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding vector(1536),
    match_threshold float,
    match_count int,
    user_id uuid
)
RETURNS TABLE (
    id uuid,
    document_id varchar,
    chunk_text text,
    page_number int,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_text,
        dc.page_number,
        1 - (dc.embedding <=> query_embedding) as similarity
    FROM document_chunks dc
    WHERE dc.user_id = match_document_chunks.user_id
        AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

---

### 5. LLM Service

**Location**: `agents/insight_forge_ai/services/llm_service.py`

**Purpose**: Centralized LLM calls for summary, Q&A, and chat

```python
from utils.llm_client import unified_llm_client

class LLMService:
    async def generate_summary(
        self, 
        content: str, 
        document_title: str, 
        document_type: str
    ) -> str:
        """
        Generate document summary
        """
        prompt = f"""
        Analyze this {document_type} titled "{document_title}" and create a comprehensive summary.
        
        Document Content:
        {content[:10000]}  # Limit to 10k chars
        
        Create a structured summary including:
        - Overview of main topic and scope
        - Key findings and achievements
        - Methodology and approach
        - Practical applications and impact
        
        Format as clear, professional text suitable for interview preparation.
        """
        
        response = await unified_llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0,
            agent_id="insight_forge",
            tags={"feature": "generate_summary"}
        )
        
        return response['content']
    
    async def generate_qa(
        self, 
        content: str, 
        document_title: str, 
        document_type: str
    ) -> str:
        """
        Generate practice Q&A
        """
        prompt = f"""
        Create interview practice questions and answers based on this {document_type} titled "{document_title}".
        
        Document Content:
        {content[:10000]}
        
        Generate 5-6 questions with detailed answers:
        - 2 basic level questions (fundamental understanding)
        - 2-3 intermediate level questions (technical details)
        - 1-2 advanced level questions (impact and implications)
        
        Format each as:
        Q: [Question]
        A: [Comprehensive answer with specific examples]
        
        Focus on interview preparation and professional discussion.
        """
        
        response = await unified_llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0,
            agent_id="insight_forge",
            tags={"feature": "generate_qa"}
        )
        
        return response['content']
    
    async def answer_question(
        self, 
        question: str, 
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Answer user question using RAG context
        """
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Document: {chunk['document_id']}, Page {chunk['page_number']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Build conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
        
        prompt = f"""
        You are InsightForge AI, helping users understand their documents.
        
        Context from user's documents:
        {context}
        
        Previous conversation:
        {history_text}
        
        User question: {question}
        
        Answer the question using the provided context. Include specific references to document pages when relevant.
        If the answer isn't in the context, say so clearly.
        """
        
        response = await unified_llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.1,
            agent_id="insight_forge",
            tags={"feature": "rag_chat"}
        )
        
        return response['content']
```

---

### 6. Session & Message Management

**Location**: `agents/insight_forge_ai/memory/session_manager.py`

```python
import supabase
from datetime import datetime
from typing import List, Dict, Optional

class SessionManager:
    def __init__(self):
        self.db = get_supabase_client()
    
    async def create_session(self, user_id: str) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        
        await self.db.table("sessions").insert({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        result = await self.db.table("sessions")\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()
        
        return result.data[0] if result.data else None
    
    async def save_message(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ):
        """Save message to session"""
        await self.db.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat()
        }).execute()
        
        # Update session updated_at
        await self.db.table("sessions")\
            .update({"updated_at": datetime.now().isoformat()})\
            .eq("session_id", session_id)\
            .execute()
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 20
    ) -> List[Dict]:
        """Get conversation history"""
        result = await self.db.table("messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []
    
    async def get_recent_sessions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's recent sessions"""
        result = await self.db.table("sessions")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []
```

---

### 7. Repository Layer

**Location**: `agents/insight_forge_ai/repository/document_repo.py`

```python
import supabase
from typing import Optional, Dict

class DocumentRepository:
    def __init__(self):
        self.db = get_supabase_client()
    
    async def get_document(
        self, 
        document_id: str, 
        user_id: str
    ) -> Optional[Dict]:
        """Get document with user validation"""
        result = await self.db.table("db9Documents")\
            .select("*")\
            .eq("id", document_id)\
            .eq("user_id", user_id)\
            .execute()
        
        return result.data[0] if result.data else None
    
    async def update_document(
        self, 
        document_id: str, 
        updates: Dict
    ) -> bool:
        """Update document fields"""
        result = await self.db.table("db9Documents")\
            .update(updates)\
            .eq("id", document_id)\
            .execute()
        
        return len(result.data) > 0
    
    async def list_user_documents(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[Dict]:
        """List user's documents"""
        result = await self.db.table("db9Documents")\
            .select("id, title, kind, uploaded_at, summary_short")\
            .eq("user_id", user_id)\
            .order("uploaded_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []
```

---

## 📁 Complete File Structure

```
agents/insight_forge_ai/
├── __init__.py
│
├── router.py                    # FastAPI endpoints
│   ├── POST /chat
│   ├── POST /generate-summary
│   ├── POST /generate-qa
│   ├── POST /upload
│   └── GET /health
│
├── agent/
│   ├── __init__.py
│   └── agent_router.py          # LangChain agent for routing
│
├── handlers/
│   ├── __init__.py
│   └── direct_handlers.py       # Direct execution (bypass agent)
│       ├── generate_summary_direct()
│       └── generate_qa_direct()
│
├── rag/
│   ├── __init__.py
│   ├── document_indexer.py      # Process & index documents
│   └── document_retriever.py    # Vector search
│
├── services/
│   ├── __init__.py
│   └── llm_service.py           # LLM calls (summary, Q&A, chat)
│
├── memory/
│   ├── __init__.py
│   └── session_manager.py      # Session & message management
│
├── repository/
│   ├── __init__.py
│   └── document_repo.py         # Database operations
│
└── schemas/
    ├── __init__.py
    ├── requests.py              # Pydantic request models
    └── responses.py             # Pydantic response models
```

---

## 🔄 Request Flow Examples

### Example 1: Generate Summary (Direct)

```
1. User: POST /generate-summary
   Body: {"document_id": "123", "user_id": "uuid"}
   ↓
2. FastAPI Router → Direct Handler
   ↓
3. DocumentRepository.get_document() → Validate access
   ↓
4. Check cache → If exists, return cached
   ↓
5. Get document content (from file or vector store)
   ↓
6. LLMService.generate_summary() → Call OpenAI
   ↓
7. DocumentRepository.update_document() → Save to db9Documents.summary_short
   ↓
8. Return summary to user
```

### Example 2: Generate Q&A (Direct)

```
1. User: POST /generate-qa
   Body: {"document_id": "123", "user_id": "uuid"}
   ↓
2. FastAPI Router → Direct Handler
   ↓
3. DocumentRepository.get_document() → Validate access
   ↓
4. Check cache → If exists, return cached
   ↓
5. Get document content
   ↓
6. LLMService.generate_qa() → Call OpenAI
   ↓
7. DocumentRepository.update_document() → Save to db9Documents.summary_json.practice_qa
   ↓
8. Return Q&A to user
```

### Example 3: RAG Chat (Agent)

```
1. User: POST /chat
   Body: {
       "session_id": "uuid",
       "user_id": "uuid",
       "message": "What did I learn about neural networks?",
       "use_agent": true
   }
   ↓
2. FastAPI Router → Agent Router
   ↓
3. SessionManager.get_conversation_history() → Load history
   ↓
4. LangChain Agent analyzes intent → "answer_question"
   ↓
5. Agent calls: answer_question tool
   ↓
6. DocumentRetriever.search() → Vector search across user docs
   ↓
7. LLMService.answer_question() → Generate answer with context
   ↓
8. SessionManager.save_message() → Save both messages
   ↓
9. Return answer to user
```

### Example 4: Generate Summary via Chat (Agent)

```
1. User: POST /chat
   Body: {
       "message": "Create summary of my ML assignment",
       "use_agent": true
   }
   ↓
2. Agent Router → LangChain Agent
   ↓
3. Agent analyzes → Intent: "generate_summary"
   ↓
4. Agent calls: generate_summary tool
   ↓
5. Tool executes → Same flow as Direct Handler
   ↓
6. Agent formats response → Returns to user
   ↓
7. Save to conversation history
```

---

## 🛠️ Technology Stack

| Component | Technology | Version/Model |
|-----------|-----------|---------------|
| API Framework | FastAPI | Latest |
| Agent Framework | LangChain | Latest |
| LLM | OpenAI | GPT-4o-mini |
| Embeddings | OpenAI | text-embedding-3-small |
| Vector DB | Supabase pgvector | Latest |
| Document Loaders | LangChain | Latest |
| Text Splitting | LangChain | RecursiveCharacterTextSplitter |
| Database | Supabase PostgreSQL | Latest |
| Session Storage | Supabase PostgreSQL | Custom tables |

---

## 📝 Pydantic Schemas

### Request Schemas

```python
# schemas/requests.py

from pydantic import BaseModel, Field
from typing import Optional
import uuid

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session ID")
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., description="User message")
    use_agent: bool = Field(True, description="Use agent routing or direct")
    
    @validator('session_id', 'user_id')
    def validate_uuid(cls, v):
        uuid.UUID(v)
        return v

class GenerateSummaryRequest(BaseModel):
    document_id: str = Field(..., description="Document ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Optional session ID for logging")

class GenerateQARequest(BaseModel):
    document_id: str = Field(..., description="Document ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Optional session ID for logging")

class UploadRequest(BaseModel):
    document_id: str = Field(..., description="Document ID")
    user_id: str = Field(..., description="User ID")
    file_path: str = Field(..., description="Path to uploaded file")
```

### Response Schemas

```python
# schemas/responses.py

from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    response: str
    metadata: Optional[Dict[str, Any]] = None

class SummaryResponse(BaseModel):
    success: bool
    summary: str
    document_id: str
    from_cache: bool = False

class QAResponse(BaseModel):
    success: bool
    qa: str
    document_id: str
    from_cache: bool = False

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    chunks_indexed: int
    status: str
```

---

## 🚀 Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Tasks**:
- [ ] Set up database tables (sessions, messages, document_chunks)
- [ ] Create FastAPI router structure
- [ ] Implement SessionManager
- [ ] Implement DocumentRepository
- [ ] Create Pydantic schemas
- [ ] Set up basic error handling

**Deliverables**:
- Database schema ready
- Basic API structure
- Session management working

---

### Phase 2: RAG System (Week 2)

**Tasks**:
- [ ] Implement DocumentIndexer
- [ ] Implement DocumentRetriever
- [ ] Create pgvector SQL functions
- [ ] Test document indexing
- [ ] Test vector search

**Deliverables**:
- Documents can be indexed
- Vector search working
- Embeddings stored correctly

---

### Phase 3: LLM Service & Direct Handlers (Week 3)

**Tasks**:
- [ ] Implement LLMService (summary, Q&A, chat)
- [ ] Implement generate_summary_direct handler
- [ ] Implement generate_qa_direct handler
- [ ] Test summary generation
- [ ] Test Q&A generation
- [ ] Implement caching logic

**Deliverables**:
- Direct endpoints working
- Summary generation working
- Q&A generation working

---

### Phase 4: Agent Router (Week 4)

**Tasks**:
- [ ] Implement LangChain agent setup
- [ ] Create agent tools (summary, Q&A, answer_question)
- [ ] Implement agent routing logic
- [ ] Test agent decision-making
- [ ] Integrate with chat endpoint

**Deliverables**:
- Agent routing working
- Tools integrated
- Chat endpoint with agent working

---

### Phase 5: Integration & Testing (Week 5)

**Tasks**:
- [ ] End-to-end testing
- [ ] Error handling improvements
- [ ] Performance optimization
- [ ] Documentation
- [ ] API testing

**Deliverables**:
- Complete system working
- All endpoints tested
- Documentation complete

---

## 🔐 Security Considerations

1. **Authentication**: All endpoints require X-Service-Secret
2. **Authorization**: User validation on all document operations
3. **Data Isolation**: Vector search filtered by user_id
4. **Input Validation**: Pydantic schemas validate all inputs
5. **Error Handling**: No sensitive data in error messages

---

## 📊 Performance Considerations

1. **Caching**: Check for existing summaries/Q&A before generation
2. **Batch Processing**: Batch embedding generation
3. **Async Operations**: All I/O operations are async
4. **Vector Index**: Proper indexing on embedding column
5. **Connection Pooling**: Reuse database connections

---

## 🧪 Testing Strategy

1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test component interactions
3. **E2E Tests**: Test complete user flows
4. **Load Tests**: Test with multiple concurrent requests

---

## 📚 Next Steps

1. Review and approve this architecture
2. Set up database migrations
3. Create initial file structure
4. Begin Phase 1 implementation
5. Iterate based on feedback

---

## ❓ Open Questions

1. **Session TTL**: How long to keep sessions? (Recommendation: 30 days)
2. **Chunk Size**: 1000 chars optimal? (Can be tuned)
3. **Top K**: How many chunks for RAG? (Recommendation: 5)
4. **Cache Strategy**: When to invalidate cached summaries?
5. **Error Recovery**: How to handle failed LLM calls?

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-06  
**Status**: Ready for Implementation

