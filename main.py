"""
MCP Chatbot — Cloud Run Backend
This is the production version of your Colab notebook, packaged for deployment.
"""

import os
import json
import warnings
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# LangChain imports (same as your notebook)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception:
    TavilySearchResults = None

# ── Global agent objects (loaded once at startup) ────────────────
agent     = None
rag_chain = None
retriever = None


# ── Your exact helper functions from the notebook ────────────────
def _final_text(res: Any) -> str:
    if isinstance(res, AIMessage):
        return res.content or ""
    if isinstance(res, dict) and "messages" in res:
        for m in reversed(res["messages"]):
            if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
                return getattr(m, "content", "") or ""
    return str(res)


def _collect_tool_calls_and_outputs(res: Any, max_len: int = 500) -> List[Dict[str, Any]]:
    messages: List[Any] = []
    if isinstance(res, dict) and "messages" in res:
        messages = res["messages"]

    tool_calls: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls = m.tool_calls or []
            break
        addkw = getattr(m, "additional_kwargs", {}) if hasattr(m, "additional_kwargs") else {}
        if addkw.get("tool_calls"):
            tool_calls = addkw["tool_calls"]
            break

    outputs: Dict[str, str] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            outputs[getattr(m, "tool_call_id", None)] = (getattr(m, "content", "") or "")

    def trunc(s: Optional[str]) -> str:
        if not s:
            return ""
        return s[:max_len] + ("…" if len(s) > max_len else "")

    structured: List[Dict[str, Any]] = []
    for c in tool_calls:
        structured.append({
            "name":   c.get("name"),
            "args":   c.get("args"),
            "output": trunc(outputs.get(c.get("id")))
        })
    return structured


def _usage(res: Any) -> Optional[Dict[str, Any]]:
    if isinstance(res, AIMessage):
        return getattr(res, "response_metadata", {}).get("token_usage")
    if isinstance(res, dict) and "messages" in res:
        for m in reversed(res["messages"]):
            meta = getattr(m, "response_metadata", {}) if hasattr(m, "response_metadata") else {}
            if meta.get("token_usage"):
                return meta["token_usage"]
    return None


def to_structured(res: Any) -> Dict[str, Any]:
    return {
        "answer": _final_text(res),
        "tools":  _collect_tool_calls_and_outputs(res),
        "usage":  _usage(res)
    }


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ── Startup: build RAG + Agent (runs once when container starts) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, rag_chain, retriever

    print("🚀 Starting MCP Chatbot — loading RAG and Agent...")

    try:
        # 1) Load URLs from scraped_urls.json (bundled with the app)
        with open("scraped_urls.json", "r") as f:
            loaded_urls = json.load(f)
        print(f"✅ Loaded {len(loaded_urls)} URLs")

        # 2) Load web documents
        print("📄 Loading documents from URLs (this may take a minute)...")
        loader = WebBaseLoader(loaded_urls[:50])   # limit to 50 for faster cold start
        docs = loader.load()

        # 3) Chunk
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # 4) Embed & index
        emb = OpenAIEmbeddings()
        vs = FAISS.from_documents(chunks, emb)
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        print(f"✅ FAISS index built with {len(chunks)} chunks")

        # 5) RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a precise assistant helping users learn about MCP. "
             "Use simple language and give examples. Use the provided CONTEXT to answer.\n"
             "If the answer isn't in the context, say you don't know.\n\nCONTEXT:\n{context}"),
            ("human", "{question}")
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

        # 6) Agent tools
        tools = []
        kb_tool = create_retriever_tool(
            retriever, name="kb_search",
            description="Search the indexed MCP docs and return relevant passages."
        )
        tools.append(kb_tool)

        if TavilySearchResults and os.getenv("TAVILY_API_KEY"):
            tavily = TavilySearchResults(
                api_key=os.getenv("TAVILY_API_KEY"),
                max_results=5, include_answer=True
            )
            tools.append(tavily)
            print("✅ Tavily web search enabled")

        # 7) Agent
        agent = create_agent(
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            tools,
            system_prompt="""
You are an evidence-grounded AI agent with access to:
- kb_search: RAG Retriever (internal indexed MCP docs)
- tavily: Tavily Web Search (external/real-time)

GOAL: Answer accurately using retrieved evidence only.

1) DECISION + ROUTING
Classify the user query:
- INTERNAL: Use kb_search for internal docs / uploaded / indexed knowledge
- EXTERNAL: Use tavily for current/latest/recent information
- HYBRID: needs both → RAG first, then Tavily if gaps remain
- AMBIGUOUS: unclear intent → ask 1 clarifying question

Rules:
- Do not answer from memory when tools can be used.
- If retrieval returns nothing relevant, say so.

2) RESPONSE GUIDELINES
- Use only retrieved content; do not invent facts.
- Be concise, structured, and directly address the question.
- If sources conflict, mention the disagreement briefly.
- When using numbered lists, ALWAYS use sequential numbers: 1. 2. 3. 4. 5. (NOT all 1.)
- Format lists properly with each item on a new line.

3) GROUNDEDNESS REQUIREMENT
Every factual claim must be supported by retrieved evidence.
If insufficient evidence: "I don't have enough verified information to answer this accurately."
"""
        )
        print("✅ Agent ready — MCP Chatbot is live!")

    except Exception as e:
        print(f"⚠️  Startup error: {e}")
        print("   Agent/RAG may be unavailable. Check your API keys.")

    yield   # App runs here

    print("👋 Shutting down MCP Chatbot")


# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(title="MCP Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve the HTML frontend ──────────────────────────────────────
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


# ── Chat endpoint ────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: dict):
    message    = req.get("message", "").strip()
    session_id = req.get("session_id", "default")

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        if agent:
            raw        = agent.invoke({"messages": [{"role": "user", "content": message}]})
            structured = to_structured(raw)
            reply      = structured.get("answer", "").strip() or "I was unable to find a verified answer."
            return {
                "reply":       reply,
                "session_id":  session_id,
                "tools_used":  structured.get("tools", []),
                "token_usage": structured.get("usage"),
            }

        elif rag_chain:
            reply = rag_chain.invoke(message)
            return {"reply": reply, "session_id": session_id, "tools_used": []}

        else:
            raise HTTPException(status_code=503, detail="Agent not ready yet. Please wait and retry.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "service":   "MCP Chatbot",
        "agent":     agent is not None,
        "rag_chain": rag_chain is not None,
        "retriever": retriever is not None,
        "model":     "gpt-4o-mini",
    }


# ── Local dev entry point ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
