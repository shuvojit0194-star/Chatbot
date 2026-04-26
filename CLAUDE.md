# CLAUDE.md — MCP Chatbot Project Context

## Project Overview
This is an AI-powered RAG chatbot focused on the **Model Context Protocol (MCP)**.
It is deployed on **Render** from the **GitHub repo: shuvojit0194-star/Chatbot**.

## Tech Stack
- **Backend**: FastAPI + Python (`main.py`)
- **LLM**: OpenAI GPT-4o-mini via LangChain
- **RAG**: FAISS vector store + OpenAI Embeddings
- **Agent Tools**: `kb_search` (internal RAG) + Tavily web search (external)
- **Frontend**: Vanilla HTML/CSS/JS (`static/index.html`)
- **Deployment**: Render (auto-deploys on merge to `main`)
- **Observability**: LangSmith tracing

## Repository Structure
```
/
├── main.py              ← FastAPI backend (RAG + Agent)
├── static/
│   └── index.html       ← Chat UI frontend
├── scraped_urls.json    ← URL sources for RAG indexing
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Container definition
├── CLAUDE.md            ← You are here
└── specs/               ← Feature specs for Claude Code to implement
    └── FEAT-XXX-name.md
```

## Key Architectural Decisions
- RAG retriever uses `k=4` chunks per query (FAISS)
- Agent routing: INTERNAL → kb_search, EXTERNAL → Tavily, HYBRID → both
- Cold start on Render: ~60 seconds for FAISS index to build on boot
- CORS is open (`allow_origins=["*"]`) — tighten for production
- URL list is capped at 50 for faster cold start (`loaded_urls[:50]`)

## API Endpoints
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves chat UI |
| `/chat` | POST | Main chat endpoint `{message, session_id}` |
| `/health` | GET | Health check with agent/rag status |

## Environment Variables Required
```
OPENAI_API_KEY
TAVILY_API_KEY
LANGCHAIN_API_KEY
LANGCHAIN_TRACING_V2=true
```

## Workflow for Claude Code
When implementing a spec from `specs/`:
1. Read the full spec file before touching any code
2. Make changes only to files referenced in the spec
3. Run a quick sanity check: ensure `/health` would return `agent: true`
4. Commit with message format: `feat(SCRUM-XXX): <short description>`
5. Push to a branch named `feat/SCRUM-XXX-short-name`
6. Do NOT modify `scraped_urls.json` unless the spec explicitly asks for it
7. Do NOT change deployment config (Dockerfile, requirements.txt) unless spec requires it

## Testing Guidance
- Test the `/chat` endpoint with: `curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message":"What is MCP?"}'`
- Confirm the `/health` endpoint returns `agent: true` and `rag_chain: true`
- For frontend changes, verify in Chrome at `http://localhost:8080`

## Jira Project
- **Project**: SCRUM
- **Site**: https://shuvojit0194.atlassian.net
- **Issue types**: Epic > Story
- **Branch naming**: `feat/SCRUM-{id}-{short-name}`
- **Commit format**: `feat(SCRUM-{id}): description`
