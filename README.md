# MCP Chatbot — RAG-Powered AI Assistant

A production-deployed conversational AI that answers questions about the Model Context Protocol (MCP) — combining RAG over curated documentation with live web search for queries outside the knowledge base.

Built and iterated using a spec-driven development workflow: features are defined as EARS-format SCRUM tickets in `specs/`, read by a Claude AI coding agent in WebStorm, and implemented directly from the spec.

---

## Architecture

```
User Message
     │
     ▼
FastAPI /chat endpoint
     │
     ▼
LangChain Agent
     ├── kb_search (RAG) ──► FAISS Vector Store ──► Scraped MCP Docs
     └── Tavily Search  ──► Live Web Results
     │
     ▼
GPT-4o-mini
     │
     ▼
Response + LangSmith Trace
```

**Routing logic:**
- INTERNAL queries → `kb_search` (RAG over MCP documentation)
- EXTERNAL queries → Tavily web search
- HYBRID queries → both tools, synthesized response

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI (Python) |
| LLM | OpenAI GPT-4o-mini |
| RAG | LangChain + FAISS vector store |
| Embeddings | OpenAI Embeddings |
| Web Search | Tavily |
| Observability | LangSmith tracing |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (auto-deploy on merge to `main`) |

---

## How Features Are Built

This project follows a spec-driven development workflow:

```
Product Requirement
       │
       ▼
EARS-format SCRUM ticket (specs/*.md)
       │
       ▼
Claude AI coding agent reads ticket in WebStorm
       │
       ▼
Feature implemented → PR → merged to main → auto-deploy on Render
```

See the `specs/` folder for real ticket examples that drove features in this codebase.

---

## Specs

| Ticket | Feature |
|---|---|
| `specs/SCRUM-8-rename-to-chatbot.md` | Rename app to Chatbot |
| `specs/SCRUM-9-rename-to-agent-assistant.md` | Rename to Agent Assistant |
| `specs/SCRUM-10-weather-display-feature.md` | Weather display feature |
| `specs/Weather_Feature_PRD.docx` | PRD driving the weather feature tickets |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves chat UI |
| `/chat` | POST | `{ message, session_id }` → agent response |
| `/health` | GET | Agent + RAG status check |

---

## Local Setup

```bash
git clone git@github.com:shuvojit0194-star/Chatbot.git
cd Chatbot
pip install -r requirements.txt

export OPENAI_API_KEY=your_key
export TAVILY_API_KEY=your_key
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_TRACING_V2=true

python main.py
# Open http://localhost:8080
```

---

## Related

- [AI Feature Delivery Pipeline](https://github.com/shuvojit0194-star/AI_Feature_Delivery_Pipeline) — the workflow used to build and iterate this project
