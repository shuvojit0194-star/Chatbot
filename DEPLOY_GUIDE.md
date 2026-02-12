# 🚀 MCP Chatbot — Deploy to Google Cloud Run
### Complete Step-by-Step Guide (Beginner Friendly)

---

## What You'll End Up With

A **live public URL** like `https://mcp-chatbot-abc123-uc.a.run.app` that anyone
can open in their browser to use your MCP Chatbot — no Colab, no ngrok needed.

---

## Files in This Folder

```
mcp-deploy/
├── main.py              ← Your FastAPI backend (agent + RAG)
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Instructions to build the container
├── .dockerignore        ← Files to exclude from the container
├── .gitignore           ← Files to exclude from Git
├── scraped_urls.json    ← ⚠️  YOU MUST ADD THIS (see Step 2)
└── static/
    └── index.html       ← Your MCP Chatbot UI
```

---

## STEP 1 — Install Google Cloud CLI

1. Go to: https://cloud.google.com/sdk/docs/install
2. Download and run the installer for your OS (Windows/Mac/Linux)
3. Open a terminal and run:
   ```bash
   gcloud init
   ```
4. Follow the prompts — log in with your Google account and select/create a project

**Verify it worked:**
```bash
gcloud --version
```

---

## STEP 2 — Add Your scraped_urls.json

Your notebook already generated this file. You need to copy it here.

**In Google Colab, run this cell:**
```python
from google.colab import files
files.download('scraped_urls.json')
```

Then move the downloaded `scraped_urls.json` into this `mcp-deploy/` folder.

---

## STEP 3 — Create a GitHub Repository

This lets Cloud Run auto-deploy whenever you push changes.

1. Go to https://github.com and click **"New repository"**
2. Name it `mcp-chatbot`
3. Set it to **Private** (your API keys will be in Cloud Run secrets, not here)
4. Click **"Create repository"**

Then in your terminal, navigate to this folder and run:
```bash
cd mcp-deploy

git init
git add .
git commit -m "Initial MCP Chatbot deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mcp-chatbot.git
git push -u origin main
```
> Replace `YOUR_USERNAME` with your actual GitHub username

---

## STEP 4 — Enable Google Cloud APIs

Run these commands in your terminal:
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

This takes about 1 minute.

---

## STEP 5 — Store Your API Keys as Secrets

**Never put API keys directly in code.** Cloud Run uses Secret Manager.

```bash
# Store your OpenAI API key
echo -n "sk-proj-YOUR_OPENAI_KEY_HERE" | \
  gcloud secrets create OPENAI_API_KEY --data-file=-

# Store your Tavily API key
echo -n "tvly-dev-YOUR_TAVILY_KEY_HERE" | \
  gcloud secrets create TAVILY_API_KEY --data-file=-

# Store your LangChain API key
echo -n "lsv2_pt_YOUR_LANGCHAIN_KEY_HERE" | \
  gcloud secrets create LANGCHAIN_API_KEY --data-file=-
```

> Replace the placeholder values with your actual keys from your notebook's .env cell.

---

## STEP 6 — Deploy to Cloud Run

Run this single command (replace `YOUR_PROJECT_ID` with your Google Cloud project ID):

```bash
gcloud run deploy mcp-chatbot \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,TAVILY_API_KEY=TAVILY_API_KEY:latest,LANGCHAIN_API_KEY=LANGCHAIN_API_KEY:latest" \
  --set-env-vars="LANGCHAIN_TRACING_V2=true" \
  --project YOUR_PROJECT_ID
```

When it asks **"Allow unauthenticated invocations?"** → type `y`

This will:
1. Build your Docker container in the cloud (~3-5 mins)
2. Push it to Google Container Registry
3. Deploy it to Cloud Run
4. Give you a live URL ✅

---

## STEP 7 — Open Your Live Chatbot

At the end of the deploy command, you'll see:
```
Service URL: https://mcp-chatbot-abc123-uc.a.run.app
```

Open that URL in your browser — your MCP Chatbot is live! 🎉

---

## STEP 8 — Verify Everything is Working

Open `https://YOUR-URL/health` in your browser. You should see:
```json
{
  "status": "ok",
  "service": "MCP Chatbot",
  "agent": true,
  "rag_chain": true,
  "retriever": true,
  "model": "gpt-4o-mini"
}
```

If `"agent": false` — the container is still starting up (takes ~60 seconds on first request).

---

## Updating Your App

Whenever you change `main.py` or `static/index.html`, just push to GitHub and redeploy:

```bash
git add .
git commit -m "Update chatbot"
git push

# Redeploy
gcloud run deploy mcp-chatbot --source . --region us-central1
```

---

## Cost Estimate

Google Cloud Run pricing (as of 2025):
- **Free tier**: 2 million requests/month, 360,000 vCPU-seconds, 180,000 GB-seconds
- For a personal chatbot with light usage: **$0/month** (free tier is generous)
- You only pay when requests are actively being processed

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied` on secrets | Run: `gcloud projects add-iam-policy-binding YOUR_PROJECT --member="serviceAccount:..." --role="roles/secretmanager.secretAccessor"` |
| Container crashes on startup | Run: `gcloud run logs read --service mcp-chatbot --region us-central1` |
| `agent: false` in /health | First request triggers a cold start — wait 60s and refresh |
| API key errors | Double-check secrets: `gcloud secrets versions access latest --secret="OPENAI_API_KEY"` |

---

## Need Help?

If you get stuck on any step, copy the error message and ask — I can help debug it!
