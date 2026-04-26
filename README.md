# Customer Support AI

Production-grade customer support automation with **defense-in-depth** safety architecture, full observability, and HIPAA-aligned PII handling.

> **Live demo:** https://customer-support-ai-production-0259.up.railway.app/docs
> **Try it:** `POST /api/process` with `{"message": "When will my order arrive?"}`

Multi-layered pipeline that processes customer messages through five sequential safety layers before responding or escalating to a human team. Built as a portfolio piece during my transition into AI Operations.

---

## Architecture
Every layer is independently observable via Langfuse — input, output, latency, token usage, and **per-call USD cost** are tracked for every customer message.

---

## What's working in production

- ✅ **Public REST API** deployed on Railway (FastAPI + Uvicorn)
- ✅ **Persistent vector DB** (ChromaDB on disk, Cohere multilingual embeddings)
- ✅ **PII masking** with Microsoft Presidio (8 entity types, mask → process → unmask)
- ✅ **Observability** via Langfuse (trace + per-LLM cost tracking)
- ✅ **Multilingual** (English + Turkish queries via Cohere `embed-multilingual-v3`)
- ✅ **CI-friendly** (GitHub push → Railway auto-redeploy)

Cost per message in production: **~$0.003** (Sonnet 4.5 ~$0.0026 + Haiku 4.5 ~$0.0004), tracked in real time on Langfuse.

---

## Stack

| Layer | Tool |
|-------|------|
| **LLMs** | Anthropic Claude Sonnet 4.5 (RAG), Haiku 4.5 (classifier) |
| **Embeddings** | Cohere `embed-multilingual-v3` |
| **Vector DB** | ChromaDB (persistent on disk) |
| **PII detection** | Microsoft Presidio + spaCy `en_core_web_lg` |
| **Observability** | Langfuse Cloud (traces, token usage, USD cost) |
| **API** | FastAPI + Uvicorn |
| **Deploy** | Railway (Docker via Nixpacks, auto-deploy from GitHub) |
| **Language** | Python 3.11 |

No LangChain wrappers — intentionally built on raw SDKs to make the architecture explicit and debuggable.

---

## Project structure
---

## Run locally

```bash
git clone https://github.com/burcutatli/customer-support-ai.git
cd customer-support-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Required env vars in .env:
#   ANTHROPIC_API_KEY=sk-ant-...
#   COHERE_API_KEY=...
#   LANGFUSE_PUBLIC_KEY=pk-lf-...     (optional, for observability)
#   LANGFUSE_SECRET_KEY=sk-lf-...     (optional)
#   LANGFUSE_HOST=https://cloud.langfuse.com  (optional)

cd src
python api.py
# → http://localhost:8001/docs
```

---

## Try the live API

```bash
curl -X POST https://customer-support-ai-production-0259.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"message": "When will my order arrive?"}'
```

Sample paths through the pipeline:

| Message | Action | Why |
|---------|--------|-----|
| `When will my order arrive?` | answer | RAG hit on shipping KB (confidence ~0.72) |
| `Hi, my name is John Smith and my email is john@example.com` | answer | PII masked → LLM never sees real values |
| `Siparişim ne zaman gelir?` | escalate `low_confidence_rag` | Multilingual embedding works; KB is English-only |
| `I want to cancel my order` | escalate `cancel_keyword` | Layer 0 keyword trigger |
| `WHY IS MY ORDER STILL NOT HERE` | escalate `all_caps_shouting` | Behavioral trigger |
| `What's the weather today?` | escalate `low_confidence_rag` | Out-of-scope |

---

## Engineering decisions worth flagging

**Filter runs on the original message; PII masking happens after.**
Discovered in production: spaCy mis-classified the Turkish word "Siparişim" as a PERSON, the masker replaced it with `<PERSON_1>`, and the placeholder's all-caps letters tripped the shouting detector. Reordering the layers fixed the false positive without weakening either check.

**Per-LLM cost tracking, not just trace.**
Both `classifier.py` and `rag_pipeline.py` wrap their Anthropic calls with Langfuse `@observe(as_type="generation")` and report token usage. This makes per-message cost visible in real time and lets us tune confidence thresholds against actual spend, not estimates.

**Confidence-based escalation at every stage.**
A low-confidence answer is more dangerous than no answer. Each stage has its own threshold and escalation reason, so when a customer escalates we know *why* — not just that something went wrong.

**Multilingual support is partial by design.**
Cohere `embed-multilingual-v3` makes Turkish queries embeddable, but the knowledge base is English-only. The system correctly routes Turkish queries to a human via `low_confidence_rag` rather than hallucinating an English answer. Adding Turkish KB content is straightforward; auto-translating answers is not, and not done here.

---

## What this is not

- Not a chat bot — it's a **stateless message processor** designed to slot behind a Zendesk webhook, web chat widget, WhatsApp Business API, or similar front-end
- Not session-aware — no conversation memory; each request stands alone
- Not finished — Sprint 6 (Zendesk webhook integration) and Sprint 7 (compliance audit log) are next

---

Built by [Burcu Tatlı](https://www.linkedin.com/in/burcutatli) · AI Operations & Automation
Portfolio: [portfolioburcu.netlify.app](https://portfolioburcu.netlify.app)
Related work: [clinical-safe-rag](https://github.com/burcutatli/clinical-safe-rag)
