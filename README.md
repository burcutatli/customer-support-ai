# Customer Support AI

Production-grade customer support automation with **defense-in-depth** safety architecture, full observability, HIPAA-aligned PII handling, and compliance-grade audit logging.

> **Live demo:** https://customer-support-ai-production-0259.up.railway.app/docs
> **Try it:** `POST /api/process` with `{"message": "When will my order arrive?"}`

Multi-layered pipeline that processes customer messages through six sequential safety layers before responding or routing to a human team via Zendesk. Built as a portfolio piece during my transition into AI Operations.

---

## Architecture

```
                   Customer message (any channel)
                              │
                              ▼
                ┌────────────────────────────┐
                │  0. Escalation filter      │  ← keywords, all-caps, !!?
                │     (on raw message)       │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  1. PII masking            │  ← Presidio: PERSON, EMAIL,
                │     (Presidio + spaCy)     │     PHONE, CREDIT_CARD, ...
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  2. Classifier (Haiku 4.5) │  ← shipping/billing/refund/...
                │     w/ confidence score    │     w/ confidence cutoff
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  3. RAG (Sonnet 4.5)       │  ← ChromaDB + Cohere embed
                │     w/ confidence score    │     top_k retrieval
                └─────────────┬──────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
         answer (PII restored)        escalate
                │                           │
                ▼                           ▼
         Customer reply              ┌────────────────────┐
                                     │ Zendesk ticket     │
                                     │ (priority + tags)  │
                                     └────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  Audit log (JSONL)         │  ← every call: hashed msg,
                │  GDPR-friendly hash        │     UUID, ticket_id, scores
                └────────────────────────────┘
```

Every layer is independently observable via **Langfuse** — input, output, latency, token usage, and **per-call USD cost** are tracked for every customer message.

---

## What's working in production

- ✅ **Public REST API** deployed on Railway (FastAPI + Uvicorn)
- ✅ **Persistent vector DB** (ChromaDB on disk, Cohere multilingual embeddings)
- ✅ **PII masking** with Microsoft Presidio (8 entity types, mask → process → unmask)
- ✅ **Zendesk integration** — escalations create real support tickets with smart priority routing and structured tags _(Sprint 6)_
- ✅ **Compliance-grade audit log** — append-only JSONL with hashed messages, UUID cross-reference, and ticket linkage _(Sprint 7)_
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
| **CRM / Ticketing** | Zendesk REST API (httpx-based client) |
| **Audit log** | JSONL (append-only, SHA256 hashed) |
| **Observability** | Langfuse Cloud (traces, token usage, USD cost) |
| **API** | FastAPI + Uvicorn |
| **Deploy** | Railway (Docker via Nixpacks, auto-deploy from GitHub) |
| **Language** | Python 3.11 |

No LangChain wrappers — intentionally built on raw SDKs to make the architecture explicit and debuggable.

---

## Sprint 6 highlights — Zendesk escalation

When the AI escalates (keyword filter, low classification confidence, or low RAG confidence), it does not just return a generic message — it creates a real Zendesk ticket with full context for the human agent.

- **Smart priority mapping**: refund/complaint keywords → `high`, others → `normal`
- **Structured tags**: `ai_escalation`, `reason_<reason>`, `team_<team>` — filter and report directly in Zendesk
- **Rich descriptions**: original message + escalation reason + classifier output + RAG attempt
- **Graceful degradation**: if Zendesk credentials are missing, the pipeline still serves customers; only ticket creation is skipped (logged as warning)

API response includes `ticket_id` and `ticket_created` so callers can track the ticket without a second round-trip.

---

## Sprint 7 highlights — compliance audit log

Pharma and healthcare-adjacent AI deployments require regulatory audit trails (GDPR Article 30, HIPAA-adjacent, GxP). Sprint 7 adds the foundational logging layer.

- **GDPR-friendly**: SHA256 hash of the masked message — raw customer text never hits disk
- **Tamper-evident**: append-only JSONL writes; one event per line
- **Cross-reference**: every event has a UUID; escalation events carry the Zendesk `ticket_id`
- **Structured fields**: action, reason, category, classification_confidence, rag_confidence, pii_entities_detected
- **Resilient**: audit-write failures are caught and logged — they never break a customer request

Sample event:

```json
{
  "event_id": "11469a28-66a5-4585-a85a-3e40e0d860e4",
  "timestamp": "2026-04-27T12:53:37.184989+00:00",
  "action": "escalate",
  "message_hash": "68d8a742747acbf8",
  "message_length": 25,
  "pii_entities_detected": 0,
  "reason": "refund_keyword",
  "destination_team": "refund_team",
  "ticket_id": 11,
  "ticket_created": true
}
```

> **Sprint 8 (next):** Audit log persistence to durable storage (S3/CloudWatch). Railway containers are ephemeral, so the current JSONL works for development and short-lived runs but is not yet compliance-grade for retention.

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
#   ZENDESK_SUBDOMAIN=your-subdomain  (optional, for ticket creation)
#   ZENDESK_EMAIL=you@example.com     (optional)
#   ZENDESK_API_TOKEN=...             (optional)

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

| Message | Action | Why | Ticket |
|---------|--------|-----|--------|
| `When will my order arrive?` | answer | RAG hit on shipping KB (confidence ~0.72) | — |
| `Hi, my name is John Smith and my email is john@example.com` | answer | PII masked → LLM never sees real values | — |
| `Siparişim ne zaman gelir?` | escalate `low_confidence_rag` | Multilingual embedding works; KB is English-only | normal |
| `I want to refund my order` | escalate `refund_keyword` | Layer 0 keyword trigger | **high** |
| `WHY IS MY ORDER STILL NOT HERE` | escalate `all_caps_shouting` | Behavioral trigger | high |
| `What's the weather today?` | escalate `low_confidence_rag` | Out-of-scope | normal |

Every escalate path also writes an audit event with the ticket_id, so any production incident can be traced back to a Zendesk ticket and a hashed message fingerprint.

---

## Engineering decisions worth flagging

**Filter runs on the original message; PII masking happens after.**
Discovered in production: spaCy mis-classified the Turkish word "Siparişim" as a PERSON, the masker replaced it with `<PERSON_1>`, and the placeholder's all-caps letters tripped the shouting detector. Reordering the layers fixed the false positive without weakening either check.

**Per-LLM cost tracking, not just trace.**
Both `classifier.py` and `rag_pipeline.py` wrap their Anthropic calls with Langfuse `@observe(as_type="generation")` and report token usage. This makes per-message cost visible in real time and lets us tune confidence thresholds against actual spend, not estimates.

**Confidence-based escalation at every stage.**
A low-confidence answer is more dangerous than no answer. Each stage has its own threshold and escalation reason, so when a customer escalates we know *why* — not just that something went wrong.

**Multilingual support is partial by design.**
Cohere `embed-multilingual-v3` makes Turkish queries embeddable, but the knowledge base is English-only. The system correctly routes Turkish queries to a human via `low_confidence_rag` rather than hallucinating an English answer.

**Audit log hashes the masked message, not the raw one.**
The masked message already has PII placeholders (`<PERSON_1>`, `<EMAIL_1>`, ...). Hashing this still gives a stable fingerprint for deduplication and incident replay, but ensures customer PII never enters the audit pipeline at all — a stronger guarantee than hashing the raw text.

**Graceful degradation for every external dependency.**
If Zendesk is down or misconfigured, the pipeline serves the customer and logs a warning. If Langfuse auth fails, traces are skipped but answers still flow. If the audit log write fails, the request still completes. No single external service can take the bot down.

---

## What this is not

- Not a chat bot — it's a **stateless message processor** designed to slot behind a Zendesk webhook, web chat widget, WhatsApp Business API, or similar front-end
- Not session-aware — no conversation memory; each request stands alone
- Not finished — Sprint 8 (audit log persistence to S3/CloudWatch) is next

---

## Roadmap

| Sprint | Status | Highlights |
|--------|--------|------------|
| 1–4 | ✅ Complete | RAG pipeline, classifier, escalation filter, FastAPI |
| 4.1 | ✅ Complete | Per-LLM cost tracking via Langfuse |
| 5 | ✅ Complete | PII masking with Microsoft Presidio |
| 5.1 | ✅ Complete | Turkish bug fix — filter ordering |
| **6** | ✅ Complete | **Zendesk integration with smart priority and tags** |
| **7** | ✅ Complete | **Compliance-grade audit logging (JSONL + hashing)** |
| 8 | ⏳ Planned | Audit log persistence (S3/CloudWatch volume mount) |

---

Built by [Burcu Tatlı](https://www.linkedin.com/in/burcutatli) · AI Operations & Automation
Portfolio: [portfolioburcu.netlify.app](https://portfolioburcu.netlify.app)
Related work: [clinical-safe-rag](https://github.com/burcutatli/clinical-safe-rag)
