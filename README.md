# Customer Support AI

A multi-layered customer support automation system with **defensive escalation routing**, built for an e-commerce scenario.

The system processes customer messages through three sequential safety layers — keyword filtering, AI classification with confidence thresholds, and RAG-based answer generation — before responding to the customer or routing to a human team.

## Why this architecture

Most customer support chatbots use a single LLM call with a system prompt like *"if you don't know, say so."* That approach has two problems: hallucination on edge cases, and no audit trail when things go wrong.

This system takes a different approach: **defense in depth**.

```
Customer message
       ↓
[Layer 1] Escalation Filter
   — Keyword scan (refund, legal, double charge, ...)
   — Behavioral triggers (ALL CAPS, multiple !!!)
       ↓
[Layer 2] Haiku Classifier
   — Categorizes into: shipping / billing / account / product / sales / order_change / general
   — Returns confidence score (0.0–1.0)
   — Below threshold → escalate
       ↓
[Layer 3] RAG Pipeline
   — Retrieves top-k relevant chunks from KB
   — Sonnet generates answer using ONLY retrieved context
   — Source attribution + confidence tracking
       ↓
Customer receives answer (or escalation message)
```

Each layer can independently route to human escalation. The customer never receives an unsafe answer.

## What's in the demo

- **6 markdown KB files** covering shipping, billing, account, product, sales, order changes (~2,000 words total)
- **9 escalation categories** with 50+ trigger keywords
- **2 behavioral triggers** (all-caps shouting, excessive punctuation)
- **5 Python modules** with full type hints, docstrings, and error handling
- **3-tier confidence checking** (filter → classifier → RAG)
- **Source attribution** in every AI-generated response

## Stack

- **Claude API** — Sonnet 4.5 for response generation, Haiku 4.5 for classification
- **ChromaDB** — In-memory vector database
- **Python 3.10+** — Plain SDK calls, no LangChain wrappers (intentional, for clarity)

## Project structure

```
customer-support-ai/
├── src/
│   ├── config.py              # Centralized settings
│   ├── escalation_filter.py   # Keyword + behavioral filter
│   ├── classifier.py          # Haiku categorization
│   ├── rag_pipeline.py        # Vector DB + Sonnet generation
│   └── main.py                # Orchestrator
├── config/
│   └── escalation_keywords.json
├── knowledge_base/
│   ├── shipping.md
│   ├── billing.md
│   ├── account.md
│   ├── product.md
│   ├── sales.md
│   └── order_change.md
├── tests/
│   └── test_questions.json    # 20 evaluation cases
├── requirements.txt
└── README.md
```

## Run it yourself

### In Google Colab (recommended)

1. Clone this repo or upload files to Colab
2. Install dependencies:
   ```python
   !pip install anthropic chromadb
   ```
3. Set your Anthropic API key:
   ```python
   import os
   from google.colab import userdata
   os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
   ```
4. Run the pipeline:
   ```python
   import sys
   sys.path.insert(0, "src")
   from main import setup_pipeline, process_message
   
   setup_pipeline()
   result = process_message("When will my order arrive?")
   print(result)
   ```

### Locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
cd src
python main.py
```

## Sample test cases

The `main.py` includes 5 test messages demonstrating each pipeline path:

| Message | Expected Action | Reason |
|---------|----------------|--------|
| "When will my order arrive?" | Answer | Shipping query, KB has answer |
| "I want to refund my order" | Escalate | `refund` keyword triggered |
| "asdfqwerty xyz blah" | Escalate | Low classification confidence |
| "How do I download my invoice?" | Answer | Billing query, KB has answer |
| "I want to speak to a real person" | Escalate | `human_request` keyword triggered |

## What this is not

- Not a production system — no conversation memory, no PII masking, no real-time observability
- Not multilingual — current setup is English only (multilingual embedding swap is straightforward)
- Not a LangChain project — intentionally built on raw SDK calls to make the architecture explicit

## What this is

A working proof of concept demonstrating **layered safety architecture** for customer support automation. The patterns here — keyword filtering before AI, confidence thresholds at every stage, source attribution in responses — are production patterns at smaller scale.

Built as a portfolio piece during my transition into AI Operations.

---

Built by [Burcu Tatlı](https://www.linkedin.com/in/burcutatli) · AI Operations & Automation  
Portfolio: [portfolioburcu.netlify.app](https://portfolioburcu.netlify.app)  
Related work: [clinical-safe-rag](https://github.com/burcutatli/clinical-safe-rag)
