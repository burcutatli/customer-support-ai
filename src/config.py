"""
config.py — Centralized configuration for Customer Support AI system.

This file contains all settings, constants, and paths used across the
application. All other modules import from here. Never hardcode settings
in other files — change them here.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-25
"""

import os
from pathlib import Path

# ============================================================
# API CONFIGURATION
# ============================================================

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Model selection — Haiku for fast classification, Sonnet for quality answers
CLASSIFIER_MODEL: str = "claude-haiku-4-5"
RESPONDER_MODEL: str = "claude-sonnet-4-5"

# Token limits per model
CLASSIFIER_MAX_TOKENS: int = 100
RESPONDER_MAX_TOKENS: int = 500

# ============================================================
# RAG CONFIGURATION
# ============================================================

# Project root and data paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
KB_DIRECTORY: Path = PROJECT_ROOT / "knowledge_base"
ESCALATION_KEYWORDS_PATH: Path = PROJECT_ROOT / "config" / "escalation_keywords.json"

# Vector DB settings (using Chroma's default embedding for MVP)
VECTOR_DB_NAME: str = "customer_support_kb"
TOP_K_RETRIEVAL: int = 3

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

# Below this threshold, AI escalates to human instead of answering
MIN_CONFIDENCE_TO_ANSWER: float = 0.65

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_LEVEL: str = "INFO"
LOG_FILE_PATH: Path = PROJECT_ROOT / "logs" / "app.log"

# ============================================================
# VALID CATEGORIES
# ============================================================

# Categories the classifier is allowed to return
VALID_CATEGORIES: list[str] = [
    "shipping",
    "billing",
    "account",
    "product",
    "sales",
    "order_change",
    "general",
]

# ============================================================
# SPRINT 2: PERSISTENT VECTOR DB + COHERE EMBEDDING
# ============================================================

COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

# Persistent Chroma DB path (created on first run, reused after)
CHROMA_PATH: Path = PROJECT_ROOT / "chroma_db"

# Cohere multilingual embedding model
EMBEDDING_MODEL: str = "embed-multilingual-v3.0"

# Aliases used by rag_pipeline.py (Sprint 2 naming)
KB_DIR: Path = KB_DIRECTORY
COLLECTION_NAME: str = VECTOR_DB_NAME
TOP_K: int = TOP_K_RETRIEVAL
GENERATION_MODEL: str = RESPONDER_MODEL

# Chunk settings for RAG indexing
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# System prompt for answer generation
SYSTEM_PROMPT: str = """You are a friendly customer support AI for an e-commerce company.
Answer customer questions clearly using only the knowledge base context provided.
Be warm, professional, and concise. If the context doesn't fully answer the question,
acknowledge what you can address and suggest the customer share more details."""
