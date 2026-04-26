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
MIN_CONFIDENCE_TO_ANSWER: float = 0.30

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
