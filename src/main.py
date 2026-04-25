"""
main.py — Customer Support AI orchestrator.

This is the entry point for the AI pipeline. It wires together:
- EscalationFilter (keyword + behavioral filter)
- MessageClassifier (Haiku-based categorization)
- RAGPipeline (KB retrieval + Sonnet generation)

Usage:
    from main import setup_pipeline, process_message
    
    setup_pipeline()
    result = process_message("When will my order arrive?")

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-25
"""

import logging
import sys

from config import (
    ANTHROPIC_API_KEY,
    CLASSIFIER_MODEL,
    CLASSIFIER_MAX_TOKENS,
    RESPONDER_MODEL,
    RESPONDER_MAX_TOKENS,
    KB_DIRECTORY,
    ESCALATION_KEYWORDS_PATH,
    TOP_K_RETRIEVAL,
    MIN_CONFIDENCE_TO_ANSWER,
    LOG_LEVEL,
    LOG_FILE_PATH,
    VALID_CATEGORIES,
    VECTOR_DB_NAME,
)
from escalation_filter import EscalationFilter
from classifier import MessageClassifier
from rag_pipeline import RAGPipeline


# ============================================================
# CONSTANTS
# ============================================================

ESCALATION_RESPONSE_TEMPLATE = (
    "I understand your concern. Let me connect you with a team member "
    "who can help you better. They'll get back to you shortly.\n\n"
    "In the meantime, if you'd like to share your order number or account "
    "details to speed up the process, please feel free to do so."
)

# Module-level component holders
_escalation_filter: EscalationFilter | None = None
_classifier: MessageClassifier | None = None
_rag_pipeline: RAGPipeline | None = None

logger = logging.getLogger(__name__)


# ============================================================
# SETUP
# ============================================================

def setup_pipeline() -> None:
    """
    One-time setup: create directories, configure logging, initialize all components.
    
    Raises:
        ValueError: If ANTHROPIC_API_KEY is missing
        FileNotFoundError: If escalation keywords or KB directory missing
    """
    global _escalation_filter, _classifier, _rag_pipeline
    
    # Validate API key
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it before running: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    
    # Create required directories
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ESCALATION_KEYWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger.info("Starting Customer Support AI pipeline setup")
    
    # Initialize escalation filter
    _escalation_filter = EscalationFilter(keywords_path=ESCALATION_KEYWORDS_PATH)
    logger.info("EscalationFilter initialized")
    
    # Initialize classifier
    _classifier = MessageClassifier(
        api_key=ANTHROPIC_API_KEY,
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        valid_categories=VALID_CATEGORIES,
    )
    logger.info("MessageClassifier initialized")
    
    # Initialize RAG pipeline
    _rag_pipeline = RAGPipeline(
        kb_directory=KB_DIRECTORY,
        anthropic_api_key=ANTHROPIC_API_KEY,
        model=RESPONDER_MODEL,
        max_tokens=RESPONDER_MAX_TOKENS,
        top_k=TOP_K_RETRIEVAL,
        collection_name=VECTOR_DB_NAME,
    )
    chunks_indexed = _rag_pipeline.index_knowledge_base()
    logger.info(f"RAGPipeline initialized — {chunks_indexed} chunks indexed")
    
    logger.info("Pipeline setup complete")


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_message(message: str) -> dict:
    """
    Process a customer message through the full AI pipeline.
    
    Pipeline:
        1. Escalation keyword filter
        2. Message classification
        3. Confidence check on classification
        4. RAG retrieval and answer generation
        5. Confidence check on RAG output
        6. Return action + response
    
    Args:
        message: Raw customer message text
    
    Returns:
        Dict with keys:
        - action: "answer" or "escalate"
        - response_to_customer: text to send to customer
        - reason: reason for escalation (if applicable)
        - metadata: detailed pipeline info for logging
    """
    if _escalation_filter is None or _classifier is None or _rag_pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
    
    logger.info(f"Processing message: {message[:80]}...")
    
    # --- Step 1: Escalation keyword filter ---
    should_escalate, reason, team = _escalation_filter.check(message)
    if should_escalate:
        logger.info(f"Escalating via filter: {reason} → {team}")
        return {
            "action": "escalate",
            "reason": reason,
            "destination_team": team,
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "metadata": {"stage": "escalation_filter"},
        }
    
    # --- Step 2: Classify message ---
    classification = _classifier.classify(message)
    logger.info(
        f"Classification: {classification['category']} "
        f"(confidence: {classification['confidence']:.2f})"
    )
    
    # --- Step 3: Confidence check on classification ---
    if classification["confidence"] < MIN_CONFIDENCE_TO_ANSWER:
        logger.info("Escalating: low classification confidence")
        return {
            "action": "escalate",
            "reason": "low_confidence_classification",
            "destination_team": "support_general",
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "metadata": {
                "stage": "classifier",
                "classification": classification,
            },
        }
    
    # --- Step 4: RAG generation ---
    rag_result = _rag_pipeline.generate_answer(
        query=message,
        category=classification["category"],
    )
    logger.info(f"RAG generated answer (confidence: {rag_result['confidence']:.2f})")
    
    # --- Step 5: Confidence check on RAG ---
    if rag_result["confidence"] < MIN_CONFIDENCE_TO_ANSWER:
        logger.info("Escalating: low RAG confidence")
        return {
            "action": "escalate",
            "reason": "low_confidence_rag",
            "destination_team": "support_general",
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "metadata": {
                "stage": "rag",
                "classification": classification,
                "rag": rag_result,
            },
        }
    
    # --- Step 6: Return successful answer ---
    return {
        "action": "answer",
        "category": classification["category"],
        "response_to_customer": rag_result["answer"],
        "metadata": {
            "stage": "answered",
            "classification": classification,
            "rag_confidence": rag_result["confidence"],
            "retrieved_chunks": rag_result["retrieved_chunks"],
        },
    }


# ============================================================
# TESTING / ENTRY POINT
# ============================================================

def _print_result(message: str, result: dict) -> None:
    """Pretty-print a pipeline result for testing."""
    print("=" * 70)
    print(f"MESSAGE: {message}")
    print(f"ACTION:  {result['action'].upper()}")
    
    if result["action"] == "escalate":
        print(f"REASON:  {result['reason']}")
        print(f"ROUTE:   {result['destination_team']}")
    else:
        print(f"CATEGORY: {result.get('category', 'N/A')}")
    
    print(f"\nRESPONSE TO CUSTOMER:\n{result['response_to_customer']}")
    print()


if __name__ == "__main__":
    setup_pipeline()
    
    test_messages = [
        "When will my order arrive?",                 # → answer (shipping)
        "I want to refund my order",                  # → escalate (refund keyword)
        "asdfqwerty xyz blah",                        # → escalate (low confidence)
        "How do I download my invoice?",              # → answer (billing)
        "I want to speak to a real person",           # → escalate (human request)
    ]
    
    for msg in test_messages:
        result = process_message(msg)
        _print_result(msg, result)
