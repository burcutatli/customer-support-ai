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
from pii_masker import PIIMasker, MaskingResult
from langfuse import observe, get_client


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
_pii_masker: PIIMasker | None = None

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
    
    # Initialize RAG pipeline (config loaded internally from config.py)
    _rag_pipeline = RAGPipeline()
    logger.info("RAGPipeline initialized")
    
    # Initialize PII masker (Microsoft Presidio)
    global _pii_masker
    _pii_masker = PIIMasker()
    logger.info("PIIMasker initialized")
    
    # Initialize Langfuse client (reads LANGFUSE_* env vars automatically)
    langfuse = get_client()
    if langfuse.auth_check():
        logger.info("Langfuse client authenticated")
    else:
        logger.warning("Langfuse auth failed — traces will not be sent")
    
    logger.info("Pipeline setup complete")


# ============================================================
# MAIN PIPELINE
# ============================================================

@observe(name="process_message")
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
    if _escalation_filter is None or _classifier is None or _rag_pipeline is None or _pii_masker is None:
        raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
    
    logger.info(f"Processing message: {message[:80]}...")
    
    # --- Step 0: Escalation filter on ORIGINAL message ---
    # Filter must see real customer tone (shouting, swearing) BEFORE PII masking.
    # Otherwise placeholders like <PERSON_1> trigger false-positive all_caps_shouting.
    # This bug was found in production with Turkish messages where spaCy NER
    # mis-classified "Siparişim" as PERSON, then the placeholder fooled the filter.
    should_escalate, reason, team = _escalation_filter.check(message)
    
    # --- Step 1: PII masking (only if not escalated) ---
    masking_result = _pii_masker.mask(message)
    masked_message = masking_result.masked_text
    pii_mapping = masking_result.mapping
    if pii_mapping:
        logger.info(f"PII masked: {len(pii_mapping)} entity(ies) replaced")
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
    classification = _classifier.classify(masked_message)
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
    rag_result = _rag_pipeline.answer_query(masked_message)
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
    
    # --- Step 6: Return successful answer (with PII restored) ---
    final_response = _pii_masker.unmask(rag_result["answer"], pii_mapping)
    return {
        "action": "answer",
        "category": classification["category"],
        "response_to_customer": final_response,
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
