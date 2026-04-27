"""
main.py — Customer Support AI orchestrator.

This is the entry point for the AI pipeline. It wires together:
- EscalationFilter (keyword + behavioral filter)
- MessageClassifier (Haiku-based categorization)
- RAGPipeline (KB retrieval + Sonnet generation)
- ZendeskClient (creates support tickets when escalating)
- AuditLogger (compliance-grade JSONL audit trail)

Author: Burcu Tatli (AI Operations)
Last updated: 2026-04-27
"""

import logging
import sys
from pathlib import Path
from typing import Optional

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
from zendesk_client import ZendeskClient, TicketResult
from audit_logger import AuditLogger
from langfuse import observe, get_client


ESCALATION_RESPONSE_TEMPLATE = (
    "I understand your concern. Let me connect you with a team member "
    "who can help you better. They'll get back to you shortly.\n\n"
    "In the meantime, if you'd like to share your order number or account "
    "details to speed up the process, please feel free to do so."
)

ESCALATION_PRIORITY_MAP = {
    "refund_keyword": "high",
    "complaint_keyword": "high",
    "human_request_keyword": "normal",
    "billing_keyword": "normal",
    "technical_keyword": "normal",
    "all_caps_shouting": "high",
    "multiple_exclamations": "normal",
    "low_confidence_classification": "normal",
    "low_confidence_rag": "normal",
}

AUDIT_LOG_PATH = Path("logs/audit.jsonl")

_escalation_filter: EscalationFilter | None = None
_classifier: MessageClassifier | None = None
_rag_pipeline: RAGPipeline | None = None
_pii_masker: PIIMasker | None = None
_zendesk_client: ZendeskClient | None = None
_audit_logger: AuditLogger | None = None

logger = logging.getLogger(__name__)


def setup_pipeline() -> None:
    global _escalation_filter, _classifier, _rag_pipeline, _zendesk_client, _audit_logger

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ESCALATION_KEYWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Starting Customer Support AI pipeline setup")

    _escalation_filter = EscalationFilter(keywords_path=ESCALATION_KEYWORDS_PATH)
    logger.info("EscalationFilter initialized")

    _classifier = MessageClassifier(
        api_key=ANTHROPIC_API_KEY,
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        valid_categories=VALID_CATEGORIES,
    )
    logger.info("MessageClassifier initialized")

    _rag_pipeline = RAGPipeline()
    logger.info("RAGPipeline initialized")

    global _pii_masker
    _pii_masker = PIIMasker()
    logger.info("PIIMasker initialized")

    try:
        _zendesk_client = ZendeskClient()
        logger.info("ZendeskClient initialized")
    except ValueError as e:
        logger.warning(f"ZendeskClient NOT initialized: {e}")
        _zendesk_client = None

    _audit_logger = AuditLogger(log_path=AUDIT_LOG_PATH)
    logger.info("AuditLogger initialized")

    langfuse = get_client()
    if langfuse.auth_check():
        logger.info("Langfuse client authenticated")
    else:
        logger.warning("Langfuse auth failed")

    logger.info("Pipeline setup complete")


def _create_zendesk_ticket(
    customer_message: str,
    reason: str,
    destination_team: str,
    classification: Optional[dict] = None,
    rag_result: Optional[dict] = None,
) -> TicketResult:
    if _zendesk_client is None:
        logger.info("Zendesk client unavailable — skipping ticket creation")
        return TicketResult(success=False, error_message="Zendesk client not configured")

    short_msg = customer_message[:60] + ("..." if len(customer_message) > 60 else "")
    subject = f"[AI escalation] {short_msg}"

    description_parts = [
        f"Original customer message:\n{customer_message}",
        f"\nEscalation reason: {reason}",
        f"Suggested team: {destination_team}",
    ]

    if classification:
        description_parts.append(
            f"\nClassifier output:\n"
            f"  - Category: {classification.get('category', 'N/A')}\n"
            f"  - Confidence: {classification.get('confidence', 0):.2f}"
        )

    if rag_result:
        description_parts.append(
            f"\nRAG attempt:\n"
            f"  - Answer drafted (low confidence): {rag_result.get('answer', '')[:200]}\n"
            f"  - RAG confidence: {rag_result.get('confidence', 0):.2f}"
        )

    description = "\n".join(description_parts)
    priority = ESCALATION_PRIORITY_MAP.get(reason, "normal")
    tags = ["ai_escalation", f"reason_{reason}", f"team_{destination_team}"]

    return _zendesk_client.create_ticket(
        subject=subject,
        description=description,
        priority=priority,
        tags=tags,
    )


def _audit(
    action: str,
    masked_message: str,
    pii_count: int,
    reason: Optional[str] = None,
    destination_team: Optional[str] = None,
    ticket_id: Optional[int] = None,
    ticket_created: Optional[bool] = None,
    category: Optional[str] = None,
    classification_confidence: Optional[float] = None,
    rag_confidence: Optional[float] = None,
) -> None:
    if _audit_logger is None:
        return
    try:
        _audit_logger.log_event(
            action=action,
            masked_message=masked_message,
            reason=reason,
            destination_team=destination_team,
            ticket_id=ticket_id,
            ticket_created=ticket_created,
            category=category,
            classification_confidence=classification_confidence,
            rag_confidence=rag_confidence,
            pii_entities_detected=pii_count,
        )
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")


@observe(name="process_message")
def process_message(message: str) -> dict:
    if _escalation_filter is None or _classifier is None or _rag_pipeline is None or _pii_masker is None:
        raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")

    logger.info(f"Processing message: {message[:80]}...")

    should_escalate, reason, team = _escalation_filter.check(message)

    masking_result = _pii_masker.mask(message)
    masked_message = masking_result.masked_text
    pii_mapping = masking_result.mapping
    pii_count = len(pii_mapping) if pii_mapping else 0
    if pii_mapping:
        logger.info(f"PII masked: {pii_count} entity(ies) replaced")

    if should_escalate:
        logger.info(f"Escalating via filter: {reason} -> {team}")
        ticket = _create_zendesk_ticket(
            customer_message=message,
            reason=reason,
            destination_team=team,
        )
        _audit(
            action="escalate",
            masked_message=masked_message,
            pii_count=pii_count,
            reason=reason,
            destination_team=team,
            ticket_id=ticket.ticket_id,
            ticket_created=ticket.success,
        )
        return {
            "action": "escalate",
            "reason": reason,
            "destination_team": team,
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "ticket_id": ticket.ticket_id,
            "ticket_created": ticket.success,
            "metadata": {"stage": "escalation_filter"},
        }

    classification = _classifier.classify(masked_message)
    logger.info(f"Classification: {classification['category']} (confidence: {classification['confidence']:.2f})")

    if classification["confidence"] < MIN_CONFIDENCE_TO_ANSWER:
        logger.info("Escalating: low classification confidence")
        ticket = _create_zendesk_ticket(
            customer_message=message,
            reason="low_confidence_classification",
            destination_team="support_general",
            classification=classification,
        )
        _audit(
            action="escalate",
            masked_message=masked_message,
            pii_count=pii_count,
            reason="low_confidence_classification",
            destination_team="support_general",
            ticket_id=ticket.ticket_id,
            ticket_created=ticket.success,
            category=classification["category"],
            classification_confidence=classification["confidence"],
        )
        return {
            "action": "escalate",
            "reason": "low_confidence_classification",
            "destination_team": "support_general",
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "ticket_id": ticket.ticket_id,
            "ticket_created": ticket.success,
            "metadata": {"stage": "classifier", "classification": classification},
        }

    rag_result = _rag_pipeline.answer_query(masked_message)
    logger.info(f"RAG generated answer (confidence: {rag_result['confidence']:.2f})")

    if rag_result["confidence"] < MIN_CONFIDENCE_TO_ANSWER:
        logger.info("Escalating: low RAG confidence")
        ticket = _create_zendesk_ticket(
            customer_message=message,
            reason="low_confidence_rag",
            destination_team="support_general",
            classification=classification,
            rag_result=rag_result,
        )
        _audit(
            action="escalate",
            masked_message=masked_message,
            pii_count=pii_count,
            reason="low_confidence_rag",
            destination_team="support_general",
            ticket_id=ticket.ticket_id,
            ticket_created=ticket.success,
            category=classification["category"],
            classification_confidence=classification["confidence"],
            rag_confidence=rag_result["confidence"],
        )
        return {
            "action": "escalate",
            "reason": "low_confidence_rag",
            "destination_team": "support_general",
            "response_to_customer": ESCALATION_RESPONSE_TEMPLATE,
            "ticket_id": ticket.ticket_id,
            "ticket_created": ticket.success,
            "metadata": {"stage": "rag", "classification": classification, "rag": rag_result},
        }

    final_response = _pii_masker.unmask(rag_result["answer"], pii_mapping)
    _audit(
        action="answer",
        masked_message=masked_message,
        pii_count=pii_count,
        category=classification["category"],
        classification_confidence=classification["confidence"],
        rag_confidence=rag_result["confidence"],
    )
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


def _print_result(message: str, result: dict) -> None:
    print("=" * 70)
    print(f"MESSAGE: {message}")
    print(f"ACTION:  {result['action'].upper()}")
    if result["action"] == "escalate":
        print(f"REASON:  {result['reason']}")
        print(f"ROUTE:   {result['destination_team']}")
        if result.get("ticket_created"):
            print(f"TICKET:  #{result['ticket_id']} created in Zendesk")
        else:
            print(f"TICKET:  not created")
    else:
        print(f"CATEGORY: {result.get('category', 'N/A')}")
    print(f"\nRESPONSE TO CUSTOMER:\n{result['response_to_customer']}")
    print()


if __name__ == "__main__":
    setup_pipeline()
    test_messages = [
        "When will my order arrive?",
        "I want to refund my order",
        "asdfqwerty xyz blah",
        "How do I download my invoice?",
        "I want to speak to a real person",
    ]
    for msg in test_messages:
        result = process_message(msg)
        _print_result(msg, result)

    if _audit_logger is not None:
        print("=" * 70)
        print(f"AUDIT LOG: {_audit_logger.count_events()} total events")
        print("Recent events:")
        for evt in _audit_logger.read_recent(n=5):
            print(f"  [{evt['timestamp']}] {evt['action']:10} reason={evt.get('reason', '-')} ticket={evt.get('ticket_id', '-')}")
