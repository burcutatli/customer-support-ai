"""
audit_logger.py — Compliance-grade audit logging for Customer Support AI.

Writes structured, append-only JSON Lines (JSONL) records for every
process_message() call. Each line is a self-contained event with:
  - timestamp (UTC, ISO 8601)
  - event_id (UUID for cross-reference)
  - action (answer / escalate)
  - reason (if escalated)
  - ticket_id (Zendesk reference, if created)
  - message_hash (SHA256 of MASKED message — original PII never logged)
  - confidence scores (classifier + RAG)

Why JSONL?
  - Append-only by design (no in-place edits → tamper-evident)
  - One line per event → trivial to grep, parse, ship to Datadog/Splunk
  - JSON-typed → no CSV escaping nightmares
  - Each line independently parseable → corrupt one, others survive

Why hash the message instead of storing it?
  - GDPR: storing customer messages = data retention obligation
  - Hash = "did this exact message come through?" without exposing content
  - Cross-reference via ticket_id if full text needed (Zendesk has it)

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-27
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# AUDIT LOGGER
# ============================================================

class AuditLogger:
    """
    Append-only structured JSON audit logger.
    
    Designed for compliance use cases (GDPR, HIPAA-adjacent, GxP):
    - Stores hash of message, not raw text
    - Each event has unique UUID for cross-reference
    - Append-only file (one JSON object per line)
    - Timezone-aware UTC timestamps
    
    Usage:
        audit = AuditLogger(log_path=Path("logs/audit.jsonl"))
        audit.log_event(
            action="escalate",
            masked_message="<PERSON_1> wants a refund",
            reason="refund_keyword",
            ticket_id=42,
        )
    """
    
    def __init__(self, log_path: Path) -> None:
        """
        Initialize audit logger.
        
        Args:
            log_path: Where to write the JSONL file. Parent dir is auto-created.
        """
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Touch the file so we can detect "no events yet" vs "missing file"
        if not self.log_path.exists():
            self.log_path.touch()
        
        logger.info(f"AuditLogger initialized at {self.log_path}")
    
    def log_event(
        self,
        action: str,
        masked_message: str,
        reason: Optional[str] = None,
        destination_team: Optional[str] = None,
        ticket_id: Optional[int] = None,
        ticket_created: Optional[bool] = None,
        category: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        rag_confidence: Optional[float] = None,
        pii_entities_detected: int = 0,
    ) -> str:
        """
        Append an audit event to the log.
        
        Args:
            action: "answer" or "escalate"
            masked_message: PII-masked message (NEVER raw)
            reason: Escalation reason (if action == "escalate")
            destination_team: Team routing (if escalated)
            ticket_id: Zendesk ticket ID (if created)
            ticket_created: Whether Zendesk ticket creation succeeded
            category: Classifier category (if available)
            classification_confidence: Classifier confidence 0..1
            rag_confidence: RAG confidence 0..1
            pii_entities_detected: How many PII entities were masked
        
        Returns:
            event_id (UUID) — useful for cross-reference debugging
        """
        event_id = str(uuid.uuid4())
        
        # Hash the masked message for compliance (no raw text in log)
        message_hash = hashlib.sha256(
            masked_message.encode("utf-8")
        ).hexdigest()[:16]  # First 16 chars enough for dedup
        
        event = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "message_hash": message_hash,
            "message_length": len(masked_message),
            "pii_entities_detected": pii_entities_detected,
        }
        
        # Add optional fields only if provided (clean JSON)
        if reason is not None:
            event["reason"] = reason
        if destination_team is not None:
            event["destination_team"] = destination_team
        if ticket_id is not None:
            event["ticket_id"] = ticket_id
        if ticket_created is not None:
            event["ticket_created"] = ticket_created
        if category is not None:
            event["category"] = category
        if classification_confidence is not None:
            event["classification_confidence"] = round(classification_confidence, 3)
        if rag_confidence is not None:
            event["rag_confidence"] = round(rag_confidence, 3)
        
        # Append-only write (one event per line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        
        logger.info(
            f"Audit event logged: {event_id[:8]} action={action} "
            f"reason={reason} ticket_id={ticket_id}"
        )
        
        return event_id
    
    def read_recent(self, n: int = 10) -> list[dict]:
        """
        Read the last N audit events. For debugging / endpoints only.
        
        Args:
            n: Max number of events to return
        
        Returns:
            List of event dicts (most recent first)
        """
        if not self.log_path.exists():
            return []
        
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Take last N, parse, reverse so most recent first
        recent_lines = lines[-n:]
        events = []
        for line in recent_lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted audit log line skipped: {e}")
        
        return list(reversed(events))
    
    def count_events(self) -> int:
        """Return total number of audit events on disk."""
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    """Run directly to verify AuditLogger works."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    print("=" * 60)
    print("AuditLogger standalone test")
    print("=" * 60)
    
    # Use a temp file so we don't pollute real audit log
    test_path = Path("logs/audit_test.jsonl")
    if test_path.exists():
        test_path.unlink()  # Clean slate
    
    audit = AuditLogger(log_path=test_path)
    
    # Test 1: answer event
    print("\n[1/3] Logging an 'answer' event...")
    eid1 = audit.log_event(
        action="answer",
        masked_message="When will my <PERSON_1> arrive?",
        category="shipping",
        classification_confidence=0.95,
        rag_confidence=0.78,
        pii_entities_detected=1,
    )
    print(f"   ✅ event_id: {eid1}")
    
    # Test 2: escalate event with ticket
    print("\n[2/3] Logging an 'escalate' event with ticket...")
    eid2 = audit.log_event(
        action="escalate",
        masked_message="I want a refund right now",
        reason="refund_keyword",
        destination_team="refund_team",
        ticket_id=999,
        ticket_created=True,
        pii_entities_detected=0,
    )
    print(f"   ✅ event_id: {eid2}")
    
    # Test 3: read back
    print("\n[3/3] Reading recent events...")
    recent = audit.read_recent(n=5)
    print(f"   Total events: {audit.count_events()}")
    for i, evt in enumerate(recent, 1):
        print(f"   [{i}] {evt['action']:10} | {evt.get('reason', '-'):25} | hash={evt['message_hash']}")
    
    # Cleanup
    test_path.unlink()
    print("\n" + "=" * 60)
    print("✅ All AuditLogger tests passed")
    print("=" * 60)
