"""
pii_masker.py — Microsoft Presidio-based PII detection and masking.

Workflow:
1. Detect PII entities in customer message (PERSON, EMAIL, PHONE, etc.)
2. Replace each entity with a placeholder (e.g. <PERSON_1>) — keep mapping
3. LLM processes masked message (no real PII reaches Anthropic logs)
4. After LLM responds, restore original values from mapping for the customer

This implements the "minimum disclosure" principle: external services receive
only what they absolutely need to answer. Critical for HIPAA/GDPR compliance.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-26
"""

import logging
from dataclasses import dataclass
from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)


# Entities we want to mask. ORDER MATTERS — longer-named first to avoid overlap.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "LOCATION",
    "IP_ADDRESS",
]

# Confidence threshold — Presidio scores below this are ignored
DEFAULT_SCORE_THRESHOLD = 0.5


@dataclass
class MaskingResult:
    """Result of masking a piece of text."""
    masked_text: str
    mapping: dict  # placeholder -> original value, e.g. {"<PERSON_1>": "Burcu"}
    entities_found: list  # list of detected entity types


class PIIMasker:
    """Detect and mask PII in customer messages, then restore in responses."""

    def __init__(
        self,
        entities: list[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        language: str = "en",
    ):
        self.analyzer = AnalyzerEngine()
        self.entities = entities or DEFAULT_ENTITIES
        self.score_threshold = score_threshold
        self.language = language
        logger.info(
            f"PIIMasker initialized "
            f"(entities={len(self.entities)}, threshold={score_threshold})"
        )

    def mask(self, text: str) -> MaskingResult:
        """Replace PII in text with placeholders. Returns masked text and mapping."""
        if not text or not text.strip():
            return MaskingResult(masked_text=text, mapping={}, entities_found=[])

        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )

        # Sort by start position (descending) so replacements don't shift indices
        results = sorted(results, key=lambda r: r.start, reverse=True)

        mapping = {}
        masked_text = text
        entity_counters = {}
        entities_found = []

        for result in results:
            entity_type = result.entity_type
            counter = entity_counters.get(entity_type, 0) + 1
            entity_counters[entity_type] = counter
            placeholder = f"<{entity_type}_{counter}>"
            original_value = text[result.start:result.end]

            # If we've already mapped this exact value (via a previous duplicate detection),
            # use the existing placeholder
            existing_placeholder = None
            for ph, val in mapping.items():
                if val == original_value:
                    existing_placeholder = ph
                    break

            if existing_placeholder:
                placeholder = existing_placeholder
                entity_counters[entity_type] = counter - 1
            else:
                mapping[placeholder] = original_value
                entities_found.append(entity_type)

            masked_text = (
                masked_text[:result.start]
                + placeholder
                + masked_text[result.end:]
            )

        if mapping:
            logger.info(f"Masked {len(mapping)} PII entity(ies): {entities_found}")

        return MaskingResult(
            masked_text=masked_text,
            mapping=mapping,
            entities_found=entities_found,
        )

    def unmask(self, text: str, mapping: dict) -> str:
        """Restore placeholders in LLM output to original PII values."""
        if not mapping:
            return text
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result
