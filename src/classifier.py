"""
classifier.py — Message classification using Claude Haiku.

Uses Claude Haiku to categorize incoming customer messages into 
predefined categories with a confidence score. Returns structured 
JSON output for downstream routing decisions.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-25
"""

import json
import logging
from anthropic import Anthropic, APIError
from langfuse import observe, get_client

logger = logging.getLogger(__name__)
_langfuse = get_client()


class MessageClassifier:
    """
    Classifies customer messages into categories using Claude Haiku.
    
    Returns category, confidence score, and brief reasoning.
    Falls back gracefully on errors (returns "general" with 0.0 confidence).
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        valid_categories: list[str],
    ) -> None:
        """
        Initialize the classifier with API credentials and valid categories.
        
        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-haiku-4-5")
            max_tokens: Maximum tokens for response
            valid_categories: List of allowed category names
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.valid_categories = valid_categories
    
    @observe(as_type="generation", name="classifier")
    def classify(self, message: str) -> dict:
        """
        Classify a customer message into a category.
        
        Args:
            message: Raw customer message text
        
        Returns:
            Dict with keys: category, confidence, reasoning
            On error, falls back to {"category": "general", "confidence": 0.0, ...}
        """
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": message}],
            )
            
            try:
                _langfuse.update_current_generation(
                    model=self.model,
                    input=message,
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )
            except Exception as e:
                logger.warning(f"Langfuse update failed (non-fatal): {e}")
            
            raw_text = response.content[0].text.strip()
            result = self._parse_response(raw_text)
            return result
        
        except APIError as e:
            logger.error(f"Anthropic API error during classification: {e}")
            return self._fallback_response("api_error")
        
        except Exception as e:
            logger.error(f"Unexpected error during classification: {e}")
            return self._fallback_response("unexpected_error")
    
    def _build_system_prompt(self) -> str:
        """Construct the system prompt with valid categories."""
        category_list = "\n".join(f"- {cat}" for cat in self.valid_categories)
        
        return f"""You are a customer support message classifier.
Your task is to categorize incoming customer messages into ONE of these categories:

{category_list}

Category descriptions:
- shipping: questions about delivery, tracking, shipping times
- billing: questions about invoices, payments, promo codes
- account: questions about login, password, account access
- product: questions about product features, sizing, stock
- sales: corporate inquiries, bulk orders, B2B questions
- order_change: order modifications, wrong orders
- general: questions that don't fit the above categories

Respond ONLY with valid JSON in this exact format:
{{"category": "shipping", "confidence": 0.92, "reasoning": "brief explanation"}}

Rules:
- category must be EXACTLY one of the listed categories (lowercase)
- confidence must be a number between 0.0 and 1.0
- reasoning must be ONE sentence, max 15 words
- Do NOT use markdown code blocks (no ```)
- Do NOT add any text before or after the JSON"""
    
    def _parse_response(self, raw_text: str) -> dict:
        """Parse Claude's JSON response and validate."""
        try:
            # Strip markdown code blocks if Claude included them despite instructions
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            
            parsed = json.loads(raw_text)
            
            category = parsed.get("category", "").lower()
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "")
            
            # Validate category
            if category not in self.valid_categories:
                logger.warning(f"Invalid category returned: {category}")
                return self._fallback_response("invalid_category")
            
            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse classifier response: {e} | Raw: {raw_text}")
            return self._fallback_response("parse_error")
    
    def _fallback_response(self, reason: str) -> dict:
        """Return safe default when classification fails."""
        return {
            "category": "general",
            "confidence": 0.0,
            "reasoning": f"fallback: {reason}",
        }
