"""
escalation_filter.py — First defense layer for Customer Support AI.

Checks customer messages for keywords or behavioral patterns that 
should bypass AI processing and route directly to human teams.

This is a stateless filter — instantiate once, call check() many times.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-25
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EscalationFilter:
    """
    Filters customer messages for direct-to-human escalation.
    
    Loads keyword lists and behavioral triggers from a JSON config file.
    Checks each message for keyword matches first, then behavioral triggers.
    Returns the first match found (no need to enumerate all matches).
    """
    
    def __init__(self, keywords_path: Path) -> None:
        """
        Load escalation rules from JSON file.
        
        Args:
            keywords_path: Path to escalation_keywords.json
        
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
        """
        if not keywords_path.exists():
            raise FileNotFoundError(
                f"Escalation keywords file not found at: {keywords_path}\n"
                f"Make sure the path in config.py is correct and the file exists."
            )
        
        with open(keywords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.keyword_categories = data.get("escalation_keywords", {})
        self.behavioral_rules = data.get("behavioral_triggers", {}).get("rules", [])
    
    def check(self, message: str) -> tuple[bool, str | None, str | None]:
        """
        Check if a message should be escalated to a human team.
        
        Args:
            message: The customer's raw message text
        
        Returns:
            A tuple of (should_escalate, reason, destination_team):
            - (True, "refund_keyword", "refund_team") if escalated
            - (False, None, None) if AI can handle it
        """
        message_lower = message.lower().strip()
        
        # --- Step 1: Keyword-based escalation ---
        for category_name, category_data in self.keyword_categories.items():
            keywords = category_data.get("keywords", [])
            destination_team = category_data.get("destination_team", "support_general")
            
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    return (True, f"{category_name}_keyword", destination_team)
        
        # --- Step 2: Behavioral triggers ---
        for rule in self.behavioral_rules:
            rule_name = rule.get("name")
            destination_team = rule.get("destination_team", "senior_support")
            
            if rule_name == "all_caps_shouting":
                if self._is_shouting(message):
                    return (True, "all_caps_shouting", destination_team)
            
            elif rule_name == "multiple_exclamations":
                if self._has_excessive_punctuation(message):
                    return (True, "multiple_exclamations", destination_team)
        
        # --- Step 3: No match — AI can handle ---
        return (False, None, None)
    
    @staticmethod
    def _is_shouting(message: str) -> bool:
        """Detect ALL CAPS messages (>70% uppercase letters, length >10)."""
        if len(message) <= 10:
            return False
        
        letters = [c for c in message if c.isalpha()]
        if not letters:
            return False
        
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        return uppercase_ratio > 0.70
    
    @staticmethod
    def _has_excessive_punctuation(message: str) -> bool:
        """Detect 3+ exclamation/question marks."""
        return message.count("!") + message.count("?") >= 3
