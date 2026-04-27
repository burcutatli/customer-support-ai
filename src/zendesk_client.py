"""
zendesk_client.py — Zendesk integration for Customer Support AI.

Wraps the Zendesk REST API for creating support tickets when the AI
pipeline escalates a customer message.

This is a stateless client — instantiate once, call create_ticket() many times.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-27
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TicketResult:
    """Result of a Zendesk ticket creation attempt."""
    success: bool
    ticket_id: Optional[int] = None
    ticket_url: Optional[str] = None
    error_message: Optional[str] = None


# ============================================================
# CLIENT
# ============================================================

class ZendeskClient:
    """
    Minimal Zendesk REST API client for ticket creation.
    
    Reads credentials from environment variables:
        - ZENDESK_SUBDOMAIN  (e.g., "freelance-22908")
        - ZENDESK_EMAIL      (the Zendesk account email)
        - ZENDESK_API_TOKEN  (API token from Admin Center)
    
    Usage:
        client = ZendeskClient()
        result = client.create_ticket(
            subject="Customer needs human help",
            description="Original message + AI context",
            priority="normal",
            tags=["ai_escalation", "low_confidence"],
        )
        if result.success:
            print(f"Created ticket #{result.ticket_id}")
    """
    
    def __init__(
        self,
        subdomain: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize the client.
        
        Args:
            subdomain: Zendesk subdomain (defaults to env var)
            email: Zendesk account email (defaults to env var)
            api_token: Zendesk API token (defaults to env var)
            timeout: Request timeout in seconds
        
        Raises:
            ValueError: If credentials are missing
        """
        self.subdomain = subdomain or os.getenv("ZENDESK_SUBDOMAIN")
        self.email = email or os.getenv("ZENDESK_EMAIL")
        self.api_token = api_token or os.getenv("ZENDESK_API_TOKEN")
        self.timeout = timeout
        
        if not all([self.subdomain, self.email, self.api_token]):
            missing = [
                name for name, val in [
                    ("ZENDESK_SUBDOMAIN", self.subdomain),
                    ("ZENDESK_EMAIL", self.email),
                    ("ZENDESK_API_TOKEN", self.api_token),
                ]
                if not val
            ]
            raise ValueError(
                f"Missing Zendesk credentials: {', '.join(missing)}. "
                f"Set them in .env"
            )
        
        self.base_url = f"https://{self.subdomain}.zendesk.com/api/v2"
        self.auth = (f"{self.email}/token", self.api_token)
        
        logger.info(f"ZendeskClient initialized for subdomain: {self.subdomain}")
    
    def create_ticket(
        self,
        subject: str,
        description: str,
        priority: str = "normal",
        tags: Optional[list[str]] = None,
        requester_email: Optional[str] = None,
    ) -> TicketResult:
        """
        Create a Zendesk support ticket.
        
        Args:
            subject: Short ticket title (visible in agent UI)
            description: Full ticket body (the customer message + AI context)
            priority: One of "low", "normal", "high", "urgent"
            tags: Optional list of tags for filtering/routing
            requester_email: Customer's email (if known). If None, uses the
                             account owner's email — fine for testing.
        
        Returns:
            TicketResult with success status and ticket ID/URL.
        """
        if priority not in {"low", "normal", "high", "urgent"}:
            logger.warning(f"Invalid priority '{priority}', defaulting to 'normal'")
            priority = "normal"
        
        ticket_payload: dict = {
            "subject": subject,
            "comment": {"body": description},
            "priority": priority,
        }
        
        if tags:
            ticket_payload["tags"] = tags
        
        if requester_email:
            ticket_payload["requester"] = {
                "name": requester_email.split("@")[0],
                "email": requester_email,
            }
        
        url = f"{self.base_url}/tickets.json"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json={"ticket": ticket_payload},
                    auth=self.auth,
                    headers={"Content-Type": "application/json"},
                )
            
            if response.status_code == 201:
                data = response.json()
                ticket = data.get("ticket", {})
                ticket_id = ticket.get("id")
                ticket_url = ticket.get("url")
                
                logger.info(
                    f"Zendesk ticket created: #{ticket_id} "
                    f"(priority={priority}, tags={tags})"
                )
                return TicketResult(
                    success=True,
                    ticket_id=ticket_id,
                    ticket_url=ticket_url,
                )
            else:
                logger.error(
                    f"Zendesk API error: {response.status_code} — "
                    f"{response.text[:200]}"
                )
                return TicketResult(
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                )
        
        except httpx.TimeoutException:
            logger.error(f"Zendesk request timed out after {self.timeout}s")
            return TicketResult(
                success=False,
                error_message=f"Request timed out after {self.timeout}s",
            )
        except httpx.RequestError as e:
            logger.error(f"Zendesk request failed: {e}")
            return TicketResult(
                success=False,
                error_message=f"Network error: {str(e)}",
            )
    
    def health_check(self) -> bool:
        """
        Verify Zendesk credentials work by hitting the /users/me endpoint.
        
        Returns:
            True if authenticated successfully, False otherwise.
        """
        url = f"{self.base_url}/users/me.json"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, auth=self.auth)
            
            if response.status_code == 200:
                logger.info("Zendesk health check passed")
                return True
            else:
                logger.error(
                    f"Zendesk health check failed: {response.status_code}"
                )
                return False
        except Exception as e:
            logger.error(f"Zendesk health check exception: {e}")
            return False


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    """
    Run this file directly to test Zendesk integration end-to-end.
    
    Usage:
        cd ~/Desktop/customer-support-ai
        python -m src.zendesk_client
    """
    from dotenv import load_dotenv
    
    # Load .env from project root
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    print("=" * 60)
    print("Zendesk Integration Test")
    print("=" * 60)
    
    # Step 1: Initialize
    client = ZendeskClient()
    
    # Step 2: Health check
    print("\n[1/2] Health check...")
    if not client.health_check():
        print("❌ Health check FAILED — credentials probably wrong.")
        exit(1)
    print("✅ Authentication works.")
    
    # Step 3: Create a test ticket
    print("\n[2/2] Creating test ticket...")
    result = client.create_ticket(
        subject="Sprint 6 — Python client test",
        description=(
            "This ticket was created by src/zendesk_client.py "
            "to verify the Python integration works end-to-end. "
            "If you see this in Zendesk, Sprint 6 Aşama 2 is complete."
        ),
        priority="normal",
        tags=["sprint_6", "ai_escalation_test"],
    )
    
    if result.success:
        print(f"✅ Ticket created!")
        print(f"   ID:  #{result.ticket_id}")
        print(f"   URL: {result.ticket_url}")
    else:
        print(f"❌ Ticket creation FAILED")
        print(f"   Error: {result.error_message}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All checks passed. Zendesk integration is live.")
    print("=" * 60)
