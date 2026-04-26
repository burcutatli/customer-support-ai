"""api.py — REST API for Customer Support AI."""
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from main import setup_pipeline, process_message

logger = logging.getLogger(__name__)

app = FastAPI(title="Customer Support AI", version="1.0.0")


@app.on_event("startup")
def startup_event():
    logger.info("Initializing pipeline...")
    setup_pipeline()
    logger.info("Pipeline ready")


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    action: str
    response_to_customer: str
    category: str | None = None
    reason: str | None = None
    destination_team: str | None = None


@app.post("/api/process", response_model=MessageResponse)
def process_endpoint(request: MessageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        result = process_message(request.message)
        return MessageResponse(
            action=result["action"],
            response_to_customer=result["response_to_customer"],
            category=result.get("category"),
            reason=result.get("reason"),
            destination_team=result.get("destination_team"),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "customer-support-ai"}


@app.get("/")
def root():
    return {"service": "Customer Support AI", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
