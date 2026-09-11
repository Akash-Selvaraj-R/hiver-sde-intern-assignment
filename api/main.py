"""
FastAPI endpoint for the customer support agent.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import SupportAgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hiver Customer Support Agent",
    description="AI-powered customer support agent for brand-specific Twitter support",
    version="1.0.0"
)

# Global pipeline instance
pipeline: Optional[SupportAgentPipeline] = None


class PredictionRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    intent: Dict[str, Any]
    retrieved_cases: List[Dict[str, Any]]
    reply: str
    should_escalate: bool
    escalation_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    logger.info("Initializing support agent pipeline...")
    try:
        pipeline = SupportAgentPipeline()
        # Fit with brand-specific data
        from classifier import load_training_data
        from retrieval import load_historical_cases

        train_texts, train_labels = load_training_data()
        historical_cases = load_historical_cases()

        pipeline.fit(
            training_data=(train_texts, train_labels),
            historical_cases=historical_cases
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # Don't fail startup, but pipeline will be None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hiver Customer Support Agent API",
        "status": "ready" if pipeline is not None else "initializing",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    status = pipeline.get_component_status()
    all_ready = all(status.values())

    return {
        "status": "healthy" if all_ready else "degraded",
        "components": status,
        "pipeline_ready": pipeline.is_fitted if pipeline else False
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Process a customer message through the support agent pipeline.

    Args:
        request: PredictionRequest containing the customer message

    Returns:
        PredictionResponse with intent, retrieved cases, reply, and escalation decision
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Process the message through the pipeline
        result = pipeline.predict(request.message.strip())

        # Convert to response format
        response = PredictionResponse(
            intent=result['intent'],
            retrieved_cases=result['retrieved_cases'],
            reply=result['reply'],
            should_escalate=result['should_escalate'],
            escalation_reason=result['escalation_reason'],
            metadata=result.get('metadata', {})
        )

        return response

    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)