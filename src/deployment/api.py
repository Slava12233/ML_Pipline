"""
Model serving API module.

This module provides a FastAPI-based API for serving deployed models.
"""

import json
import logging
import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml

from src.deployment.vertex import predict, get_endpoint_info
from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Gemini PDF Fine-tuning API",
    description="API for serving fine-tuned Gemini models",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
class PredictionRequest(BaseModel):
    """Prediction request model."""
    
    inputs: List[Dict[str, Any]] = Field(
        ...,
        description="List of input instances",
        example=[{"text": "What is the capital of France?"}],
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional parameters for prediction",
        example={"temperature": 0.5, "max_output_tokens": 1024},
    )


class PredictionResponse(BaseModel):
    """Prediction response model."""
    
    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="List of predictions",
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Information about the model",
    )


# Define API configuration
class APIConfig:
    """API configuration."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize API configuration.

        Args:
            config_path: Path to configuration file.
        """
        self.config = {}
        
        # Load configuration from file if provided
        if config_path is not None:
            self.config = load_config(config_path)
        
        # Load configuration from environment variables
        self.endpoint_id = os.environ.get("ENDPOINT_ID", self.config.get("endpoint_id", ""))
        self.project_id = os.environ.get("PROJECT_ID", self.config.get("project_id", ""))
        self.location = os.environ.get("LOCATION", self.config.get("location", "us-central1"))
        self.api_key = os.environ.get("API_KEY", self.config.get("api_key", ""))
        self.log_predictions = os.environ.get("LOG_PREDICTIONS", self.config.get("log_predictions", "true")).lower() == "true"
        self.enable_monitoring = os.environ.get("ENABLE_MONITORING", self.config.get("enable_monitoring", "true")).lower() == "true"


# Create API configuration
api_config = APIConfig()


# Define dependency for API configuration
def get_api_config():
    """Get API configuration."""
    return api_config


# Define routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Gemini PDF Fine-tuning API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    config: APIConfig = Depends(get_api_config),
):
    """
    Make predictions with deployed model.

    Args:
        request: Prediction request.
        background_tasks: Background tasks.
        config: API configuration.

    Returns:
        PredictionResponse: Prediction response.
    """
    try:
        # Make prediction
        response = predict(
            endpoint=config.endpoint_id,
            instances=request.inputs,
            parameters=request.parameters,
        )
        
        # Log prediction if enabled
        if config.log_predictions:
            background_tasks.add_task(
                log_prediction,
                request=request.dict(),
                response=response,
            )
        
        # Get model information if available
        model_info = None
        try:
            endpoint_info = get_endpoint_info(config.endpoint_id)
            model_info = {
                "endpoint_id": config.endpoint_id,
                "deployed_models": endpoint_info.get("deployed_models", []),
            }
        except Exception as e:
            logger.warning(f"Error getting model information: {str(e)}")
        
        # Return response
        return PredictionResponse(
            predictions=response.predictions,
            model_info=model_info,
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def models(
    config: APIConfig = Depends(get_api_config),
):
    """
    Get information about deployed models.

    Args:
        config: API configuration.

    Returns:
        Dict[str, Any]: Model information.
    """
    try:
        # Get endpoint information
        endpoint_info = get_endpoint_info(config.endpoint_id)
        
        # Return model information
        return {
            "endpoint_id": config.endpoint_id,
            "endpoint_info": endpoint_info,
        }
    
    except Exception as e:
        logger.error(f"Error getting model information: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    config: APIConfig = Depends(get_api_config),
):
    """
    Submit feedback for a prediction.

    Args:
        request: Feedback request.
        background_tasks: Background tasks.
        config: API configuration.

    Returns:
        Dict[str, Any]: Feedback response.
    """
    try:
        # Log feedback
        background_tasks.add_task(
            log_feedback,
            feedback=request,
        )
        
        # Return response
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Define logging functions
def log_prediction(
    request: Dict[str, Any],
    response: Dict[str, Any],
):
    """
    Log prediction.

    Args:
        request: Prediction request.
        response: Prediction response.
    """
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request": request,
            "response": response,
        }
        
        # Log entry
        logger.info(f"Prediction: {json.dumps(log_entry)}")
    
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")


def log_feedback(
    feedback: Dict[str, Any],
):
    """
    Log feedback.

    Args:
        feedback: Feedback.
    """
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "feedback": feedback,
        }
        
        # Log entry
        logger.info(f"Feedback: {json.dumps(log_entry)}")
    
    except Exception as e:
        logger.error(f"Error logging feedback: {str(e)}")


# Define middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add process time header.

    Args:
        request: Request.
        call_next: Call next function.

    Returns:
        Response: Response.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Define exception handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Generic exception handler.

    Args:
        request: Request.
        exc: Exception.

    Returns:
        JSONResponse: JSON response.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# Define API server
def run_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: Optional[Union[str, Path]] = None,
    log_level: str = "info",
):
    """
    Run API server.

    Args:
        host: Host.
        port: Port.
        config_path: Path to configuration file.
        log_level: Log level.
    """
    # Load configuration
    global api_config
    api_config = APIConfig(config_path)
    
    # Run server
    uvicorn.run(
        "src.deployment.api:app",
        host=host,
        port=port,
        log_level=log_level,
    )


# Define main function
def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    args = parser.parse_args()
    
    # Run API server
    run_api(
        host=args.host,
        port=args.port,
        config_path=args.config,
        log_level=args.log_level,
    )


# Run main function
if __name__ == "__main__":
    main()
