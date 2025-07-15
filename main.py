from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from model import load_model, get_embedding
from pydantic import BaseModel
from typing import List
import numpy as np

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

model = load_model()


class EmbeddingRequest(BaseModel):
    X_seq: List[List[float]]  # Shape should be (100, 6) for a single lap
    X_context: List[float]    # Shape should be (22,) for context features


@app.post("/embed")
@limiter.limit("10/minute")
async def model_endpoint(request: Request, data: EmbeddingRequest):
    try:
        # Validate input shapes
        X_seq = np.array(data.X_seq)
        X_context = np.array(data.X_context)

        # Check X_seq shape: should be (100, 6)
        if X_seq.shape != (100, 6):
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"X_seq must have shape (100, 6), got {X_seq.shape}"}
            )

        # Check X_context shape: should be (22,)
        if X_context.shape != (22,):
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"X_context must have shape (22,), got {X_context.shape}"}
            )

        # Add batch dimension for model inference
        X_seq_batch = X_seq[np.newaxis, :]  # Shape: (1, 100, 6)
        X_context_batch = X_context[np.newaxis, :]  # Shape: (1, 22)

        # Get embedding (lap time)
        embedding = get_embedding(model, X_seq_batch, X_context_batch)

        return JSONResponse(content={
            "embedded_lap_time": embedding[0] if isinstance(embedding, list) else float(embedding),
            "status": "success"
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Error processing request", "details": str(e)}
        )


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "model_loaded": model is not None})


@app.get("/model_info")
async def model_info():
    return JSONResponse(content={
        "model_type": "F1Embedder",
        "input_shapes": {
            "X_seq": "(100, 6)",
            "X_context": "(22,)"
        },
        "features": {
            "telemetry": ["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"],
            "context": "One-hot encoded categorical features (Driver, Team, Compound, etc.)"
        },
        "output": "embedded lap time in seconds"
    })


@app.exception_handler(RateLimitExceeded)
def rate_limit_exceeded_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
