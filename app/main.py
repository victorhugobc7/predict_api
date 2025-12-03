"""
Salary Prediction API
FastAPI application that loads a saved ML model and makes salary predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import ModelLoader

app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting average salary based on job features using ML models",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader
model_loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    """Load the model on application startup."""
    model_loader.load_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Salary Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(request: PredictionRequest):
    """
    Predict average salary based on input features.
    
    Returns the predicted average salary in thousands (K).
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists.",
        )
    
    try:
        prediction = model_loader.predict(request)
        return PredictionResponse(
            predicted_salary=round(prediction, 2),
            currency="USD",
            unit="thousands (K)",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return model_loader.get_model_info()
