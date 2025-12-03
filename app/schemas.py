"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Request schema for salary prediction.
    
    These fields correspond to the features used in the trained model.
    Adjust based on your actual model's feature requirements.
    """
    
    # Numeric features
    rating: float = Field(..., ge=0, le=5, description="Company rating (0-5)")
    age: int = Field(..., ge=0, description="Company age in years")
    
    # Binary features (0 or 1)
    same_state: int = Field(0, ge=0, le=1, description="Job in same state as HQ")
    python_yn: int = Field(0, ge=0, le=1, description="Python skill required")
    R_yn: int = Field(0, ge=0, le=1, description="R skill required")
    spark: int = Field(0, ge=0, le=1, description="Spark skill required")
    aws: int = Field(0, ge=0, le=1, description="AWS skill required")
    excel: int = Field(0, ge=0, le=1, description="Excel skill required")
    
    # Categorical features (will be one-hot encoded)
    job_simp: Optional[str] = Field("data scientist", description="Simplified job title")
    seniority: Optional[str] = Field("na", description="Seniority level")
    desc_len: Optional[int] = Field(0, ge=0, description="Job description length")
    num_comp: Optional[int] = Field(0, ge=0, description="Number of competitors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rating": 3.5,
                "age": 10,
                "same_state": 1,
                "python_yn": 1,
                "R_yn": 0,
                "spark": 1,
                "aws": 1,
                "excel": 0,
                "job_simp": "data scientist",
                "seniority": "senior",
                "desc_len": 500,
                "num_comp": 3,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for salary prediction."""
    
    predicted_salary: float = Field(..., description="Predicted average salary")
    currency: str = Field("USD", description="Currency of the salary")
    unit: str = Field("thousands (K)", description="Unit of the salary value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_salary": 120.50,
                "currency": "USD",
                "unit": "thousands (K)",
            }
        }
