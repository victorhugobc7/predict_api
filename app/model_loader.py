import os
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
import joblib

from app.schemas import PredictionRequest


class ModelLoader:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.feature_columns: Optional[list] = None
        
    def load_model(self) -> bool:
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created models directory at {self.model_dir}")
            return False
        
        joblib_files = [
            "voting_regressor.joblib",
            "ridge_regression.joblib",
            "random_forest.joblib", 
            "linear_regression.joblib",
            "model.joblib",
        ]
        
        for filename in joblib_files:
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    self.model = joblib.load(model_path)
                    self.model_type = "sklearn"
                    print(f"Loaded sklearn model from {model_path}")
                    self._load_scaler()
                    self._load_feature_columns()
                    return True
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
        
        keras_files = ["model.keras", "model.h5", "meu_modelo.keras"]
        
        for filename in keras_files:
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(model_path)
                    self.model_type = "keras"
                    print(f"Loaded Keras model from {model_path}")
                    self._load_scaler()
                    self._load_feature_columns()
                    return True
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
        
        print("No model found. Please place your model in the 'models' directory.")
        return False
    
    def _load_scaler(self) -> None:
        scaler_path = self.model_dir / "scaler.joblib"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Failed to load scaler: {e}")
    
    def _load_feature_columns(self) -> None:
        columns_path = self.model_dir / "feature_columns.joblib"
        if columns_path.exists():
            try:
                self.feature_columns = joblib.load(columns_path)
                print(f"Loaded feature columns from {columns_path}")
            except Exception as e:
                print(f"Failed to load feature columns: {e}")
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def preprocess_input(self, request: PredictionRequest) -> np.ndarray:
        data = request.model_dump()
        df = pd.DataFrame([data])
        
        categorical_cols = ["job_simp", "seniority"]
        if any(col in df.columns for col in categorical_cols):
            df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)
        
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        
        features = df.values.astype(np.float64)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def predict(self, request: PredictionRequest) -> float:
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        features = self.preprocess_input(request)
        
        if self.model_type == "keras":
            prediction = self.model.predict(features, verbose=0)
            return float(prediction[0][0])
        else:
            prediction = self.model.predict(features)
            return float(prediction[0])
    
    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded():
            return {
                "loaded": False,
                "message": "No model loaded. Place your model file in the 'models' directory.",
            }
        
        info = {
            "loaded": True,
            "model_type": self.model_type,
            "scaler_loaded": self.scaler is not None,
            "feature_columns_count": len(self.feature_columns) if self.feature_columns else "unknown",
        }
        
        if self.model_type == "sklearn":
            info["model_class"] = type(self.model).__name__
        
        return info
