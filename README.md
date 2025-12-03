# Salary Prediction API

A FastAPI-based REST API for predicting average salaries using machine learning models.

## Features

- Load and serve ML models (scikit-learn, Keras/TensorFlow)
- RESTful API endpoint for salary predictions
- Automatic feature preprocessing and scaling
- Health check and model info endpoints
- Interactive API documentation (Swagger UI)

## Project Structure

```
predict_api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Pydantic request/response schemas
│   └── model_loader.py  # Model loading utilities
├── models/              # Place your trained models here
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Model

Place your trained model files in the `models/` directory:

- `voting_regressor.joblib` - Ensemble model (recommended)
- `random_forest.joblib` - Random Forest model
- `linear_regression.joblib` - Linear Regression model
- `model.keras` or `model.h5` - Keras model

Optional files:
- `scaler.joblib` - MinMaxScaler or StandardScaler for feature preprocessing
- `feature_columns.joblib` - List of feature column names in the correct order

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check status |
| `/predict` | POST | Make salary prediction |
| `/model/info` | GET | Get loaded model information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## Usage Example

### Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "num_comp": 3
  }'
```

### Response

```json
{
  "predicted_salary": 120.50,
  "currency": "USD",
  "unit": "thousands (K)"
}
```

## Saving Models from Your Notebook

After training your models, save them for the API:

```python
import joblib

# Save the trained models
joblib.dump(search_lr.best_estimator_, "models/linear_regression.joblib")
joblib.dump(search_rf.best_estimator_, "models/random_forest.joblib")
joblib.dump(voting_reg, "models/voting_regressor.joblib")

# Save the scaler
joblib.dump(scaler, "models/scaler.joblib")

# Save feature column names
joblib.dump(list(X.columns), "models/feature_columns.joblib")
```

## Development

Run with auto-reload for development:

```bash
uvicorn app.main:app --reload
```

Access the interactive documentation at: http://localhost:8000/docs

## License

MIT License
