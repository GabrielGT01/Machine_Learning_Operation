

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import logging
import warnings
from typing import List
from xgboost import XGBRegressor

# Suppress category_encoders warning
warnings.filterwarnings('ignore', category=FutureWarning, module='category_encoders')

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trip Duration Prediction API",
    description="API and Web UI for predicting NYC taxi trip duration using XGBoost",
    version="1.0.0"
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Globals
preprocessor = None
model = None

# Input schema
class RideData(BaseModel):
    passenger_count: float
    trip_distance: float
    fare_amount: float
    total_amount: float
    PULocationID: int
    DOLocationID: int
    

    class Config:
        schema_extra = {
            "example": {
                "passenger_count":1.0,
                "trip_distance": 4.12,
                "fare_amount":21.20,
                "total_amount":36.77,
                "PULocationID": 171,
                "DOLocationID": 73,
            }
        }


# Output schema
class PredictionResponse(BaseModel):
    predicted_duration: float
    status: str
    message: str

# Load preprocessor
def load_preprocessor(path: str):
    try:
        preprocessor = joblib.load(path)
        logger.info("Preprocessor loaded.")
        return preprocessor
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        raise

# Load model
def load_model(path: str):
    try:
        model = XGBRegressor()
        model.load_model(path) 
        logger.info("Model loaded.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Prediction logic
def predict_duration(preprocessor, model, ride_df):
    try:
        X_processed = preprocessor.transform(ride_df)
        prediction = model.predict(X_processed)
        return float(prediction[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# Load on startup
@app.on_event("startup")
async def startup_event():
    global preprocessor, model
    preprocessor = load_preprocessor("preprocessing.pkl")
    model = load_model("my_model.ubj")

# Health check
@app.get("/")
async def root():
    return {"message": "Trip Duration Prediction API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None
    }

# Single prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(ride_data: RideData):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    try:
        df = pd.DataFrame([ride_data.dict()])
        duration = predict_duration(preprocessor, model, df)
        return PredictionResponse(
            predicted_duration=duration,
            status="success",
            message="Prediction completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction
@app.post("/predict_batch")
async def predict_batch(rides: List[RideData]):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    try:
        results = []
        for ride in rides:
            df = pd.DataFrame([ride.dict()])
            duration = predict_duration(preprocessor, model, df)
            results.append(duration)
        return {
            "predictions": results,
            "status": "success",
            "message": f"Predicted durations for {len(rides)} rides"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Web form GET
@app.get("/form", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request, "result": None})

# Web form POST
# Web form POST
@app.post("/form", response_class=HTMLResponse)
async def form_post(
    request: Request,
    passenger_count: float = Form(...),
    trip_distance: float = Form(...),
    fare_amount: float = Form(...),
    total_amount: float = Form(...),
    PULocationID: int = Form(...),
    DOLocationID: int = Form(...),
):
    try:
        if preprocessor is None or model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        # Build the full feature set expected by the preprocessor/model
        df = pd.DataFrame([{
            "passenger_count": float(passenger_count),
            "trip_distance": float(trip_distance),
            "fare_amount": float(fare_amount),
            "total_amount": float(total_amount),
            "PULocationID": int(PULocationID),
            "DOLocationID": int(DOLocationID),
        }])
        result = predict_duration(preprocessor, model, df)
        return templates.TemplateResponse(
            "predict_form.html",
            {"request": request, "result": f"{result:.2f}"}
        )
    except Exception as e:
        logger.error(f"Form prediction error: {e}")
        return templates.TemplateResponse(
            "predict_form.html",
            {"request": request, "result": "Error during prediction"}
        )


# Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9696, reload=True)
