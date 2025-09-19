
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
import joblib
import pandas as pd
import uvicorn
import logging
import warnings
from xgboost import XGBRegressor
import os, time, tempfile, io, getpass
import psycopg2

# ---------------- Logging & warnings ----------------
warnings.filterwarnings('ignore', category=FutureWarning, module='category_encoders')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FastAPI & templates ----------------
app = FastAPI(
    title="Trip Duration Prediction API (Batch CSV → CSV + PostgreSQL)",
    description="Upload a CSV, get a CSV with predicted_duration, and store results in PostgreSQL",
    version="3.1.0"
)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------- Model globals ----------------
preprocessor = None
model = None

# Uploaded CSV columns (label 'duration' present but not used for inference)
EXPECTED_COLS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "PULocationID",
    "DOLocationID",
    "duration",
]
# Model inputs (exclude 'duration')
MODEL_INPUT_COLS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "PULocationID",
    "DOLocationID",
]

# ---------------- PostgreSQL config (your details) ----------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": getpass.getuser(),   # e.g., 'gabriel'
    "password": "example",       # if your local role uses peer auth, fallback below handles it
}
DATABASE_NAME = "ml_predictions_db"

def get_conn():
    """
    Try TCP with provided password first, then fall back to Unix socket without password
    (common for Homebrew Postgres with peer auth).
    """
    try:
        return psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname=DATABASE_NAME,
        )
    except Exception as e:
        logger.warning(f"TCP connect failed ({e}); trying local socket without password …")
        return psycopg2.connect(
            host="/tmp",  # Homebrew Postgres socket
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            dbname=DATABASE_NAME,
        )

def ensure_table():
    """
    Create table if missing. Drop legacy source_file and rename processed_at->"time" if present.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS batch_predictions (
        id BIGSERIAL PRIMARY KEY,
        passenger_count DOUBLE PRECISION,
        trip_distance   DOUBLE PRECISION,
        fare_amount     DOUBLE PRECISION,
        total_amount    DOUBLE PRECISION,
        "PULocationID"  INTEGER,
        "DOLocationID"  INTEGER,
        duration        DOUBLE PRECISION,
        predicted_duration DOUBLE PRECISION NOT NULL,
        "time"          TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """
    alter = """
    -- Drop old column if it exists
    ALTER TABLE batch_predictions DROP COLUMN IF EXISTS source_file;

    -- Rename processed_at -> "time" if needed
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='batch_predictions' AND column_name='processed_at'
      ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='batch_predictions' AND column_name='time'
      ) THEN
        ALTER TABLE batch_predictions RENAME COLUMN processed_at TO "time";
      END IF;
    END$$;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(alter)
        conn.commit()
    logger.info('Ensured table batch_predictions exists (no source_file; has "time").')

def copy_chunk_to_db(df: pd.DataFrame):
    """
    Fast bulk insert via COPY. We let "time" default to now().
    """
    cols = [
        "passenger_count","trip_distance","fare_amount","total_amount",
        "PULocationID","DOLocationID","duration","predicted_duration"
    ]
    out = io.StringIO()
    df[cols].to_csv(out, index=False, header=False, na_rep="")
    out.seek(0)

    copy_sql = """
        COPY batch_predictions (
            passenger_count, trip_distance, fare_amount, total_amount,
            "PULocationID", "DOLocationID", duration, predicted_duration
        )
        FROM STDIN WITH (FORMAT CSV, NULL '');
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, out)
        conn.commit()

# ---------------- Loaders ----------------
def load_preprocessor(path: str):
    pp = joblib.load(path)
    logger.info("Preprocessor loaded.")
    return pp

def load_model(path: str):
    m = XGBRegressor()
    m.load_model(path)  # .ubj
    logger.info("Model loaded.")
    return m

# ---------------- Helpers ----------------
def _validate_expected_columns(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Expected at least: {EXPECTED_COLS}"
        )

def _predict_series(df_inputs: pd.DataFrame) -> pd.Series:
    X = preprocessor.transform(df_inputs)
    y = model.predict(X)
    return pd.Series(y, index=df_inputs.index, name="predicted_duration")

def _reserve_output_path(base_name: str) -> str:
    """Create a unique output filename in the current working directory."""
    root = f"{base_name}_with_predictions"
    path = os.path.join(os.getcwd(), f"{root}.csv")
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        candidate = os.path.join(os.getcwd(), f"{root}_{i}.csv")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# ---------------- Startup / health ----------------
@app.on_event("startup")
async def startup_event():
    global preprocessor, model
    preprocessor = load_preprocessor("preprocessing.pkl")
    model = load_model("my_model.ubj")
    ensure_table()

@app.get("/")
async def root():
    return {"message": "Trip Duration Prediction API is running (upload at /upload)"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None,
        "db_user": DB_CONFIG["user"],
        "db_name": DATABASE_NAME,
    }

# ---------------- Upload page (only UI we keep) ----------------
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

# ---------------- CSV upload → CSV download + PostgreSQL load ----------------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), chunksize: int = 50_000):
    """
    Upload a CSV with EXPECTED_COLS (includes 'duration').
    - Uses only MODEL_INPUT_COLS for inference.
    - Appends 'predicted_duration' to each row.
    - Saves the result in the current directory and returns it as a download.
    - Also bulk-inserts all rows into PostgreSQL table 'batch_predictions'.
    """
    if preprocessor is None or model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    if not (file.filename and file.filename.lower().endswith(".csv")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    t0 = time.time()
    # Save upload to temp so we can stream in chunks
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_in_path = tmp.name

    base_name = os.path.splitext(file.filename or "predictions")[0]
    output_path = _reserve_output_path(base_name)

    try:
        first = True
        header_written = False
        with open(output_path, "w", newline="") as out_f:
            for chunk in pd.read_csv(tmp_in_path, chunksize=chunksize):
                if first:
                    _validate_expected_columns(chunk)
                    first = False

                # Model prediction (exclude 'duration')
                X_df = chunk[MODEL_INPUT_COLS].copy()
                preds = _predict_series(X_df)

                # Append prediction to the original rows
                out_chunk = chunk.copy()
                out_chunk["predicted_duration"] = preds.values

                # 1) Append to output CSV on disk
                out_chunk.to_csv(out_f, index=False, header=not header_written)
                header_written = True

                # 2) COPY this chunk into PostgreSQL
                copy_chunk_to_db(out_chunk)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(f"Saved predictions to {output_path} and loaded into PostgreSQL in {elapsed_ms} ms")

        return FileResponse(
            path=output_path,
            media_type="text/csv",
            filename=os.path.basename(output_path),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing CSV")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
    finally:
        try:
            os.remove(tmp_in_path)
        except Exception:
            pass

# ---------------- Run server ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9696, reload=True)
