import os
from urllib.parse import quote_plus
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import date

# =============================
# FastAPI App
# =============================
app = FastAPI(title="MSE API", version="1.0")

# =============================
# Database connection setup
# =============================
load_dotenv()

PGHOST = os.getenv("PGHOST").strip()
PGPORT = int(''.join(filter(str.isdigit, os.getenv("PGPORT").strip())))
PGDATABASE = os.getenv("PGDATABASE").strip()
PGUSER = os.getenv("PGUSER").strip()
PGPASSWORD = os.getenv("PGPASSWORD").strip()
encoded_password = quote_plus(PGPASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{PGUSER}:{encoded_password}@{PGHOST}:{PGPORT}/{PGDATABASE}"
)

# =============================
# Helper function
# =============================
def run_query(sql: str, params: tuple = ()):
    df = pd.read_sql(sql, engine, params=params)
    return df.to_dict(orient="records")

# =============================
# DATA MODELS DEFINITION
# =============================
class Counter(BaseModel):
    counter_id: str
    ticker: Optional[str] = None
    name: Optional[str] = None
    date_listed: Optional[date] = None  
    listing_price: Optional[float] = None

class Price(BaseModel):
    counter_id: str
    daily_range_high: Optional[float] = None
    daily_range_low: Optional[float] = None
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    previous_closing_price: Optional[float] = None
    today_closing_price: Optional[float] = None

# =============================
# Endpoints
# =============================
@app.get("/", summary="Root Endpoint")
def root():
    return {"message": "API is running"}

@app.get("/counters", response_model=List[Counter], summary="Get all counters")
def get_counters():
    sql = "SELECT * FROM counters"
    return run_query(sql)

@app.get("/prices", response_model=List[Price], summary="Get daily prices")
def get_prices(counter_id: Optional[str] = Query(None, description="Filter by counter_id")):
    if counter_id:
        sql = "SELECT * FROM prices_daily WHERE counter_id = %s"
        return run_query(sql, (counter_id,))
    else:
        sql = "SELECT * FROM prices_daily"
        return run_query(sql)
