import os
from urllib.parse import quote_plus
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import date

# =============================
# FastAPI app
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
# Helper function to run queries
# =============================
def run_query(sql: str, params: dict = None):
    df = pd.read_sql(sql, engine, params=params)
    return df.to_dict(orient="records")

# =============================
# DATA MODELS
# =============================
class Counter(BaseModel):
    counter_id: str
    ticker: Optional[str] = None
    name: Optional[str] = None
    date_listed: Optional[date] = None  
    listing_price: Optional[float] = None
    sector: Optional[str] = None  

class PriceDaily(BaseModel):
    counter_id: str
    trade_date: Optional[date] = None
    open_mwk: Optional[float] = None
    high_mwk: Optional[float] = None
    low_mwk: Optional[float] = None
    close_mwk: Optional[float] = None
    volume: Optional[int] = None

# =============================
# Endpoints
# =============================

# Root endpoint
@app.get("/", summary="Root Endpoint")
def root():
    return {"message": "API is running"}

# Counters endpoint
@app.get("/counters", response_model=List[Counter], summary="Get all counters")
def get_counters():
    sql = "SELECT * FROM counters"
    return run_query(sql)

# Companies by sector endpoint
@app.get("/companies", response_model=List[Counter], summary="Get companies by sector")
def get_companies_by_sector(sector: str = Query(..., description="Sector to filter by")):
    query = text("SELECT * FROM counters WHERE sector ILIKE :sector")
    return run_query(query, {"sector": sector})

# Prices endpoint — latest 50 daily prices
@app.get("/prices", response_model=List[PriceDaily], summary="Get latest daily prices")
def get_prices(counter_id: Optional[str] = Query(None, description="Filter by counter_id")):
    with engine.connect() as conn:
        if counter_id:
            query = text("""
                SELECT *
                FROM prices_daily
                WHERE counter_id = :counter_id
                ORDER BY trade_date DESC
                LIMIT 10
            """)
            result = conn.execute(query, {"counter_id": counter_id}).fetchall()
            if not result:
                raise HTTPException(status_code=404, detail="No prices found for this counter_id")
        else:
            query = text("""
                SELECT *
                FROM prices_daily
                ORDER BY trade_date DESC
                LIMIT 10
            """)
            result = conn.execute(query).fetchall()
        
        # Convert SQLAlchemy rows to dicts
        return [dict(row._mapping) for row in result]
