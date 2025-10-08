import os
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import date, timedelta

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
# Pydantic models
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

class CompanyDetail(BaseModel):
    counter_id: str
    ticker: Optional[str] = None
    name: Optional[str] = None
    date_listed: Optional[date] = None
    listing_price: Optional[float] = None
    sector: Optional[str] = None
    description: Optional[str] = None
    total_records: int

class PriceRangeSummary(BaseModel):
    ticker: str
    year: int
    start_month: Optional[str] = None
    end_month: Optional[str] = None
    total_records: int
    period_high: Optional[float] = None
    period_low: Optional[float] = None
    total_volume: Optional[int] = None
    avg_close: Optional[float] = None
    daily_prices: List[PriceDaily]

class PriceLatest(BaseModel):
    counter_id: str
    ticker: Optional[str] = None
    trade_date: Optional[date] = None
    close_mwk: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None

# =============================
# Helper function to run queries
# =============================

def run_query(sql, params: dict = None) -> list[dict]:

    try:
        df = pd.read_sql(sql, engine, params=params)
        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)

                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None
                    else:
                        record[key] = float(value)

                elif pd.isna(value):
                    record[key] = None
        
        return records
    except Exception as e:
        print(f"Query error: {e}")
        raise

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
@app.get(
    "/companies",
    response_model=List[Counter],
    summary="Return all companies listed on the MSE optionally filtered by sector"
)
def get_companies_by_sector(sector: Optional[str] = Query(None, description="Sector to filter by")):
    if sector:
        query = text("SELECT * FROM counters WHERE sector ILIKE :sector")
        params = {"sector": f"%{sector}%"} 
    else:
        query = text("SELECT * FROM counters")
        params = {}
    
    return run_query(query, params)

# Company details by ticker with total records in entire dataset
@app.get(
    "/companies/{ticker}",
    response_model=CompanyDetail,
    summary="Get detailed information about a specific company by ticker"
)
def get_company_by_ticker(ticker: str):
    company_query = text("SELECT * FROM counters WHERE ticker = :ticker")
    company_result = run_query(company_query, {"ticker": ticker})
    
    if not company_result:
        raise HTTPException(status_code=404, detail=f"Company with ticker '{ticker}' not found")
    
    company = company_result[0]
    
    count_query = text("SELECT COUNT(*) AS total_records FROM prices_daily WHERE counter_id = :counter_id")
    count_result = run_query(count_query, {"counter_id": company["counter_id"]})
    total_records = count_result[0]["total_records"] if count_result else 0
    
    company["total_records"] = total_records
    
    if "description" not in company:
        company["description"] = None

    return company

# Daily prices with date filtering and limit

@app.get(
    "/prices/daily",
    response_model=List[PriceDaily],
    summary="Get daily stock prices with date filtering"
)
def get_daily_prices(
    ticker: str = Query(...),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: Optional[int] = Query(100, ge=1, le=1000)
):

    if start_date and end_date and end_date < start_date:
        raise HTTPException(
            status_code=422, 
            detail="End date cannot be earlier than start date"
        )

    counter_query = text("SELECT counter_id FROM counters WHERE ticker = :ticker")
    counter_result = run_query(counter_query, {"ticker": ticker})
    
    if not counter_result:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")
    
    counter_id = counter_result[0]["counter_id"]
    
    sql = """
        SELECT * FROM prices_daily 
        WHERE counter_id = :counter_id
    """
    params = {"counter_id": counter_id}

    if start_date:
        sql += " AND trade_date >= :start_date"
        params["start_date"] = start_date
    
    if end_date:
        sql += " AND trade_date <= :end_date"
        params["end_date"] = end_date

    sql += " ORDER BY trade_date DESC"
    
    query = text(sql)
    results = run_query(query, params)
    
    return results[:limit]


# Daily prices within a date range with summary statistics
@app.get(
    "/prices/range",
    response_model=PriceRangeSummary,
    summary="Get daily stock prices within a date range with summary statistics"
)
def get_daily_prices_range(
    ticker: str = Query(...),
    year: int = Query(...),
    start_month: Optional[int] = Query(None, ge=1, le=12),
    end_month: Optional[int] = Query(None, ge=1, le=12),
):
    # Calculate date range based on provided parameters
    if start_month and end_month:
        # Both provided: specific range 
        start_date = date(year, start_month, 1)
        end_date = date(year, end_month + 1, 1) - timedelta(days=1)
        start_month_str = f"{start_month:02d}"
        end_month_str = f"{end_month:02d}"
    elif start_month:
        # Only start_month: from start_month to December 
        start_date = date(year, start_month, 1)
        end_date = date(year, 12, 31)
        start_month_str = f"{start_month:02d}"
        end_month_str = "12"
    elif end_month:
        # Only end_month: from January to end_month
        start_date = date(year, 1, 1)
        end_date = date(year, end_month + 1, 1) - timedelta(days=1)
        start_month_str = "01"
        end_month_str = f"{end_month:02d}"
    else:
        # Neither provided: entire year 
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        start_month_str = "01"
        end_month_str = "12"

    counter_query = text("SELECT counter_id FROM counters WHERE ticker = :ticker")
    counter_result = run_query(counter_query, {"ticker": ticker})

    if not counter_result:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    counter_id = counter_result[0]["counter_id"]

    sql = """
        SELECT * FROM prices_daily
        WHERE counter_id = :counter_id
        AND trade_date >= :start_date
        AND trade_date <= :end_date
        ORDER BY trade_date DESC
    """
    params = {
        "counter_id": counter_id,
        "start_date": start_date,
        "end_date": end_date
    }

    query = text(sql)
    results = run_query(query, params)

    # Calculate summary statistics
    high_values = []
    low_values = []
    volume_values = []
    close_values = []

    for r in results:
        if r["high_mwk"] is not None:
            high_values.append(r["high_mwk"])
        
        if r["low_mwk"] is not None:
            low_values.append(r["low_mwk"])
        
        if r["volume"] is not None:
            volume_values.append(r["volume"])
        
        if r["close_mwk"] is not None:
            close_values.append(r["close_mwk"])

    if high_values:
        period_high = max(high_values)
    else:
        period_high = None

    if low_values:
        period_low = min(low_values)
    else:
        period_low = None

    if volume_values:
        total_volume = sum(volume_values)
    else:
        total_volume = 0

    if close_values:
        avg_close = sum(close_values) / len(close_values)
    else:
        avg_close = None

    summary = {
        "ticker": ticker,
        "year": year,
        "start_month": start_month_str,
        "end_month": end_month_str,
        "total_records": len(results),
        "period_high": period_high,
        "period_low": period_low,
        "total_volume": total_volume,
        "avg_close": avg_close,
        "daily_prices": results
    }
    
    return summary

@app.get(
    "/prices/latest",
    response_model=PriceLatest,
    summary="Get the latest stock prices"
)
def get_latest_prices(
    ticker: Optional[str] = Query(...),
):
    # Get the latest price for the given ticker
    sql = """
        SELECT * FROM prices_daily
        WHERE counter_id = (SELECT counter_id FROM counters WHERE ticker = :ticker)
        ORDER BY trade_date DESC
        LIMIT 1
    """
    params = {"ticker": ticker}
    query = text(sql)
    result = run_query(query, params)

    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    return result[0]


# Latest prices with change and change_percent
@app.get(
    "/prices/latest",
    response_model=PriceLatest,
    summary="Get the latest stock prices"
)
def get_latest_prices(ticker: Optional[str] = Query(None)):

    sql = """
        SELECT *
        FROM prices_daily
        WHERE counter_id = (SELECT counter_id FROM counters WHERE ticker = :ticker)
        ORDER BY trade_date DESC
        LIMIT 2
    """
    params = {"ticker": ticker}
    query = text(sql)
    results = run_query(query, params)

    if not results:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    latest = results[0]
    previous = results[1] if len(results) > 1 else None

    if previous:
        latest["change"] = latest["close_mwk"] - previous["close_mwk"]
        latest["change_percent"] = (latest["change"] / previous["close_mwk"]) * 100
    else:
        # No previous day price, set change values to None
        latest["change"] = None
        latest["change_percent"] = None

    return latest
