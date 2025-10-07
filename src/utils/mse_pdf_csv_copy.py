# mse_pdf_csv_copy.py

import logging
import os
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pdfplumber
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# GLOBAL VARIABLES
# ===============================================
_MONTHS = {
    'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
    'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,
    'sep':9,'sept':9,'september':9,'oct':10,'october':10,
    'nov':11,'november':11,'dec':12,'december':12
}

COUNTER_LIST = {'2021-2025': [
    'AIRTEL', 'BHL', 'CIPLA', 'FDHB', 'FMBCH', 'ICON', 'ILLOVO',
    'NBS', 'NICO', 'NITL', 'OMU', 'PCL', 'STANDARD', 'SUNBIRD',
    'TNM', 'UNIVERSAL'
]}

COLS = {
    '2021-2025': ['counter_id', 'daily_range_high', 'daily_range_low',
        'counter', 'buy_price', 'sell_price', 'previous_closing_price',
        'today_closing_price', 'volume_traded', 'dividend_mk', 'dividend_yield_pct',
        'earnings_yield_pct', 'pe_ratio', 'pbv_ratio', 'market_capitalization_mkmn',
        'profit_after_tax_mkmn', 'num_shares_issue']
}

# ===============================================
# HELPER FUNCTIONS
# ===============================================

def _mkdate(y, m, d):
    return date(int(y), int(m), int(d))

def _norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip()

def _parse_date_str(s: str, day_first: bool = True):
    if not s:
        return None
    s = _norm_text(s)

    # 1) 5 September 2025
    m = re.search(r'(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(20\d{2})\b', s)
    if m:
        d, mon, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num: return _mkdate(y, mon_num, d)

    # 2) September 5, 2025
    m = re.search(r'(?i)\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b', s)
    if m:
        mon, d, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num: return _mkdate(y, mon_num, d)

    # 3) ISO-like: 2025-09-05
    m = re.search(r'\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b', s)
    if m:
        y, mth, d = m.groups()
        try: return _mkdate(y, mth, d)
        except ValueError: pass

    # 4) Numeric: 05-09-2025
    m = re.search(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b', s)
    if m:
        a, b, y = m.groups()
        d, mth = (a, b) if day_first else (b, a)
        try: return _mkdate(y, mth, d)
        except ValueError: pass

    return None

def extract_date_from_filename(filename):
    filename = Path(filename).name

    pattern1 = r'Daily_Report_(\d{1,2})_([A-Za-z]+)_(\d{4})\.pdf'
    match = re.search(pattern1, filename)
    if match:
        day, month_str, year = match.groups()
        month_num = _MONTHS.get(month_str.lower())
        if month_num: return date(int(year), month_num, int(day))

    pattern2 = r'mse-daily-(\d{2})-(\d{2})-(\d{4})\.pdf'
    match = re.search(pattern2, filename)
    if match:
        day, month, year = match.groups()
        return date(int(year), int(month), int(day))

    pattern3 = r'mse-daily-(\d{4})-(\d{2})-(\d{2})\.pdf'
    match = re.search(pattern3, filename)
    if match:
        year, month, day = match.groups()
        return date(int(year), int(month), int(day))

    return _parse_date_str(filename)

def to_numeric_clean(val):
    if val is None: return np.nan
    val = str(val).strip()
    if val.lower() == "none" or val == "": return np.nan
    if val.startswith("(") and val.endswith(")"):
        val = "-" + val[1:-1]
    val = val.replace(",", "")
    try: return float(val)
    except ValueError: return np.nan

def clean_cell(x):
    if x is None: return None
    x = re.sub(r'\s+', ' ', str(x)).strip()
    x = x.replace('–', '-').replace('—', '-')
    return x if x else None

def normalize_to_width(rows: list[list], width: int) -> list[list]:
    out = []
    for r in rows:
        r = list(r)
        if len(r) < width:
            r = r + [None] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        out.append(r)
    return out

# ===============================================
# CORE EXTRACTION
# ===============================================

def extract_first_table(pdf_path: str | Path,
                        out_dir: Optional[str | Path] = None,
                        header: Optional[List[str]] = None,
                        skip_header_rows: int = 0,
                        auto_skip_header_like: bool = True) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_tables = page.extract_tables() or []
            if not raw_tables:
                continue
            raw = raw_tables[0]
            rows = [[clean_cell(c) for c in row] for row in raw]
            rows = [r for r in rows if any(c for c in r)]
            if not rows:
                continue

            cols = header if header else [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = normalize_to_width(rows[skip_header_rows:], len(cols))
            df = pd.DataFrame(data_rows, columns=cols).dropna(how="all")

            df['counter_id'] = pd.to_numeric(df['counter_id'], errors='coerce').astype('Int64')
            for c in df.columns:
                if c != "counter":
                    df[c] = df[c].apply(to_numeric_clean)

            df['trade_date'] = extract_date_from_filename(pdf_path)

            if out_dir:
                out_csv = out_dir / f"mse-daily-{df['trade_date'].iloc[0]}.csv"
                df.to_csv(out_csv, index=False)
                print(f"✅ Saved to {out_csv}")
                return out_csv
            return df
    return pd.DataFrame()

def get_most_recent_mse_report(directory_path):
    directory = Path(directory_path)
    pdf_files = sorted(directory.glob("*.pdf"), key=os.path.getmtime, reverse=True)
    return str(pdf_files[0]) if pdf_files else None

def process_multiple_pdfs(input_dir: Path, out_dir: Path, start_date: date, end_date: date,
                          cols: List[str], logs_dir: Optional[str | Path] = None) -> List[Optional[Path]]:
    not_processed = []
    processed_any = False

    for pdf_path in input_dir.glob('*.pdf'):
        try:
            file_date = extract_date_from_filename(pdf_path)
            if not file_date:
                print(f"⚠️ Skipping (no date): {pdf_path.name}")
                continue

            if start_date <= file_date <= end_date:
                processed_any = True
                print(f"Processing {pdf_path.name} dated {file_date}")
                output_file = extract_first_table(
                    pdf_path=pdf_path,
                    out_dir=out_dir,
                    header=cols,
                    skip_header_rows=1,
                    auto_skip_header_like=True
                )
                if output_file:
                    print(f"✅ Saved {output_file}")
                else:
                    print(f"❌ Failed to process {pdf_path.name}")
                    not_processed.append(pdf_path.name)
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")

    if not processed_any:
        print(f"⚠️ No reports found between {start_date} and {end_date} in {input_dir}")

    if not_processed and logs_dir:
        log_file = logs_dir / "unprocessed_daily_pdfs.txt"
        with open(log_file, "w") as f:
            for fname in not_processed:
                f.write(f"{fname}\n")
        print(f"⚠️ Some PDFs could not be processed. See {log_file}")


# ===============================================
# MAIN CLI
# ===============================================

def main():
    parser = argparse.ArgumentParser(description="Extract MSE daily report PDFs into CSVs.")
    parser.add_argument("--latest", action="store_true", help="Process only the most recent report")
    parser.add_argument("--range", action="store_true", help="Process reports within a date range")
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    DIR_DATA = script_dir.parent.parent / "data"
    DIR_REPORTS_PDF = DIR_DATA / "mse-daily-reports"
    DIR_REPORTS_CSV = DIR_DATA / "test-data"
    DIR_LOGS = script_dir / "logs/unprocessed_daily_pdfs"

    cols = COLS['2021-2025']

    if args.latest:
        pdf_path = get_most_recent_mse_report(DIR_REPORTS_PDF)
        if not pdf_path:
            print("❌ No PDF found")
            sys.exit(1)
        extract_first_table(pdf_path, DIR_REPORTS_CSV, header=cols, skip_header_rows=1)

    elif args.range:
        if not args.start_date or not args.end_date:
            print("❌ Provide both --start-date and --end-date")
            sys.exit(1)

        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            sys.exit(1)

        if start_date > end_date:
            print("❌ Start date cannot be after end date")
            sys.exit(1)

        process_multiple_pdfs(DIR_REPORTS_PDF, DIR_REPORTS_CSV, start_date, end_date, cols, DIR_LOGS)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
