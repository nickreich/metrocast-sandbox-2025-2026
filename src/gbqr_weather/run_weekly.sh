#!/bin/bash
# Weekly run script for gbqr_weather model
# Usage: ./run_weekly.sh [--today_date YYYY-MM-DD]
#
# This script:
# 1. Downloads latest weather data (appends to existing)
# 2. Runs the gbqr_weather model
# 3. Generates forecast plots

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEATHER_FILE="$REPO_ROOT/auxiliary-data/weather_by_hsa.csv"

# Parse arguments
TODAY_DATE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --today_date)
      TODAY_DATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Default to today if not specified
if [ -z "$TODAY_DATE" ]; then
  TODAY_DATE=$(date +%Y-%m-%d)
fi

echo "========================================"
echo "GBQR Weather Model - Weekly Run"
echo "========================================"
echo "Date: $TODAY_DATE"
echo "Repo: $REPO_ROOT"
echo ""

# Step 1: Update weather data
echo "Step 1: Updating weather data..."
cd "$REPO_ROOT"

python3 << 'PYTHON_SCRIPT'
import pandas as pd
import requests
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(".")
WEATHER_FILE = REPO_ROOT / "auxiliary-data" / "weather_by_hsa.csv"
LOCATIONS_FILE = REPO_ROOT / "auxiliary-data" / "locations.csv"

def calculate_absolute_humidity(temp_c, rh_pct):
    if pd.isna(temp_c) or pd.isna(rh_pct):
        return None
    a, b, c = 17.625, 243.04, 6.1094
    es = c * math.exp((a * temp_c) / (b + temp_c))
    e = (rh_pct / 100.0) * es
    return (e * 100) / (461.5 * (temp_c + 273.15)) * 1000

def fetch_weather(lat, lon, start_date, end_date, max_retries=5):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum",
        "timezone": "America/New_York"
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error {resp.status_code}, retrying...")
                time.sleep(30)
        except Exception as e:
            print(f"    Exception: {e}, retrying...")
            time.sleep(30)
    return None

# Load existing data
print("Loading existing weather data...")
existing = pd.read_csv(WEATHER_FILE)
existing["week_end_date"] = pd.to_datetime(existing["week_end_date"])
last_date = existing["week_end_date"].max()
print(f"  Last date in file: {last_date.date()}")

# Determine date range to download
# Start from 2 weeks before last date (to handle any late data) through yesterday
start_date = (last_date - timedelta(days=14)).strftime("%Y-%m-%d")
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

if pd.to_datetime(end_date) <= last_date:
    print("  Weather data is already up to date!")
else:
    print(f"  Downloading: {start_date} to {end_date}")

    # Load locations
    locations = pd.read_csv(LOCATIONS_FILE)
    locations = locations[locations["location"].isin(existing["location"].unique())]

    all_data = []
    n_locs = len(locations)

    for i, row in locations.iterrows():
        loc = row["location"]
        lat, lon = row["latitude"], row["longitude"]
        print(f"  [{i+1}/{n_locs}] {loc}...", end="", flush=True)

        if i > 0:
            time.sleep(1)  # Small delay between requests

        data = fetch_weather(lat, lon, start_date, end_date)
        if data and "daily" in data:
            daily = data["daily"]
            print(f" {len(daily['time'])} days")

            for j in range(len(daily["time"])):
                temp_c = daily["temperature_2m_mean"][j]
                rh = daily["relative_humidity_2m_mean"][j]
                all_data.append({
                    "location": loc,
                    "location_name": row["location_name"],
                    "state": row["state"],
                    "state_abb": row["state_abb"],
                    "date": daily["time"][j],
                    "temp_avg_c": temp_c,
                    "temp_max_c": daily["temperature_2m_max"][j],
                    "temp_min_c": daily["temperature_2m_min"][j],
                    "humidity_avg_pct": rh,
                    "humidity_abs_gm3": calculate_absolute_humidity(temp_c, rh),
                    "precip_total_mm": daily["precipitation_sum"][j]
                })
        else:
            print(" FAILED")

    if all_data:
        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df["temp_avg_f"] = df["temp_avg_c"] * 9/5 + 32
        df["temp_max_f"] = df["temp_max_c"] * 9/5 + 32
        df["temp_min_f"] = df["temp_min_c"] * 9/5 + 32

        # Weekly aggregation (Saturday)
        df["week_end_date"] = df["date"] + pd.to_timedelta((5 - df["date"].dt.dayofweek) % 7, unit="D")

        weekly = df.groupby(["location", "location_name", "state", "state_abb", "week_end_date"]).agg({
            "temp_avg_f": "mean",
            "temp_max_f": "mean",
            "temp_min_f": "mean",
            "humidity_avg_pct": "mean",
            "humidity_abs_gm3": "mean",
            "precip_total_mm": "sum"
        }).reset_index()

        # Merge with existing (update overlapping dates)
        combined = pd.concat([existing, weekly], ignore_index=True)
        combined = combined.drop_duplicates(subset=["location", "week_end_date"], keep="last")
        combined = combined.sort_values(["location", "week_end_date"])

        combined.to_csv(WEATHER_FILE, index=False)
        print(f"\nUpdated weather file: {len(combined)} total obs")
        print(f"Date range: {combined['week_end_date'].min().date()} to {combined['week_end_date'].max().date()}")

print("Weather update complete!")
PYTHON_SCRIPT

echo ""
echo "Step 2: Running gbqr_weather model..."
cd "$REPO_ROOT"

# Activate virtual environment and run model
source src/gbqr/.venv/bin/activate
export PYTHONPATH="$REPO_ROOT"

python src/gbqr_weather/main.py --today_date "$TODAY_DATE" --use_local_mchub

# Get reference date (next Saturday from today_date)
REF_DATE=$(python3 -c "
from datetime import datetime
from dateutil.relativedelta import relativedelta, SA
d = datetime.strptime('$TODAY_DATE', '%Y-%m-%d')
ref = d + relativedelta(weekday=SA)
print(ref.strftime('%Y-%m-%d'))
")

echo ""
echo "Step 3: Generating forecast plots..."
cd "$REPO_ROOT/src/mchub_gbqr"
Rscript plot-forecasts.R "../../model-output/UMass-gbqr_weather/${REF_DATE}-UMass-gbqr_weather.csv"

echo ""
echo "========================================"
echo "Weekly run complete!"
echo "========================================"
echo "Reference date: $REF_DATE"
echo "Output: model-output/UMass-gbqr_weather/${REF_DATE}-UMass-gbqr_weather.csv"
echo "Plots:  src/mchub_gbqr/plots/${REF_DATE}-gbqr_weather-forecasts.pdf"
