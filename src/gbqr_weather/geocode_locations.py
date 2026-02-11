#!/usr/bin/env python3
"""One-time script to add latitude/longitude to locations.csv.

Uses the Open-Meteo geocoding API with state-level filtering to resolve
ambiguous city names (e.g., "Portland" -> Portland, ME not Portland, OR).

For state-level aggregate rows, geocodes the state capital.
For NC flu regions, geocodes the first listed county seat.
"""

import time
from pathlib import Path

import pandas as pd
import requests


def geocode(search: str, state: str, count: int = 10) -> tuple[float, float] | None:
    """Geocode a place name, filtering results to the correct US state.

    Args:
        search: Place name to search for (e.g., "Portland", "Denver").
        state: Full state name to filter by (e.g., "Maine", "Colorado").
        count: Number of API results to request for filtering.

    Returns:
        (latitude, longitude) rounded to 2 decimal places, or None.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": search,
        "count": count,
        "language": "en",
        "format": "json",
        "country_code": "US",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "results" not in data:
            return None

        # Filter to correct state
        for r in data["results"]:
            if r.get("admin1", "").lower() == state.lower():
                return (round(r["latitude"], 2), round(r["longitude"], 2))

        # Fallback: return first US result if no state match
        if data["results"]:
            r = data["results"][0]
            return (round(r["latitude"], 2), round(r["longitude"], 2))
    except Exception as e:
        print(f"  API error for '{search}': {e}")
    return None


# State capitals for geocoding state-level aggregate rows.
STATE_CAPITALS = {
    "Colorado": "Denver",
    "Georgia": "Atlanta",
    "Indiana": "Indianapolis",
    "Maine": "Augusta",
    "Maryland": "Annapolis",
    "Massachusetts": "Boston",
    "Minnesota": "Saint Paul",
    "South Carolina": "Columbia",
    "Texas": "Austin",
    "Utah": "Salt Lake City",
    "Virginia": "Richmond",
    "North Carolina": "Raleigh",
    "New York": "New York City",
    "Oregon": "Salem",
}


def extract_search_name(row: pd.Series) -> str:
    """Extract a geocodable search name from a locations.csv row.

    Strategy:
    - State-level rows (original_location_code == "All"): use state capital.
    - NC flu regions: use first county in hsa_counties.
    - Normal HSAs: parse city from location_name (text before last comma).
    """
    loc = row["location"]
    name = row["location_name"]
    state = row["state"]
    counties = str(row["hsa_counties"])

    # State-level aggregate rows
    if row["original_location_code"] == "All":
        return STATE_CAPITALS.get(state, state)

    # NC flu regions have non-city location_names like "Northeastern, NC"
    if row["location_type"] == "nc_flu_region_id" and counties:
        # Use first county name as search term
        first_county = counties.split(",")[0].strip()
        return first_county

    # Standard HSA: extract city name before state abbreviation
    # "Denver, CO" -> "Denver"
    # "Salt Lake City, UT" -> "Salt Lake City"
    # "Boston Metro, South Shore, Cape & Islands, MA" -> complex, use location id
    parts = name.rsplit(",", 1)
    city = parts[0].strip()

    # If city part still contains commas (complex name), use location id cleaned up
    if "," in city:
        # Fall back to the location slug: "boston" -> "Boston"
        return loc.replace("-", " ").title()

    return city


def main():
    repo_root = Path(__file__).parent.parent.parent
    locations_file = repo_root / "auxiliary-data" / "locations.csv"

    # Drop any existing lat/lon columns so we start fresh
    df = pd.read_csv(locations_file)
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")
    print(f"Loaded {len(df)} locations from {locations_file}")

    lats, lons = [], []

    for _, row in df.iterrows():
        loc = row["location"]
        state = row["state"]
        search = extract_search_name(row)

        result = geocode(search, state)
        if result:
            lat, lon = result
            print(f"  {loc:25s} -> {lat:7.2f}, {lon:8.2f}  (search: '{search}', state: {state})")
        else:
            print(f"  {loc:25s} -> FAILED  (search: '{search}', state: {state})")
            lat, lon = None, None

        lats.append(lat)
        lons.append(lon)
        time.sleep(0.3)  # Rate limit

    df["latitude"] = lats
    df["longitude"] = lons

    # Check for failures
    failures = df[df["latitude"].isna()]
    if len(failures) > 0:
        print(f"\nWARNING: {len(failures)} locations failed to geocode:")
        for _, row in failures.iterrows():
            print(f"  - {row['location']}: {row['location_name']}")
    else:
        print("\nAll locations geocoded successfully!")

    df.to_csv(locations_file, index=False)
    print(f"Wrote updated locations to {locations_file}")


if __name__ == "__main__":
    main()
