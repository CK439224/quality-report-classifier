"""
Pull real complaint narratives + category labels from NHTSA's public
complaints API and write them to a CSV with the same shape the rest of the
pipeline expects (report_id, category, narrative).

No API key required. Endpoint:
    GET https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={year}

NHTSA's API field-name casing has been inconsistent across versions and
documentation, so this script checks a few possible casings defensively
rather than assuming one. If it can't find the fields it expects, it will
save the raw first response to data/_debug_raw_response.json so you can see
exactly what came back and adjust FIELD candidates below.

NOTE: this was written and tested against public documentation, but this
project's sandbox has restricted network egress and can't reach
api.nhtsa.gov directly -- run this from your own machine. Start with
--limit 5 on one vehicle and inspect the output before pulling a lot of
data.

Usage:
    python src/fetch_data.py --make honda --model accord --model-year 2015 2016 2017 --out data/nhtsa_real.csv
    python src/fetch_data.py --make honda --model accord --model-year 2015 --limit 5 --out data/_test.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

# Possible key names for each field we need, in priority order. NHTSA has
# used both PascalCase (older docs / legacy API) and camelCase/lowercase
# (current api.nhtsa.gov) at different times.
FIELD_CANDIDATES = {
    "narrative": ["summary", "Summary", "SUMMARY"],
    "category": ["components", "Components", "component", "Component", "COMPONENTS"],
    "odi_number": ["odiNumber", "ODINumber", "odinumber"],
}


def extract_field(record: dict, field: str):
    for key in FIELD_CANDIDATES[field]:
        if key in record and record[key]:
            return record[key]
    return None


def fetch_one(make: str, model: str, year: int, session: requests.Session) -> list:
    params = {"make": make, "model": model, "modelYear": year}
    resp = session.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Response wrapper key has also varied ("Results" vs "results") -- check both.
    results = payload.get("results") or payload.get("Results")
    if results is None:
        # Unknown shape -- dump it so the user can inspect and fix FIELD_CANDIDATES.
        debug_path = Path("data/_debug_raw_response.json")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(payload, indent=2))
        print(
            f"WARNING: couldn't find a 'results' array for {make} {model} {year}. "
            f"Raw response saved to {debug_path} -- inspect it and update FIELD_CANDIDATES "
            f"in this script if the field names have changed.",
            file=sys.stderr,
        )
        return []

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make", required=True, help="Vehicle manufacturer, e.g. honda")
    parser.add_argument("--model", required=True, help="Vehicle model, e.g. accord")
    parser.add_argument("--model-year", type=int, nargs="+", required=True, help="One or more model years, e.g. 2015 2016 2017")
    parser.add_argument("--out", type=str, default="data/nhtsa_real.csv")
    parser.add_argument("--limit", type=int, default=None, help="Cap total rows written (useful for a quick sanity check)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests to be polite to the API")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "quality-report-classifier-portfolio-project/1.0"})

    rows = []
    report_id = 1
    for year in args.model_year:
        print(f"Fetching {args.make} {args.model} {year}...")
        try:
            results = fetch_one(args.make, args.model, year, session)
        except requests.RequestException as e:
            print(f"  Request failed for {year}: {e}", file=sys.stderr)
            continue

        kept = 0
        for record in results:
            narrative = extract_field(record, "narrative")
            category = extract_field(record, "category")
            if not narrative or not category:
                continue
            # `components` can come back as a "/"-delimited hierarchy, e.g.
            # "ENGINE AND ENGINE COOLING:ENGINE" -- keep just the top-level.
            if isinstance(category, str) and ":" in category:
                category = category.split(":")[0].strip()

            rows.append({"report_id": report_id, "category": category, "narrative": narrative})
            report_id += 1
            kept += 1

            if args.limit and len(rows) >= args.limit:
                break

        print(f"  kept {kept} of {len(results)} returned records")

        if args.limit and len(rows) >= args.limit:
            break

        time.sleep(args.sleep)

    if not rows:
        print("No rows collected -- check the warnings above and data/_debug_raw_response.json if it was created.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", "category", "narrative"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} real complaint reports to {out_path}")
    print("Sanity-check the first few rows before trusting this for training:")
    for r in rows[:3]:
        print(f"  [{r['category']}] {r['narrative'][:120]}...")


if __name__ == "__main__":
    main()
