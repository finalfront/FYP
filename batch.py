"""
batch_process.py
----------------
Batch-processes all drug JSON files in the data/ folder,
applies AE anomaly filtering, and saves .txt reports to dataoutput/.

Usage:
    python batch_process.py

Output:
    dataoutput/<DrugName>.txt  for every data/<DrugName>.json found
    (skips green_ranking.json which is not a retrosynthesis graph file)
"""

import os
import json
import time
from pathlib import Path

from extraction import RetrosynthesisParser, filter_graphs, save_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("dataoutput")

# Files to skip (not retrosynthesis graph files)
SKIP_FILES = {"green_ranking.json"}

# Sort key for graphs within each drug
# Options: 'avg_plausibility', 'avg_score', 'precursor_cost', 'atom_economy'
SORT_KEY     = "avg_plausibility"
SORT_REVERSE = True   # True = highest first


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(json_path: Path, output_path: Path) -> dict:
    """Process a single JSON file. Returns a summary dict."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = RetrosynthesisParser()
    graphs = parser.parse(data)

    total_before = len(graphs)
    graphs = sorted(graphs, key=lambda g: getattr(g.metadata, SORT_KEY), reverse=SORT_REVERSE)
    graphs = filter_graphs(graphs, remove_ae_anomalies=True)
    total_after = len(graphs)

    save_report(graphs, str(output_path))

    return {
        "drug":           json_path.stem,
        "total_graphs":   total_before,
        "after_filter":   total_after,
        "removed":        total_before - total_after,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(
        p for p in DATA_DIR.glob("*.json")
        if p.name not in SKIP_FILES
    )

    if not json_files:
        print(f"No JSON files found in {DATA_DIR}/")
        return

    print(f"Found {len(json_files)} drug file(s) to process.\n")
    print("=" * 50)

    summaries = []
    t0 = time.time()

    for json_path in json_files:
        drug_name   = json_path.stem
        output_path = OUTPUT_DIR / f"{drug_name}.txt"

        print(f"\n[{drug_name}]")
        try:
            summary = process_file(json_path, output_path)
            summaries.append(summary)
        except Exception as exc:
            print(f"  [ERROR] Failed to process {json_path.name}: {exc}")
            summaries.append({
                "drug": drug_name,
                "total_graphs": "?",
                "after_filter": "?",
                "removed": "?",
                "error": str(exc),
            })

    elapsed = time.time() - t0

    # Summary table
    print("\n" + "=" * 50)
    print("BATCH SUMMARY")
    print("=" * 50)
    print(f"{'Drug':<20} {'Total':>7} {'Kept':>6} {'Removed':>8}")
    print("-" * 50)
    for s in summaries:
        if "error" in s:
            print(f"  {s['drug']:<18} ERROR: {s['error']}")
        else:
            print(f"  {s['drug']:<18} {s['total_graphs']:>7} {s['after_filter']:>6} {s['removed']:>8}")
    print("-" * 50)
    print(f"Completed in {elapsed:.1f}s.  Output folder: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()