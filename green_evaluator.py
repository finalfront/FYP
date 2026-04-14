"""
green_evaluator.py
==================
Retrosynthesis Green Chemistry Evaluator

Evaluates retrosynthetic routes against the 12 Principles of Green Chemistry
using live PubChem GHS hazard data and an Azure OpenAI LLM scoring engine.

Setup
-----
Set credentials as environment variables before running:

    export AZURE_OPENAI_KEY='your-key-here'
    export AZURE_OPENAI_ENDPOINT='https://your-endpoint.azure-api.net'

Usage (single drug)
-------------------
    python green_evaluator.py --input ./dataoutput/Aspirin.txt

Usage (batch — all .txt files in a folder)
-------------------------------------------
    python green_evaluator.py --batch ./dataoutput

    Add --force to re-evaluate drugs that already have a _green_scores.json.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import re
import json
import os
import time
import argparse
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_WORKERS   = 5      # parallel PubChem threads
REQUEST_DELAY = 0.2    # seconds between requests per thread
MAX_RETRIES   = 3      # retry attempts on transient network failures

MODEL         = "gpt-4.1-mini"
MAX_TOKENS    = 32768
TEMPERATURE   = 0
API_VERSION   = "2025-02-01-preview"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (retrosynthesis-green-eval/1.0; contact: you@example.com)"
}

H_CODE_RE = re.compile(r"\bH\d{3}\b")

# ---------------------------------------------------------------------------
# PubChem helpers
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, retries: int = MAX_RETRIES):
    """HTTP GET with exponential back-off retry on network / SSL errors."""
    for attempt in range(retries):
        try:
            return requests.get(url, headers=HEADERS, timeout=10)
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise exc


def get_cid_from_smiles(smiles: str):
    """Resolve a SMILES string to a PubChem CID via PUG REST. Returns None on failure."""
    encoded = urllib.parse.quote(smiles, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/cids/JSON"
    try:
        resp = _get_with_retry(url)
        if resp.status_code == 200:
            cids = resp.json().get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]
    except Exception as exc:
        print(f"[WARN] CID lookup failed for {smiles[:40]}…: {exc}")
    return None


def _collect_hcodes_recursive(obj, out_set: set):
    """Recursively scan a PubChem PUG View JSON object and collect all H-codes."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_hcodes_recursive(v, out_set)
    elif isinstance(obj, list):
        for item in obj:
            _collect_hcodes_recursive(item, out_set)
    elif isinstance(obj, str):
        for m in H_CODE_RE.findall(obj):
            out_set.add(m)


def get_ghs_hazards(cid) -> list:
    """Return sorted list of GHS H-codes for a given PubChem CID."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
        "?heading=GHS+Classification"
    )
    h_codes: set = set()
    try:
        resp = _get_with_retry(url)
        if resp.status_code == 200:
            _collect_hcodes_recursive(resp.json(), h_codes)
    except Exception as exc:
        print(f"[WARN] GHS lookup failed for CID {cid}: {exc}")
    return sorted(h_codes)


def _lookup_smiles(smiles: str, cid_cache: dict, ghs_cache: dict):
    """Thread-safe worker: SMILES → CID → GHS H-codes (with dual caching)."""
    if len(smiles) < 3:
        return smiles, []

    cid = cid_cache.get(smiles)
    if cid is None:
        cid = get_cid_from_smiles(smiles)
        cid_cache[smiles] = cid  # may be None

    if not cid:
        return smiles, []

    if cid in ghs_cache:
        return smiles, ghs_cache[cid]

    time.sleep(REQUEST_DELAY)
    h_codes = get_ghs_hazards(cid)
    ghs_cache[cid] = h_codes
    return smiles, h_codes

# ---------------------------------------------------------------------------
# SMILES extraction
# ---------------------------------------------------------------------------

def extract_unique_smiles(graph_text: str) -> set:
    """
    Parse all unique SMILES from retrosynthesis graph text.

    Handles:
      Product: <SMILES>
      Reactants: [SMILES, SMILES, ...]

    SMILES containing brackets (e.g. [O-]) are captured correctly by
    matching to end-of-line rather than the first ']'.
    """
    smiles_set: set = set()

    for m in re.finditer(r"^\s*Product:\s*(\S+)\s*$", graph_text, flags=re.M):
        smiles_set.add(m.group(1).strip())

    for m in re.finditer(r"^\s*Reactants:\s*\[(.*)\]\s*$", graph_text, flags=re.M):
        content = m.group(1).strip()
        parts = [p.strip().strip("'\"") for p in content.split(",")]
        smiles_set.update(smi for smi in parts if smi)

    return smiles_set

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an elite expert in cheminformatics, green chemistry, and process engineering.

Your objective is to systematically evaluate multiple computational retrosynthetic routes
("Graphs") according to the 12 Principles of Green Chemistry, using deterministic
rule-based scoring.

You will receive:
  1) A plain-text string containing multiple retrosynthesis routes labeled
     "=== Graph 1 ===", "=== Graph 2 ===", etc.
  2) An "External Hazard Data" dictionary mapping SMILES to GHS H-codes.

Each graph uses this step format:
  Step N: <reaction type>
    Reactants: [SMILES, ...]
    Product: SMILES
    Atom economy: XX.XX%
    PMI: X.XX

You MUST process ALL graphs in the input. Do not truncate.

------------------------------------------------------------
I. REQUIRED PARSING (for each Graph)

For each graph extract:
  graph_id               – integer after "Graph "
  atom_economy_percent   – float from "Atom economy: XX.XX%"
  total_steps            – count of "Step n:" entries
  pmi_values             – list of PMI floats (skip steps that lack PMI)
  route_mean_PMI         – arithmetic mean of pmi_values
                           (0.0 if no PMI values found)

DATA VALIDATION:
  • atom_economy_percent > 100.0 is a physically impossible value and indicates
    a parsing anomaly. Flag it (add "AE_anomaly" to hazard_flags_detected) and
    cap AE_score at 1.0 for scoring: AE_score = min(atom_economy_percent / 100.0, 1.0)

------------------------------------------------------------
II. GREEN CHEMISTRY SCORING COMPONENTS

1) Mass Efficiency & Catalysis  (Principles 1, 2, 9)

  AE_score = min(atom_economy_percent / 100.0, 1.0)   ← cap at 1.0

  WP_score = 1.0 / (1.0 + route_mean_PMI)

  Catalytic classification — a step is catalytic when ANY of the following apply:
    a) The reaction type name matches a known catalytic transformation:
       Suzuki, Heck, Sonogashira, Buchwald–Hartwig, Ullmann,
       hydrogenation, transfer hydrogenation, metal-catalyzed
       oxidation/reduction, or other well-known named catalytic reactions.
    b) A catalyst SMILES is present in the reactant list
       (e.g. Pd/C, PdCl2, Pd(PPh3)4, RhCl3, RuCl3, Ni complexes, etc.).
    c) A known organocatalyst or enzyme is listed (DMAP, proline, lipase, etc.).
    Mark non-catalytic ONLY if none of (a)–(c) apply.

  CAT_score = catalytic_steps / total_steps
              (0.0 if total_steps = 0)

------------------------------------------------------------
2) Reduce Derivatives  (Principle 8)

  Scan intermediate SMILES for temporary protecting groups:
    Boc, Fmoc, Cbz, THP, TBDMS, MOM, Trityl.

  NOTE: Benzyl ether (Bn) groups are counted here as protecting groups,
  NOT as hazard flags. Do NOT include "Benzyl ether" in hazard_flags_detected.

  derivatization_cycles = number of complete protection–deprotection cycles detected
  DP_pen = 0.15 × derivatization_cycles

------------------------------------------------------------
3) Renewable Feedstocks  (Principle 7)

  Examine terminal leaf-node (starting material) SMILES.
  Check whether any is a primary derivative of the DOE Top-12 bio-based platform
  chemicals: succinic acid, lactic acid, glycerol, furfural, levulinic acid,
  itaconic acid, sorbitol.

  uses_renewable_feedstock = true if detected, else false
  REN_score = 0.10 if true, else 0.0

------------------------------------------------------------
4) Hazard & Accident Prevention  (Principles 3, 4, 12)

  A) Structural SMARTS scan of all reactant and product SMILES:
     Explosive/reactive:
       Azides:              [NX1]-[NX2+]=[NX1-]
       Peroxides:           [OX2]-[OX2]
       Poly-nitro aromatics
     Highly toxic/corrosive:
       Phosgene:            ClC(=O)Cl
       Methyl iodide:       CI
       Cyanides:            C#N
       Acid halides:        C(=O)[F,Cl,Br,I]

  B) External Hazard Data check — if any SMILES maps to severe GHS codes:
       Explosive:           H200–H211
       Fatal acute toxicity: H300, H310, H330

  HAZ_pen = 0.20 if severe hazards detected, else 0.0
  Record all detected hazard type names in hazard_flags_detected (string array).
  Do NOT include "Benzyl ether" here — see Section II-2.

------------------------------------------------------------
5) Solvents & Energy Efficiency  (Principles 5 & 6)

  If solvent or temperature data is present in the graph, apply:
    Energy-intensive: cryogenic (−78 °C organolithium) or prolonged high-T reflux.
    Problematic solvents: DCM, DMF, diethyl ether.
    Biocatalysis / aqueous enzymatic steps count as a bonus.

    ES_pen = 0.10   (energy-intensive or toxic-solvent dependent)
    ES_pen = −0.05  (aqueous biocatalysis confirmed)
    ES_pen = 0.0    (insufficient data — default)

  If no solvent or temperature data is available, set ES_pen = 0.0 and add
  "ES_data_unavailable" to hazard_flags_detected for transparency.

------------------------------------------------------------
6) Design for Degradation  (Principle 10)

  If the final product or major byproduct contains extensive polyfluorination
  (e.g. C(F)(F)F groups) with no degradable handles:
    DEG_pen = 0.05
  Else:
    DEG_pen = 0.0

------------------------------------------------------------
III. COMBINED GREEN SCORE

Compute after all components are finalized:

  overall_score =
      (0.25 × AE_score)
    + (0.25 × WP_score)
    + (0.15 × CAT_score)
    + REN_score
    − DP_pen
    − HAZ_pen
    − ES_pen
    − DEG_pen

  Clamp: overall_score = max(0.0, min(1.0, overall_score))

------------------------------------------------------------
IV. OUTPUT FORMAT (STRICT)

Output ONLY a valid JSON array. No markdown, no explanations,
no extra text, no trailing commas.
Do NOT include a "rank" field — ranking is computed in post-processing.

Each element MUST follow exactly:
{
  "graph_id": <int>,
  "atom_economy_percent": <float>,
  "catalytic_step_ratio": <float>,
  "route_mean_PMI": <float>,
  "derivatization_cycles": <int>,
  "uses_renewable_feedstock": <boolean>,
  "hazard_flags_detected": [<string>, ...],
  "overall_score": <float>
}

Process ALL graphs. Return strictly valid JSON only.
"""


def build_user_prompt(graph_text: str, hazard_json_str: str) -> str:
    return f"""Below is hazard context retrieved from PubChem, followed by the retrosynthesis graphs.
Each graph is labelled "=== Graph N ===" and contains steps in this format:
  Step N: <reaction type>
    Reactants: [SMILES, ...]
    Product: SMILES
    Atom economy: XX.XX%
    PMI: X.XX

External Hazard Data (GHS H-Codes from PubChem):
{hazard_json_str}

Retrosynthesis Graphs:
{graph_text}
"""

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _assign_ranks(results: list) -> list:
    """
    Assign ranks in Python based on overall_score (descending).
    Uses dense ranking: tied scores share the same rank,
    next rank is consecutive with no gaps.
    Replaces the LLM-generated rank which is unreliable.
    """
    sorted_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    rank = 1
    for i, entry in enumerate(sorted_results):
        if i > 0 and entry["overall_score"] < sorted_results[i - 1]["overall_score"]:
            rank = i + 1
        entry["rank"] = rank
    return sorted_results

# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_retrosynthesis_routes(graph_text: str, llm_client) -> list | None:
    """
    Full pipeline:
      1. Extract unique SMILES from graph_text.
      2. Query PubChem for GHS hazard H-codes (parallel, with retry).
      3. Build prompt and call the LLM.
      4. Parse JSON, validate, re-rank in Python, and return the result list.

    Returns a ranked list of dicts, or None on LLM parse failure.
    """
    # ------------------------------------------------------------------
    # Step 1 — SMILES extraction
    # ------------------------------------------------------------------
    print("[1/3] Extracting SMILES and querying PubChem…")
    unique_smiles = extract_unique_smiles(graph_text)
    print(f"  Unique SMILES found: {len(unique_smiles)}")

    # ------------------------------------------------------------------
    # Step 2 — Parallel PubChem lookups
    # ------------------------------------------------------------------
    hazard_context: dict = {}
    cid_cache: dict = {}
    ghs_cache: dict = {}

    smiles_to_lookup = [s for s in unique_smiles if len(s) >= 3]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_lookup_smiles, smi, cid_cache, ghs_cache): smi
            for smi in smiles_to_lookup
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  PubChem lookup"):
            smi, h_codes = future.result()
            if h_codes:
                hazard_context[smi] = h_codes

    hazard_json_str = json.dumps(hazard_context, indent=2, ensure_ascii=False)
    print(f"  Hazard entries retrieved: {len(hazard_context)}")

    # ------------------------------------------------------------------
    # Step 3 — LLM call
    # ------------------------------------------------------------------
    print("[2/3] Calling LLM…")
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(graph_text, hazard_json_str)},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw = response.choices[0].message.content

    # ------------------------------------------------------------------
    # Step 4 — Parse, validate, re-rank
    # ------------------------------------------------------------------
    print("[3/3] Parsing, validating and ranking…")
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        results = json.loads(clean)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] JSON parse failed: {exc}")
        print("Raw LLM output (first 2000 chars):\n", raw[:2000])
        return None

    results_ranked = _assign_ranks(results)

    top = results_ranked[0]
    print(f"\n  Evaluated {len(results_ranked)} routes.")
    print(
        f"  Best route: Graph {top['graph_id']}  "
        f"score={top['overall_score']:.3f}  "
        f"AE={top['atom_economy_percent']:.1f}%  "
        f"PMI={top['route_mean_PMI']:.2f}"
    )

    return results_ranked

# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def batch_evaluate(
    txt_dir: str = "./dataoutput",
    scores_dir: str = "./dataoutput",
    llm_client=None,
    skip_existing: bool = True,
) -> dict:
    """
    Process every .txt retrosynthesis graph file in txt_dir,
    run green chemistry evaluation, and save a _green_scores.json
    for each drug into scores_dir.

    Parameters
    ----------
    txt_dir       : folder containing <Drug>.txt files
    scores_dir    : folder where <Drug>_green_scores.json will be written
    llm_client    : initialised AzureOpenAI (or OpenAI) client
    skip_existing : if True, skip drugs whose _green_scores.json already exists

    Returns
    -------
    dict mapping drug_name -> list of ranked result dicts
    """
    txt_path    = Path(txt_dir)
    scores_path = Path(scores_dir)
    scores_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {txt_dir}/")
        return {}

    print(f"Found {len(txt_files)} .txt file(s) to evaluate.\n")

    all_results: dict = {}
    summary: list = []

    for txt_file in txt_files:
        drug_name   = txt_file.stem
        output_json = scores_path / f"{drug_name}_green_scores.json"

        # Skip if already evaluated
        if skip_existing and output_json.exists():
            print(f"[{drug_name}] Already exists — skipping.  ({output_json})")
            with open(output_json, encoding="utf-8") as f:
                all_results[drug_name] = json.load(f)
            summary.append({"drug": drug_name, "status": "skipped (cached)",
                             "routes": len(all_results[drug_name])})
            continue

        print(f"\n[{drug_name}] Reading {txt_file.name} …")
        with open(txt_file, encoding="utf-8") as f:
            graph_text = f.read()

        if not graph_text.strip():
            print("  [WARN] File is empty — skipping.")
            summary.append({"drug": drug_name, "status": "skipped (empty)", "routes": 0})
            continue

        try:
            results = evaluate_retrosynthesis_routes(graph_text, llm_client)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            summary.append({"drug": drug_name, "status": f"ERROR: {exc}", "routes": 0})
            continue

        if results is None:
            summary.append({"drug": drug_name, "status": "ERROR: LLM parse failed", "routes": 0})
            continue

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        all_results[drug_name] = results
        top = results[0]
        summary.append({
            "drug":       drug_name,
            "status":     "done",
            "routes":     len(results),
            "best_graph": top["graph_id"],
            "best_score": top["overall_score"],
        })
        print(f"  Saved {len(results)} routes → {output_json}")

    # Summary table
    print("\n" + "=" * 60)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Drug':<20} {'Status':<28} {'Routes':>6}  Best (graph / score)")
    print("-" * 60)
    for s in summary:
        best = ""
        if s["status"] == "done":
            best = f"Graph {s['best_graph']} / {s['best_score']:.3f}"
        print(f"  {s['drug']:<20} {s['status']:<28} {s['routes']:>6}  {best}")
    print("=" * 60)

    return all_results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_client() -> AzureOpenAI:
    """Initialise AzureOpenAI client from environment variables."""
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=API_VERSION,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Green Chemistry Retrosynthesis Evaluator"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", metavar="FILE",
        help="Path to a single .txt graph file (single-drug mode)"
    )
    mode.add_argument(
        "--batch", metavar="DIR",
        help="Path to a folder of .txt files (batch mode)"
    )
    parser.add_argument(
        "--output", metavar="DIR", default="./dataoutput",
        help="Output directory for _green_scores.json files (default: ./dataoutput)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if _green_scores.json already exists"
    )
    args = parser.parse_args()

    client = _build_client()

    # ── Single-drug mode ──────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        with open(input_path, encoding="utf-8") as f:
            graph_text = f.read()

        print(f"Evaluating: {input_path.name}\n")
        results = evaluate_retrosynthesis_routes(graph_text, client)
        if results is None:
            print("Evaluation failed.")
            return

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{input_path.stem}_green_scores.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {out_path}")

    # ── Batch mode ────────────────────────────────────────────────────
    else:
        batch_evaluate(
            txt_dir      = args.batch,
            scores_dir   = args.output,
            llm_client   = client,
            skip_existing= not args.force,
        )


if __name__ == "__main__":
    main()
