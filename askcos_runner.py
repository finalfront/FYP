"""
askcos_runner.py
================
Upstream automation script: submit retrosynthesis jobs to a local AskCOS
instance for each drug, poll until complete, convert the MCTS result trees
into the graph JSON format expected by extraction.py, and save to data/.

Full pipeline position
----------------------
  askcos_runner.py  →  data/<Drug>.json
  batch_process.py  →  dataoutput/<Drug>.txt
  green_evaluator.py → dataoutput/<Drug>_green_scores.json

Setup
-----
  1. Deploy AskCOS v2 locally (Docker Compose):
       https://askcos-docs.mit.edu/guide/1-Introduction/1.1-Introduction.html

  2. (Optional) set credentials for authenticated endpoints:
       export ASKCOS_USERNAME=your_user
       export ASKCOS_PASSWORD=your_password

  3. Run:
       python askcos_runner.py                     # all 10 drugs
       python askcos_runner.py --drugs Aspirin     # single drug
       python askcos_runner.py --list              # show available drugs
       python askcos_runner.py --force             # re-run even if JSON exists
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Configuration  ← edit HOST / PORT to match your AskCOS deployment
# ---------------------------------------------------------------------------

HOST = os.environ.get("ASKCOS_HOST", "http://0.0.0.0")
PORT = os.environ.get("ASKCOS_PORT", "9100")
BASE_URL = f"{HOST}:{PORT}"

# Authentication (only needed for /call-sync endpoints; not needed for
# /call-sync-without-token which is used by default)
ASKCOS_USERNAME = os.environ.get("ASKCOS_USERNAME", "")
ASKCOS_PASSWORD = os.environ.get("ASKCOS_PASSWORD", "")

# MCTS tree search parameters
MCTS_PARAMS = {
    "retro_backend_options": [
        {
            "retro_backend": "template_relevance",
            "retro_model_name": "reaxys",
            "max_num_templates": 1000,
            "max_cum_prob": 0.999,
            "attribute_filter": [],
            "threshold": 0.1,
            "top_k": 50,
        }
    ],
    "banned_chemicals": [],
    "banned_reactions": [],
    "use_fast_filter": True,
    "fast_filter_threshold": 0.75,
    "retro_rerank_backend": "relevance_heuristic",
    "cluster_precursors": True,
    "cluster_setting": {
        "feature": "original",
        "cluster_method": "hdbscan",
        "fp_type": "morgan",
        "fp_length": 512,
        "fp_radius": 1,
        "classification_threshold": 0.2,
    },
    # Tree search limits
    "max_depth": 6,           # max retrosynthesis depth
    "max_branching": 20,      # max branching factor per node
    "expansion_time": 60,     # seconds per target (increase for complex molecules)
    "max_trees": 200,         # max number of routes to return
    "buyable_logic": "and",
    "max_ppg": 100,
    "max_scscore": 6,
    "max_elements": {},
    "return_first": False,
}

# Polling configuration (for async calls)
POLL_INTERVAL = 5    # seconds between status checks
POLL_TIMEOUT  = 900  # max seconds to wait per drug (15 min)

# Output directory
DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Drug SMILES registry
# ---------------------------------------------------------------------------

DRUGS: dict[str, str] = {
    "Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol":  "CC(=O)Nc1ccc(O)cc1",
    "Phenacetin":   "CCOC1=CC=C(NC(C)=O)C=C1",
    "Benzocaine":   "CCOC(=O)c1ccc(N)cc1",
    "Ibuprofen":    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Lidocaine":    "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "Celecoxib":    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Erlotinib":    "C#Cc1cccc(Nc2ncnc3cc(OCCO)c(OCCO)cc23)c1",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Favipiravir":  "NC(=O)c1ncc(F)c(=O)[nH]1",
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
_TOKEN: str | None = None


def _authenticate() -> str | None:
    """Obtain a bearer token from /api/admin/token. Returns token or None."""
    if not (ASKCOS_USERNAME and ASKCOS_PASSWORD):
        return None
    url = f"{BASE_URL}/api/admin/token"
    resp = SESSION.post(
        url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"username": ASKCOS_USERNAME, "password": ASKCOS_PASSWORD},
        timeout=15,
    )
    if resp.status_code == 200:
        token = resp.json().get("access_token")
        print(f"  [AUTH] Token obtained successfully.")
        return token
    print(f"  [AUTH WARN] Token request failed ({resp.status_code}). "
          f"Falling back to no-token endpoint.")
    return None


def _headers_with_token(token: str | None) -> dict:
    if token:
        return {"cookie": f"access_token=bearer {token};"}
    return {}


def _post(endpoint: str, payload: dict, token: str | None = None,
          timeout: int = 120) -> requests.Response:
    url = f"{BASE_URL}{endpoint}"
    return SESSION.post(
        url,
        json=payload,
        headers=_headers_with_token(token),
        timeout=timeout,
    )


def _get(endpoint: str, token: str | None = None, timeout: int = 30) -> requests.Response:
    url = f"{BASE_URL}{endpoint}"
    return SESSION.get(url, headers=_headers_with_token(token), timeout=timeout)


# ---------------------------------------------------------------------------
# AskCOS MCTS — submit and poll
# ---------------------------------------------------------------------------

def submit_mcts_sync(smiles: str, token: str | None) -> dict | None:
    """
    Submit a synchronous MCTS tree search.
    Uses /call-sync-without-token when no token is available,
    /call-sync otherwise.

    Returns the raw result dict, or None on failure.
    """
    payload = {"smiles": smiles, **MCTS_PARAMS}

    if token:
        endpoint = "/api/tree-search/mcts/call-sync"
    else:
        endpoint = "/api/tree-search/mcts/call-sync-without-token"

    print(f"  [MCTS] Submitting sync job to {endpoint} …")
    try:
        resp = _post(endpoint, payload, token=token,
                     timeout=MCTS_PARAMS["expansion_time"] + 30)
        if resp.status_code == 200:
            return resp.json()
        print(f"  [ERROR] Sync call returned HTTP {resp.status_code}: {resp.text[:300]}")
    except requests.exceptions.Timeout:
        print("  [WARN] Sync call timed out — switching to async mode.")
    except Exception as exc:
        print(f"  [ERROR] Sync call failed: {exc}")
    return None


def submit_mcts_async(smiles: str, token: str | None) -> str | None:
    """
    Submit an asynchronous MCTS job.
    Returns a task_id string, or None on failure.
    """
    payload = {"smiles": smiles, **MCTS_PARAMS}

    if token:
        endpoint = "/api/tree-search/mcts/call-async"
    else:
        # Try no-token async endpoint; fall back to sync-without-token
        endpoint = "/api/tree-search/mcts/call-async-without-token"

    print(f"  [MCTS] Submitting async job to {endpoint} …")
    try:
        resp = _post(endpoint, payload, token=token, timeout=30)
        if resp.status_code in (200, 202):
            data = resp.json()
            task_id = data.get("task_id") or data.get("id")
            if task_id:
                print(f"  [MCTS] Task ID: {task_id}")
                return task_id
        print(f"  [ERROR] Async submit returned HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        print(f"  [ERROR] Async submit failed: {exc}")
    return None


def poll_mcts_result(task_id: str, token: str | None) -> dict | None:
    """
    Poll /api/tree-search/mcts/result/<task_id> until done or timeout.
    Returns result dict or None.
    """
    endpoint = f"/api/tree-search/mcts/result/{task_id}"
    deadline = time.time() + POLL_TIMEOUT

    print(f"  [POLL] Waiting for task {task_id} …", end="", flush=True)
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            resp = _get(endpoint, token=token)
            if resp.status_code == 200:
                data = resp.json()
                state = data.get("status") or data.get("state", "")
                if state.upper() in ("SUCCESS", "DONE", "COMPLETED"):
                    print(" done.")
                    return data.get("result") or data
                elif state.upper() in ("FAILURE", "ERROR", "FAILED"):
                    print(f" FAILED: {data}")
                    return None
                else:
                    print(".", end="", flush=True)
            else:
                print(f"\n  [WARN] Poll returned HTTP {resp.status_code}")
        except Exception as exc:
            print(f"\n  [WARN] Poll error: {exc}")

    print(f"\n  [TIMEOUT] Task {task_id} did not complete within {POLL_TIMEOUT}s.")
    return None


def run_mcts(drug_name: str, smiles: str, token: str | None) -> dict | None:
    """
    Run MCTS for a target SMILES. Tries sync first, falls back to async.
    Returns the raw AskCOS result dict or None.
    """
    # Try synchronous first
    result = submit_mcts_sync(smiles, token)
    if result:
        return result

    # Fall back to async
    task_id = submit_mcts_async(smiles, token)
    if task_id:
        return poll_mcts_result(task_id, token)

    print(f"  [ERROR] All MCTS submission strategies failed for {drug_name}.")
    return None


# ---------------------------------------------------------------------------
# Chemistry helpers (for graph metadata calculation)
# ---------------------------------------------------------------------------

def mol_weight(smiles: str) -> float:
    """Return exact molecular weight for a SMILES string, or 0.0 on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return Descriptors.ExactMolWt(mol)


def compute_atom_economy(reaction_smiles: str) -> float:
    """
    Atom economy = MW(desired product) / sum(MW(all reactants)).
    reaction_smiles format: "R1.R2>>P"
    Returns value in range [0, ~1.2] (>1.0 indicates data anomaly).
    Returns 0.0 on parse failure.
    """
    try:
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            return 0.0
        reactant_smiles = [s.strip() for s in parts[0].split(".") if s.strip()]
        product_smiles  = parts[1].strip().split(".")[0]  # take first product

        reactant_mass = sum(mol_weight(s) for s in reactant_smiles)
        product_mass  = mol_weight(product_smiles)

        if reactant_mass == 0:
            return 0.0
        return product_mass / reactant_mass
    except Exception:
        return 0.0


def compute_precursor_cost(leaf_smiles: list[str]) -> float:
    """
    Estimate precursor cost as sum of SC-scores (simplified: use molecular
    weight as proxy when SC-score API is unavailable).
    Positive values; lower = cheaper/more accessible.
    """
    # Idealiy call /api/scscore/ for each SMILES.
    # As a robust fallback: use log(MW) as a rough accessibility proxy.
    total = 0.0
    for smi in leaf_smiles:
        mw = mol_weight(smi)
        if mw > 0:
            total += math.log(mw + 1)
    return round(total, 4)


# ---------------------------------------------------------------------------
# AskCOS result → project JSON format conversion
# ---------------------------------------------------------------------------

TARGET_ID = "00000000-0000-0000-0000-000000000000"


def _new_id() -> str:
    return str(uuid.uuid4())


def _parse_tree_node(node: dict, parent_chemical_id: str,
                     nodes: list, edges: list,
                     depth_counter: list[int],
                     scores: list[float], plausibilities: list[float],
                     leaf_smiles: list[str]):
    """
    Recursively walk an AskCOS MCTS path node and populate nodes/edges.

    AskCOS path node structure:
      {
        "smiles": "...",          # SMILES of this chemical
        "type": "chemical",
        "children": [             # list of reaction nodes
          {
            "smiles": "R1.R2>>P", # reaction SMILES
            "type": "reaction",
            "template_score": 0.4,
            "plausibility": 0.99,
            "children": [...]     # list of reactant chemical nodes
          }
        ],
        "is_chemical": true,
        "terminal": true/false
      }
    """
    children = node.get("children") or []

    if not children:
        # Leaf node (starting material / building block)
        leaf_smiles.append(node.get("smiles", ""))
        return

    # This chemical node has one or more reaction children
    for rxn_node in children:
        rxn_smiles = rxn_node.get("smiles", "")
        score       = float(rxn_node.get("template_score", rxn_node.get("score", 0.0)))
        plausibility = float(rxn_node.get("plausibility", rxn_node.get("forward_score", 0.0)))

        scores.append(score)
        plausibilities.append(plausibility)
        depth_counter[0] = max(depth_counter[0], 1)

        rxn_id = _new_id()
        nodes.append({"id": rxn_id, "smiles": rxn_smiles, "type": "reaction"})
        # Edge: parent chemical → reaction
        edges.append({"from": parent_chemical_id, "to": rxn_id, "id": _new_id()})

        # Reactant chemical nodes
        reactant_children = rxn_node.get("children") or []
        for reactant_node in reactant_children:
            chem_smiles = reactant_node.get("smiles", "")
            chem_id = _new_id()
            nodes.append({"id": chem_id, "smiles": chem_smiles, "type": "chemical"})
            # Edge: reaction → reactant chemical
            edges.append({"from": rxn_id, "to": chem_id, "id": _new_id()})
            # Recurse
            _parse_tree_node(
                reactant_node, chem_id, nodes, edges,
                depth_counter, scores, plausibilities, leaf_smiles
            )


def path_to_graph(path: dict, target_smiles: str) -> dict:
    """
    Convert one AskCOS MCTS path tree into the project graph JSON format.

    Returns a dict with keys: directed, multigraph, graph, nodes, edges.
    """
    nodes: list = []
    edges: list = []
    depth_counter = [0]
    scores: list[float] = []
    plausibilities: list[float] = []
    leaf_smiles: list[str] = []

    # Root node is always the target molecule with fixed ID
    nodes.append({"id": TARGET_ID, "smiles": target_smiles, "type": "chemical"})

    _parse_tree_node(
        path, TARGET_ID, nodes, edges,
        depth_counter, scores, plausibilities, leaf_smiles
    )

    num_reactions = sum(1 for n in nodes if n["type"] == "reaction")

    # Atom economy — use the first reaction in the route (root reaction)
    root_reaction_smiles = next(
        (n["smiles"] for n in nodes if n["type"] == "reaction"), ""
    )
    ae = compute_atom_economy(root_reaction_smiles)

    # Aggregate scores
    avg_score        = float(sum(scores) / len(scores))       if scores        else 0.0
    avg_plausibility = float(sum(plausibilities) / len(plausibilities)) if plausibilities else 0.0
    min_score        = float(min(scores))        if scores        else 0.0
    min_plausibility = float(min(plausibilities)) if plausibilities else 0.0
    first_score      = float(scores[0])          if scores        else 0.0
    first_plaus      = float(plausibilities[0])  if plausibilities else 0.0

    precursor_cost = compute_precursor_cost(leaf_smiles)

    graph_meta = {
        "depth":                   num_reactions,
        "precursor_cost":          precursor_cost,
        "score":                   None,
        "cluster_id":              None,
        "first_step_score":        round(first_score, 8),
        "first_step_plausibility": round(first_plaus, 8),
        "num_reactions":           num_reactions,
        "avg_score":               round(avg_score, 8),
        "avg_plausibility":        round(avg_plausibility, 8),
        "min_score":               round(min_score, 8),
        "min_plausibility":        round(min_plausibility, 8),
        "atom_economy":            round(ae, 10),
    }

    return {
        "directed":   True,
        "multigraph": False,
        "graph":      graph_meta,
        "nodes":      nodes,
        "edges":      edges,
    }


def convert_askcos_result(raw_result: dict, target_smiles: str) -> list[dict]:
    """
    Convert an AskCOS MCTS raw result dict into a list of project graph dicts.

    AskCOS v2 MCTS result shape (common variants):
      { "paths": [...] }
      { "result": { "paths": [...] } }
      { "trees": [...] }
      [ {...}, {...} ]   (already a list of paths)
    """
    # Unwrap common nesting patterns
    if isinstance(raw_result, list):
        paths = raw_result
    elif isinstance(raw_result, dict):
        if "paths" in raw_result:
            paths = raw_result["paths"]
        elif "result" in raw_result and isinstance(raw_result["result"], dict):
            inner = raw_result["result"]
            paths = inner.get("paths") or inner.get("trees") or []
        elif "trees" in raw_result:
            paths = raw_result["trees"]
        else:
            # Sometimes the result is wrapped one more level under "result"
            result_val = raw_result.get("result", raw_result)
            paths = result_val if isinstance(result_val, list) else []
    else:
        paths = []

    if not paths:
        print("  [WARN] No paths found in AskCOS result. Check raw response.")
        return []

    graphs = []
    for path in paths:
        try:
            g = path_to_graph(path, target_smiles)
            if g["graph"]["num_reactions"] > 0:
                graphs.append(g)
        except Exception as exc:
            print(f"  [WARN] Could not convert path: {exc}")

    return graphs


# ---------------------------------------------------------------------------
# SCScore lookup (optional enhancement)
# ---------------------------------------------------------------------------

def fetch_scscore(smiles: str, token: str | None) -> float | None:
    """
    Query /api/scscore/ for a SMILES string.
    Returns score (1–5 scale) or None on failure.
    This is used to improve precursor_cost accuracy over the MW-proxy fallback.
    """
    try:
        resp = _post(
            "/api/scscore/call-sync-without-token",
            {"smiles": smiles},
            token=token,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("result", {}).get("score") or data.get("score", 0))
    except Exception:
        pass
    return None


def enrich_precursor_costs(graphs: list[dict], token: str | None) -> None:
    """
    Optionally replace the MW-proxy precursor_cost with real SCScores.
    Modifies graphs in-place. Skips gracefully if the API is unavailable.
    """
    # Quick availability check
    try:
        resp = SESSION.get(f"{BASE_URL}/api/scscore/call-sync-without-token",
                           timeout=5)
        if resp.status_code == 404:
            return  # endpoint not available
    except Exception:
        return

    print("  [SCSCORE] Enriching precursor costs with SCScore API …")
    for g in graphs:
        leaf_smiles = [
            n["smiles"] for n in g["nodes"]
            if n["type"] == "chemical" and n["id"] != TARGET_ID
            and not any(e["to"] == n["id"] or e["from"] != TARGET_ID
                        for e in g["edges"] if e["from"] == n["id"])
        ]
        if not leaf_smiles:
            continue
        scores = [fetch_scscore(smi, token) for smi in leaf_smiles]
        valid  = [s for s in scores if s is not None]
        if valid:
            g["graph"]["precursor_cost"] = round(sum(valid), 4)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_drug(drug_name: str, smiles: str, token: str | None,
                 force: bool = False) -> bool:
    """
    Run the full pipeline for one drug and save data/<drug_name>.json.
    Returns True on success.
    """
    output_path = DATA_DIR / f"{drug_name}.json"

    if not force and output_path.exists():
        print(f"[{drug_name}] Already exists — skipping.  ({output_path})")
        return True

    print(f"\n{'='*55}")
    print(f"[{drug_name}]  SMILES: {smiles}")
    print(f"{'='*55}")

    # 1. Run MCTS
    raw_result = run_mcts(drug_name, smiles, token)
    if raw_result is None:
        print(f"  [FAIL] No result for {drug_name}.")
        return False

    # 2. Convert to project format
    print("  [CONVERT] Converting paths to graph format …")
    graphs = convert_askcos_result(raw_result, smiles)

    if not graphs:
        # Save the raw result for manual inspection
        raw_path = DATA_DIR / f"{drug_name}_raw.json"
        raw_path.write_text(json.dumps(raw_result, indent=2, ensure_ascii=False))
        print(f"  [WARN] Zero usable graphs. Raw result saved to {raw_path}")
        return False

    print(f"  [OK] Converted {len(graphs)} route(s).")

    # 3. (Optional) enrich precursor costs with SCScore
    enrich_precursor_costs(graphs, token)

    # 4. Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graphs, indent=2, ensure_ascii=False))
    print(f"  [SAVED] {len(graphs)} routes → {output_path}")
    return True


def batch_process(drug_names: list[str], force: bool = False) -> None:
    """Process a list of drugs and print a final summary."""
    # Authenticate once
    global _TOKEN
    _TOKEN = _authenticate()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}

    for name in drug_names:
        if name not in DRUGS:
            print(f"[WARN] Unknown drug '{name}'. Available: {list(DRUGS)}")
            results[name] = "unknown"
            continue
        smiles = DRUGS[name]
        ok = process_drug(name, smiles, _TOKEN, force=force)
        results[name] = "done" if ok else "FAILED"

    # Summary
    print("\n" + "=" * 55)
    print("BATCH SUMMARY")
    print("=" * 55)
    for name, status in results.items():
        out = DATA_DIR / f"{name}.json"
        size = f"{out.stat().st_size // 1024} KB" if out.exists() else "—"
        print(f"  {name:<20} {status:<10}  {size}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AskCOS upstream runner — generate retrosynthesis JSON for all drugs."
    )
    parser.add_argument(
        "--drugs", nargs="+", metavar="NAME",
        help="Drug name(s) to process (default: all). E.g. --drugs Aspirin Ibuprofen"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available drugs and their SMILES, then exit."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if data/<Drug>.json already exists."
    )
    parser.add_argument(
        "--host", default=None,
        help="AskCOS host URL (overrides ASKCOS_HOST env var). E.g. http://localhost"
    )
    parser.add_argument(
        "--port", default=None,
        help="AskCOS port (overrides ASKCOS_PORT env var). E.g. 9100"
    )
    parser.add_argument(
        "--time", type=int, default=None,
        help="MCTS expansion time in seconds per drug (default: 60)."
    )
    parser.add_argument(
        "--max-routes", type=int, default=None,
        help="Maximum number of routes per drug (default: 200)."
    )
    args = parser.parse_args()

    # Override globals from CLI flags
    global BASE_URL, MCTS_PARAMS
    if args.host:
        os.environ["ASKCOS_HOST"] = args.host
    if args.port:
        os.environ["ASKCOS_PORT"] = args.port
    if args.host or args.port:
        h = os.environ.get("ASKCOS_HOST", "http://0.0.0.0")
        p = os.environ.get("ASKCOS_PORT", "9100")
        BASE_URL = f"{h}:{p}"
    if args.time:
        MCTS_PARAMS["expansion_time"] = args.time
    if args.max_routes:
        MCTS_PARAMS["max_trees"] = args.max_routes

    if args.list:
        print(f"\n{'Drug':<20} SMILES")
        print("-" * 70)
        for name, smi in DRUGS.items():
            out = DATA_DIR / f"{name}.json"
            status = f"[exists {out.stat().st_size // 1024}KB]" if out.exists() else ""
            print(f"  {name:<18} {smi[:45]}  {status}")
        print()
        return

    drug_names = args.drugs if args.drugs else list(DRUGS.keys())
    batch_process(drug_names, force=args.force)


if __name__ == "__main__":
    main()
