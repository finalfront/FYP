import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import textwrap
import argparse 
import sys
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------- Configuration ----------
DEFAULT_FILES = {
    "Celecoxib": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Celecoxib.json",
    "Ibuprofen": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Ibuprofen.json",
    "Aspirin": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Aspirin.json",
    "Benzocaine": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Benzocaine.json",
    "Favipiravir": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Favipiravir.json",
    "Paracetamol": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Paracetamol.json",
    "Phenacetin": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Phenacetin.json",
    "Erlotinib": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Erlotinib.json",
    "Lidocaine": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Lidocaine.json",
    "Atrovastatin": r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Atrovastatin.json"
}
DEFAULT_OUTDIR = r"C:\Users\qsijw\Desktop\Fall 2025-2026\FYP\Github\Output"
DEFAULT_MAXPLOTS = 5

# ---------- Dataclasses for structured parsing ----------
@dataclass
class Node:
    id: str
    smiles: str
    type: str  # 'chemical' or 'reaction'
    def is_reaction(self) -> bool:
        return self.type == 'reaction'
    def is_chemical(self) -> bool:
        return self.type == 'chemical'

@dataclass
class Edge:
    from_id: str
    to_id: str
    id: str

@dataclass
class GraphMetadata:
    depth: int
    precursor_cost: float
    num_reactions: int
    first_step_score: float
    first_step_plausibility: float
    avg_score: float
    avg_plausibility: float
    min_score: float
    min_plausibility: float
    atom_economy: float
    score: Optional[float] = None
    cluster_id: Optional[Any] = None

@dataclass
class RetrosynthesisGraph:
    directed: bool
    multigraph: bool
    metadata: GraphMetadata
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    def get_reactions(self) -> List[Node]:
        return [node for node in self.nodes if node.is_reaction()]
    def get_chemicals(self) -> List[Node]:
        return [node for node in self.nodes if node.is_chemical()]
    def get_target_molecule(self) -> Optional[Node]:
        return self.get_node_by_id("00000000-0000-0000-0000-000000000000")
    def get_precursors(self, node_id: str) -> List[Node]:
        precursor_ids = [edge.to_id for edge in self.edges if edge.from_id == node_id]
        return [self.get_node_by_id(pid) for pid in precursor_ids if self.get_node_by_id(pid)]
    def get_product(self, node_id: str) -> Optional[Node]:
        for edge in self.edges:
            if edge.to_id == node_id:
                return self.get_node_by_id(edge.from_id)
        return None
    def traverse_by_depth(self) -> Dict[int, List[Node]]:
        depth_map = {}
        visited = set()
        def dfs(node_id: str, depth: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.get_node_by_id(node_id)
            if node:
                if depth not in depth_map:
                    depth_map[depth] = []
                depth_map[depth].append(node)
                for precursor in self.get_precursors(node_id):
                    dfs(precursor.id, depth + 1)
        target = self.get_target_molecule()
        if target:
            dfs(target.id, 0)
        return depth_map
    def get_reaction_steps(self) -> List[Dict[str, Any]]:
        steps = []
        for reaction_node in self.get_reactions():
            product = self.get_product(reaction_node.id)
            reactants = self.get_precursors(reaction_node.id)
            steps.append({
                'reaction': reaction_node,
                'product': product,
                'reactants': reactants,
                'smiles': reaction_node.smiles
            })
        return steps

class RetrosynthesisParser:
    @staticmethod
    def parse(data: List[Dict[str, Any]]) -> List[RetrosynthesisGraph]:
        graphs = []
        for graph_data in data:
            graphs.append(RetrosynthesisParser.parse_single(graph_data))
        return graphs
    @staticmethod
    def parse_single(graph_data: Dict[str, Any]) -> RetrosynthesisGraph:
        graph_info = graph_data['graph']
        metadata = GraphMetadata(
            depth=graph_info['depth'],
            precursor_cost=graph_info['precursor_cost'],
            num_reactions=graph_info['num_reactions'],
            first_step_score=graph_info['first_step_score'],
            first_step_plausibility=graph_info['first_step_plausibility'],
            avg_score=graph_info['avg_score'],
            avg_plausibility=graph_info['avg_plausibility'],
            min_score=graph_info['min_score'],
            min_plausibility=graph_info['min_plausibility'],
            atom_economy=graph_info['atom_economy'],
            score=graph_info.get('score'),
            cluster_id=graph_info.get('cluster_id')
        )
        nodes = [Node(id=n['id'], smiles=n['smiles'], type=n['type']) for n in graph_data['nodes']]
        edges = [Edge(from_id=e['from'], to_id=e['to'], id=e['id']) for e in graph_data['edges']]
        return RetrosynthesisGraph(
            directed=graph_data['directed'],
            multigraph=graph_data['multigraph'],
            metadata=metadata,
            nodes=nodes,
            edges=edges
        )

# ---------- Utility Functions ----------
def load_routes(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "routes" in data and isinstance(data["routes"], list):
            data = data["routes"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {path}")
    cleaned = []
    for r in data:
        if "nodes" in r and "edges" in r:
            cleaned.append(r)
    return cleaned

def build_nx_graph(route: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in route.get("nodes", []):
        G.add_node(n["id"], **n)
    for e in route.get("edges", []):
        frm = e.get("from") or e.get("source")
        to = e.get("to") or e.get("target")
        if frm is None or to is None:
            continue
        G.add_edge(frm, to, **{k: v for k, v in e.items() if k not in ("from","to","source","target")})
    return G

def graph_metrics(route: Dict[str, Any]) -> Dict[str, Any]:
    g = route.get("graph", {})
    return {
        "depth": g.get("depth"),
        "precursor_cost": g.get("precursor_cost"),
        "num_reactions": g.get("num_reactions"),
        "atom_economy": g.get("atom_economy"),
        "avg_score": g.get("avg_score"),
        "avg_plausibility": g.get("avg_plausibility"),
        "min_score": g.get("min_score"),
        "min_plausibility": g.get("min_plausibility"),
        "first_step_score": g.get("first_step_score"),
        "first_step_plausibility": g.get("first_step_plausibility"),
        "cluster_id": g.get("cluster_id"),
    }

def annotate_smiles(smiles: str, max_len: int = 60) -> str:
    if not smiles:
        return ""
    s = smiles.replace("\n", " ")
    return textwrap.shorten(s, width=max_len, placeholder="…")



def find_root(route: Dict[str, Any]) -> str:
    zero_id = "00000000-0000-0000-0000-000000000000"
    ids = {n["id"] for n in route["nodes"]}
    if zero_id in ids:
        return zero_id
    G_tmp = nx.DiGraph()
    for e in route["edges"]:
        frm = e.get("from") or e.get("source")
        to = e.get("to") or e.get("target")
        if frm is not None and to is not None:
            G_tmp.add_edge(frm, to)
    candidates = [n["id"] for n in route["nodes"] if G_tmp.in_degree(n["id"]) == 0 and n.get("type") == "chemical"]
    return candidates[0] if candidates else route["nodes"][0]["id"]

def layered_layout(G: nx.DiGraph, root_id: str) -> Dict[Any, Tuple[float, float]]:
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        order = list(G.nodes())
    layers: Dict[int, List[Any]] = {}
    for node in order:
        try:
            dist = nx.shortest_path_length(G, source=root_id, target=node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = 0
        layers.setdefault(dist, []).append(node)
    pos = {}
    x_gap, y_gap = 0.4, 0.2
    for layer_idx, nodes in sorted(layers.items()):
        for i, n in enumerate(nodes):
            pos[n] = (layer_idx * x_gap, -i * y_gap)
    return pos

def draw_route(route: Dict[str, Any], title: str, outpath: Path) -> None:
    G = build_nx_graph(route)
    if len(G) == 0:
        return
    root = find_root(route)
    pos = layered_layout(G, root)
    chem_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "chemical"]
    rxn_nodes  = [n for n, d in G.nodes(data=True) if d.get("type") == "reaction"]
    plt.figure(figsize=(20, 12))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=14)
    nx.draw_networkx_nodes(G, pos, nodelist=chem_nodes, node_shape='o', node_size=6000)
    nx.draw_networkx_nodes(G, pos, nodelist=rxn_nodes, node_shape='s', node_size=6000)
    labels = {n: ("Rxn" if G.nodes[n].get("type") != "chemical" else annotate_smiles(G.nodes[n].get("smiles","")))
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def summarize_routes(name: str, routes: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(routes, start=1):
        m = graph_metrics(r)
        n_chem = sum(1 for n in r.get("nodes", []) if n.get("type") == "chemical")
        n_rxn  = sum(1 for n in r.get("nodes", []) if n.get("type") == "reaction")
         # Product/reactants extraction for PMI Calculation 
        step_rows = extract_stepwise_smiles(r)
        if step_rows:
            product_smiles = step_rows[0]['product_smiles']
            all_reactant_smiles = []
            for s in step_rows:
                all_reactant_smiles += s['reactant_smiles']
            pmi = calc_pmi(product_smiles, all_reactant_smiles)
        else:
            pmi = None

        rows.append({
            "target": name,
            "route": idx,
            "depth": m.get("depth"),
            "num_reactions": m.get("num_reactions"),
            "chem_nodes": n_chem,
            "rxn_nodes": n_rxn,
            "precursor_cost": m.get("precursor_cost"),
            "atom_economy": m.get("atom_economy"),
            "avg_score": m.get("avg_score"),
            "avg_plausibility": m.get("avg_plausibility"),
            "first_step_score": m.get("first_step_score"),
            "first_step_plausibility": m.get("first_step_plausibility"),
            "PMI": pmi,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target","depth","precursor_cost","route"], na_position="last").reset_index(drop=True)
    return df

def extract_stepwise_smiles(route: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract step-by-step SMILES for each reaction step in the route."""
    parser = RetrosynthesisParser()
    graph = parser.parse_single(route)
    steps = graph.get_reaction_steps()
    stepwise = []
    for i, step in enumerate(steps, 1):
        stepwise.append({
            "step": i,
            "product_smiles": step['product'].smiles if step['product'] else None,
            "reactant_smiles": [r.smiles for r in step['reactants']],
            "reaction_smiles": step['reaction'].smiles,
        })
    return stepwise

def print_stepwise_smiles(name: str, route: Dict[str, Any], route_idx: int):
    print(f"\n=== {name} — Route {route_idx} Stepwise SMILES ===")
    steps = extract_stepwise_smiles(route)
    for step in steps:
        print(f"Step {step['step']}:")
        print(f"  Product: {step['product_smiles']}")
        print(f"  Reactants: {step['reactant_smiles']}")
        print(f"  Reaction: {step['reaction_smiles']}")

def process_targets(files: Dict[str, str], out_dir: Path, max_routes_to_plot: int = 3) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for name, path in files.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] File not found: {p}", file=sys.stderr)
            continue
        routes = load_routes(str(p))
        if not routes:
            print(f"[WARN] No routes parsed: {p}", file=sys.stderr)
            continue
        summary_df = summarize_routes(name, routes)
        all_summaries.append(summary_df)
        to_plot = min(max_routes_to_plot, len(routes))
        for i in range(to_plot):
            img_path = out_dir / f"{name}_route_{i+1}.png"
            draw_route(routes[i], f"{name} — Route {i+1}", img_path)
            print(f"[OK] Saved {img_path}")
            print_stepwise_smiles(name, routes[i], i+1)
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
    else:
        combined = pd.DataFrame()
    return combined

def smiles_to_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol) if mol else 0.0

def calc_pmi(product_smiles, reactant_smiles_list):
    prod_mass = smiles_to_mol_weight(product_smiles)
    react_mass_sum = sum(smiles_to_mol_weight(s) for s in reactant_smiles_list)
    if prod_mass == 0:
        return None
    return react_mass_sum / prod_mass

def cli_entry():
    parser = argparse.ArgumentParser(description="Retrosynthesis visualizer", add_help=True)
    parser.add_argument("--celecoxib", type=str, default=DEFAULT_FILES["Celecoxib"])
    parser.add_argument("--ibuprofen", type=str, default=DEFAULT_FILES["Ibuprofen"])
    parser.add_argument("--aspirin", type=str, default=DEFAULT_FILES["Aspirin"])
    parser.add_argument("--favipiravir", type=str, default=DEFAULT_FILES["Favipiravir"])
    parser.add_argument("--paracetamol", type=str, default=DEFAULT_FILES["Paracetamol"])
    parser.add_argument("--phenacetin", type=str, default=DEFAULT_FILES["Phenacetin"])
    parser.add_argument("--benzocaine", type=str, default=DEFAULT_FILES["Benzocaine"])
    parser.add_argument("--Atrovastatin", type=str, default=DEFAULT_FILES["Atrovastatin"])
    parser.add_argument("--Lidocaine", type=str, default=DEFAULT_FILES["Lidocaine"])
    parser.add_argument("--Erlotinib", type=str, default=DEFAULT_FILES["Erlotinib"])
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--max_plots", type=int, default=DEFAULT_MAXPLOTS)
    args, _unknown = parser.parse_known_args()
    files = {"Celecoxib": args.celecoxib, "Ibuprofen": args.ibuprofen, 
             "Aspirin": args.aspirin, "Favipiravir": args.favipiravir, "Paracetamol": args.paracetamol, 
             "Phenacetin": args.phenacetin, "Benzocaine": args.benzocaine}
    out_dir = Path(args.outdir)
    combined = process_targets(files, out_dir, max_routes_to_plot=args.max_plots)
    if not combined.empty:
        csv_path = out_dir / "route_summaries.csv"
        combined.to_csv(csv_path, index=False)
        from IPython.display import display
        display(combined.head(20))
        print(f"[OK] Wrote {csv_path}")
    else:
        print("[WARN] No summaries produced.")

def notebook_run(
    celecoxib_path: str = DEFAULT_FILES["Celecoxib"],
    ibuprofen_path: str = DEFAULT_FILES["Ibuprofen"],
    benzocaine_path: str = DEFAULT_FILES["Benzocaine"],
    aspirin_path: str = DEFAULT_FILES["Aspirin"],
    favipiravir_path: str = DEFAULT_FILES["Favipiravir"],
    paracetamol_path: str = DEFAULT_FILES["Paracetamol"],
    phenacetin_path: str = DEFAULT_FILES["Phenacetin"],
    erlotinib_path: str = DEFAULT_FILES["Erlotinib"],
    atrovastatin_path: str = DEFAULT_FILES["Atrovastatin"],
    lidocaine_path: str = DEFAULT_FILES["Lidocaine"],

    outdir: str = DEFAULT_OUTDIR,
    max_plots: int = DEFAULT_MAXPLOTS,
):
    files = {"Celecoxib": celecoxib_path, "Ibuprofen": ibuprofen_path, 
             "Benzocaine": benzocaine_path, "Aspirin": aspirin_path, 
             "Favipiravir": favipiravir_path, "Paracetamol": paracetamol_path, 
             "Phenacetin": phenacetin_path, "Erlotinib": erlotinib_path,
             "Atrovastatin": atrovastatin_path, "Lidocaine": lidocaine_path}
    out_dir = Path(outdir)
    combined = process_targets(files, out_dir, max_routes_to_plot=max_plots)
    if not combined.empty:
        csv_path = out_dir / "route_summaries.csv"
        combined.to_csv(csv_path, index=False)
        from IPython.display import display
        display(combined)
        print(f"[OK] Wrote {csv_path}")
    else:
        print("[WARN] No summaries produced.")
    return combined

if __name__ == "__main__":
    cli_entry()
