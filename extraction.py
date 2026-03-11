from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_mol_weight(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return Descriptors.MolWt(mol)

def calc_pmi(product_smiles, reactant_smiles_list):
    prod_mass = smiles_to_mol_weight(product_smiles)
    react_mass_sum = sum(smiles_to_mol_weight(s) for s in reactant_smiles_list)
    if prod_mass == 0:
        return None
    return react_mass_sum / prod_mass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    atom_economy: float          # stored as fraction (0-1), capped at 1.0
    atom_economy_anomaly: bool   # True if the raw JSON value exceeded 1.0
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
        precursor_ids = [e.to_id for e in self.edges if e.from_id == node_id]
        return [self.get_node_by_id(pid) for pid in precursor_ids if self.get_node_by_id(pid)]

    def get_product(self, node_id: str) -> Optional[Node]:
        for edge in self.edges:
            if edge.to_id == node_id:
                return self.get_node_by_id(edge.from_id)
        return None

    def traverse_by_depth(self) -> Dict[int, List[Node]]:
        depth_map: Dict[int, List[Node]] = {}
        visited: set = set()

        def dfs(node_id: str, depth: int):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.get_node_by_id(node_id)
            if node:
                depth_map.setdefault(depth, []).append(node)
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
                'smiles': reaction_node.smiles,
            })
        return steps


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class RetrosynthesisParser:
    @staticmethod
    def parse(data: List[Dict[str, Any]]) -> List[RetrosynthesisGraph]:
        return [RetrosynthesisParser.parse_single(g) for g in data]

    @staticmethod
    def parse_single(graph_data: Dict[str, Any]) -> RetrosynthesisGraph:
        graph_info = graph_data['graph']

        # Validate atom economy
        raw_ae = graph_info['atom_economy']
        ae_anomaly = raw_ae > 1.0
        if ae_anomaly:
            print(
                f"  [WARN] atom_economy={raw_ae:.4f} > 1.0 — "
                f"upstream data anomaly (missing oxidant/reagent in retrosynthesis). "
                f"Capped at 1.0."
            )
            raw_ae = 1.0

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
            atom_economy=raw_ae,
            atom_economy_anomaly=ae_anomaly,
            score=graph_info.get('score'),
            cluster_id=graph_info.get('cluster_id'),
        )

        nodes = [Node(id=n['id'], smiles=n['smiles'], type=n['type']) for n in graph_data['nodes']]
        edges = [Edge(from_id=e['from'], to_id=e['to'], id=e['id']) for e in graph_data['edges']]

        return RetrosynthesisGraph(
            directed=graph_data['directed'],
            multigraph=graph_data['multigraph'],
            metadata=metadata,
            nodes=nodes,
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_graphs(
    graphs: List[RetrosynthesisGraph],
    remove_ae_anomalies: bool = True,
) -> List[RetrosynthesisGraph]:
    """
    Filter out chemically invalid routes.

    Parameters
    ----------
    graphs : list of RetrosynthesisGraph
    remove_ae_anomalies : bool
        If True (default), drop graphs whose raw atom_economy exceeded 1.0.
        These routes are missing reagents (e.g. oxidants) and are unreliable.

    Returns
    -------
    Filtered list. Prints a summary of how many were removed.
    """
    original_count = len(graphs)

    if remove_ae_anomalies:
        graphs = [g for g in graphs if not g.metadata.atom_economy_anomaly]
        removed = original_count - len(graphs)
        if removed:
            print(f"  [FILTER] Removed {removed} graph(s) with AE > 100% "
                  f"(missing reagents in retrosynthesis data).")

    return graphs


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def save_report(graphs: List[RetrosynthesisGraph], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for i, graph in enumerate(graphs):
            ae_pct = graph.metadata.atom_economy * 100
            f.write(f"\n=== Graph {i + 1} ===\n")
            f.write(f"Depth: {graph.metadata.depth}\n")
            f.write(f"Number of reactions: {graph.metadata.num_reactions}\n")
            f.write(f"Precursor cost: {graph.metadata.precursor_cost}\n")
            f.write(f"Atom economy: {ae_pct:.2f}%\n")

            target = graph.get_target_molecule()
            if target:
                f.write(f"Target: {target.smiles}\n")

            f.write("\nReaction steps:\n")
            for j, step in enumerate(graph.get_reaction_steps(), 1):
                product = step['product']
                reactants = step['reactants']
                product_smiles = product.smiles if product else None
                reactant_smiles_list = [r.smiles for r in reactants]

                pmi_value = None
                if product_smiles and reactant_smiles_list:
                    pmi_value = calc_pmi(product_smiles, reactant_smiles_list)

                f.write(f"\n  Step {j}:\n")
                f.write(f"  Product: {product_smiles or 'N/A'}\n")
                f.write(f"  Reactants: {reactant_smiles_list}\n")
                f.write(f"  PMI: {pmi_value:.2f}\n" if pmi_value is not None else "  PMI: N/A\n")

            f.write("\nNodes by depth:\n")
            for depth, nodes in sorted(graph.traverse_by_depth().items()):
                f.write(f"  Depth {depth}: {len(nodes)} nodes\n")

            f.write("-" * 30 + "\n")

    print(f"  Saved -> {filename}")


# ---------------------------------------------------------------------------
# Single-file entry point (kept for backward compatibility)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "data/Phenacetin.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dataoutput/Phenacetin.txt"

    print(f"Processing {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = RetrosynthesisParser()
    graphs = parser.parse(data)
    graphs = sorted(graphs, key=lambda g: g.metadata.avg_plausibility, reverse=True)
    graphs = filter_graphs(graphs, remove_ae_anomalies=True)

    save_report(graphs, output_file)
    print("Done.")
