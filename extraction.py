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
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_reactions(self) -> List[Node]:
        """Get all reaction nodes."""
        return [node for node in self.nodes if node.is_reaction()]
    
    def get_chemicals(self) -> List[Node]:
        """Get all chemical nodes."""
        return [node for node in self.nodes if node.is_chemical()]
    
    def get_target_molecule(self) -> Optional[Node]:
        """Get the target molecule (node with all-zeros UUID)."""
        return self.get_node_by_id("00000000-0000-0000-0000-000000000000")
    
    def get_precursors(self, node_id: str) -> List[Node]:
        """Get all precursor nodes for a given node."""
        precursor_ids = [edge.to_id for edge in self.edges if edge.from_id == node_id]
        return [self.get_node_by_id(pid) for pid in precursor_ids if self.get_node_by_id(pid)]
    
    def get_product(self, node_id: str) -> Optional[Node]:
        """Get the product node for a given node."""
        for edge in self.edges:
            if edge.to_id == node_id:
                return self.get_node_by_id(edge.from_id)
        return None
    
    def traverse_by_depth(self) -> Dict[int, List[Node]]:
        """Organize nodes by their depth in the synthesis tree."""
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
                
                # Traverse to precursors
                for precursor in self.get_precursors(node_id):
                    dfs(precursor.id, depth + 1)
        
        target = self.get_target_molecule()
        if target:
            dfs(target.id, 0)
        
        return depth_map
    
    def get_reaction_steps(self) -> List[Dict[str, Any]]:
        """Extract each reaction step with its reactants and products."""
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
        """Parse a list of retrosynthesis graphs."""
        graphs = []
        for graph_data in data:
            graphs.append(RetrosynthesisParser.parse_single(graph_data))
        return graphs
    
    @staticmethod
    def parse_single(graph_data: Dict[str, Any]) -> RetrosynthesisGraph:
        """Parse a single retrosynthesis graph."""
        # Parse metadata
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
        
        # Parse nodes
        nodes = [
            Node(id=n['id'], smiles=n['smiles'], type=n['type'])
            for n in graph_data['nodes']
        ]
        
        # Parse edges
        edges = [
            Edge(from_id=e['from'], to_id=e['to'], id=e['id'])
            for e in graph_data['edges']
        ]
        
        return RetrosynthesisGraph(
            directed=graph_data['directed'],
            multigraph=graph_data['multigraph'],
            metadata=metadata,
            nodes=nodes,
            edges=edges
        )


# Example usage
if __name__ == "__main__":
    # Your data here
    data = json.load(open("data/Benzocaine.json","r"))  # Your input data
    # Parse the data
    parser = RetrosynthesisParser()
    graphs = parser.parse(data)
    # sort to key of your choice, default low to high, set reverse = True for high to low
    graphs = sorted(graphs,key = lambda graph: graph.metadata.avg_plausibility, reverse= True) 
    # Analyze each graph
    for i, graph in enumerate(graphs):
        print(f"\n=== Graph {i + 1} ===")
        print(f"Depth: {graph.metadata.depth}")
        print(f"Number of reactions: {graph.metadata.num_reactions}")
        print(f"Precursor cost: {graph.metadata.precursor_cost}")
        print(f"Atom economy: {graph.metadata.atom_economy:.2%}")
        
        # Get target molecule
        target = graph.get_target_molecule()
        if target:
            print(f"Target: {target.smiles}")
        
        # Get PMI values
        print(f"\nReaction Steps")
        for j, step in enumerate(graph.get_reaction_steps(), 1):
            product = step['product']
            reactants = step['reactants']

            product_smiles = product.smiles if product else None
            reactant_smiles_list = [r.smiles for r in reactants]

            pmi_value = None
            if product_smiles is not None and reactant_smiles_list:
                pmi_value = calc_pmi(product_smiles, reactant_smiles_list)

        # Get all reaction steps
        print(f"\nReaction steps:")
        for j, step in enumerate(graph.get_reaction_steps(), 1):
            print(f"\n  Step {j}:")
            print(f"  Product: {step['product'].smiles if step['product'] else 'N/A'}")
            print(f"  Reactants: {[r.smiles for r in step['reactants']]}")
            print(f"  PMI: {pmi_value:.2f}" if pmi_value is not None else "  PMI: N/A")
        
        # Organize by depth
        print(f"\nNodes by depth:")
        depth_map = graph.traverse_by_depth()
        for depth in sorted(depth_map.keys()):
            print(f"  Depth {depth}: {len(depth_map[depth])} nodes")
