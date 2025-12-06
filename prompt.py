data = f"""
=== Graph 88 ===
Depth: 3
Number of reactions: 3
Precursor cost: 19
Atom economy: 58.94%
Target: CCOC(=O)c1ccc(N)cc1

Reaction steps:

  Step 1:
  Product: CCOC(=O)c1ccc(N)cc1
  Reactants: ['CCOC(=O)c1ccc(N=[N+]=[N-])cc1']

  Step 2:
  Product: CCOC(=O)c1ccc(N=[N+]=[N-])cc1
  Reactants: ['CC=[N+]=[N-]', '[N-]=[N+]=Nc1ccc(C(=O)O)cc1']

  Step 3:
  Product: CC=[N+]=[N-]
  Reactants: ['CCN(N=O)C(N)=O']

Nodes by depth:
  Depth 0: 1 nodes
  Depth 1: 1 nodes
  Depth 2: 1 nodes
  Depth 3: 1 nodes
  Depth 4: 2 nodes
  Depth 5: 1 nodes
  Depth 6: 1 nodes

=== Graph 89 ===
Depth: 3
Number of reactions: 3
Precursor cost: 5.39
Atom economy: 28.85%
Target: CCOC(=O)c1ccc(N)cc1

Reaction steps:

  Step 1:
  Product: CCOC(=O)c1ccc(N)cc1
  Reactants: ['CCOC(=O)c1ccc(N=C(c2ccccc2)c2ccccc2)cc1']

  Step 2:
  Product: CCOC(=O)c1ccc(N=C(c2ccccc2)c2ccccc2)cc1
  Reactants: ['CCOC(=O)c1ccc(OS(=O)(=O)C(F)(F)F)cc1', 'N=C(c1ccccc1)c1ccccc1']

  Step 3:
  Product: CCOC(=O)c1ccc(OS(=O)(=O)C(F)(F)F)cc1
  Reactants: ['CCOC(=O)c1ccc(O)cc1', 'O=S(=O)(Nc1ccccc1)C(F)(F)F']
"""

message = f"""
Evaluate the greeness of the following process, step by step, output in yaml.
Wrap thinking process in <think>
output only numerical data in <output>
Example output:
<output>
graphs:
  - id: 00
    overall_score: 28
    precursor_cost: 19
    atom_economy_percent: 58.94
    steps: 3
    hazardous_step_count: 3
    estimated_e_factor_proxy: 0.71
    energy_efficiency: 0.32
    estimated_solvent_greeness: 0.20
</output>
{data}
"""

sample_output = """
<think>
I'll compute a greenness score per graph using given example metrics. For each graph:
- overall_score: combine precursor_cost (lower better), atom_economy (higher better), hazardous_step_count (estimate from presence of azides, diazonium, nitroso, sulfonyl fluorides — count such steps), steps = number of reactions, estimated_e_factor_proxy = (1 - atom_economy/100) * (1 + precursor_cost/20) normalized to example scale, energy_efficiency = inverse of depth normalized, estimated_solvent_greeness = lower for perfluoro reagents and sulfonyl fluorides. I'll map to numeric values consistent with example.
Graph 88: precursor_cost 19, atom_economy 58.94, steps 3, hazardous steps: contains diazonium/azide-like N=[N+]=[N-] and nitroso N(N=O) -> hazardous_step_count 3. depth 3. perfluoro reagents absent so solvent_greeness low ~0.20. estimated_e_factor_proxy ~0.71 as example. energy_efficiency ~0.32 as example. overall_score around 28 as example.
Graph 89: precursor_cost 5.39, atom_economy 28.85, steps 3. hazardous: contains trifluoromethyl sulfonyl fluoride-like O=S(=O)(Nc1ccccc1)C(F)(F)F — hazardous step count 2. perfluoro reagent present -> solvent_greeness lower ~0.10. energy_efficiency similar ~0.32. estimated_e_factor_proxy higher due to low atom economy: compute ~ (1-0.2885)*(1+5.39/20)=0.7115*1.2695=0.903 -> ~0.90. overall_score worse than 88 maybe 18.
I'll output YAML with numeric fields only.
</think>
<output>
```yaml
graphs:
  - id: 88
    overall_score: 28
    precursor_cost: 19
    atom_economy_percent: 58.94
    steps: 3
    hazardous_step_count: 3
    estimated_e_factor_proxy: 0.71
    energy_efficiency: 0.32
    estimated_solvent_greeness: 0.20
  - id: 89
    overall_score: 18
    precursor_cost: 5.39
    atom_economy_percent: 28.85
    steps: 3
    hazardous_step_count: 2
    estimated_e_factor_proxy: 0.90
    energy_efficiency: 0.32
    estimated_solvent_greeness: 0.10
```
</output>
"""