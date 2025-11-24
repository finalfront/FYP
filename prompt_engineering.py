import openai
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="*************************",
    api_version="2025-02-01-preview",
    azure_endpoint="https://hkust.azure-api.net"
)


with open("output.txt", "r", encoding="utf-8") as f:
    retrosynthesis_text = f.read()

system_prompt = """
You are a green chemistry expert.

You receive textual descriptions of many retrosynthesis routes ("graphs") to the same target molecule. 
Each route is labeled like "=== Graph 1 ===", "=== Graph 2 ===", etc., and contains a line 
starting with "Atom economy: XX.XX%" that gives the atom economy of that route.

Your task:
1. Parse all graphs and extract:
   - graph_id (integer after "Graph ")
   - atom_economy_percent (float value from the line "Atom economy: XX.XX%")

2. Evaluate "greenness" ONLY based on atom economy.
   - Higher atom economy means greener route.
   - Lower number of reactions means greener route.
   - Ignore all other metrics (precursor cost, number of reactions, depth, etc.) for this task.

3. Rank all routes from greenest to least green based on atom_economy_percent (descending).

4. Output ONLY a JSON array (no extra text), where each element is:
   {
     "graph_id": <int>,
     "score of number of reaction": <int>, 
     "score of atom_economy_percent": <float>, // keep the score to 1-10
     "rank": <int>   // 1 = greenest (highest atom economy)
   }

Additional rules:
- If two routes have exactly the same atom economy, assign them the same rank and keep any order.
- Do NOT include any explanation or extra text outside the JSON.
- If there is any parsing ambiguity, do your best guess and still output strictly valid JSON.
"""


few_shot_user_1 = """=== Graph 1 ===
Depth: 1
Number of reactions: 1
Precursor cost: 14.2
Atom economy: 77.48%
Target: CCOC(=O)c1ccc(N)cc1

=== Graph 2 ===
Depth: 1
Number of reactions: 1
Precursor cost: 1
Atom economy: 85.06%
Target: CCOC(=O)c1ccc(N)cc1

=== Graph 3 ===
Depth: 1
Number of reactions: 1
Precursor cost: 2
Atom economy: 56.54%
Target: CCOC(=O)c1ccc(N)cc1
"""

few_shot_assistant_1 = """[
  {
     "graph_id": <int>,
     "score of number of reaction": <int>
     "score of atom_economy_percent": <float>,
     "rank": <int>   // 1 = greenest (highest atom economy)

    "graph_id": <int>,
     "score of number of reaction": <int>
     "score of atom_economy_percent": <float>,
     "rank": <int>   // 1 = greenest (highest atom economy)

    "graph_id": <int>,
     "score of number of reaction": <int>
     "score of atom_economy_percent": <float>,
     "rank": <int>   // 1 = greenest (highest atom economy)
  }
]"""


few_shot_user_2 = """=== Graph 10 ===
Depth: 1
Number of reactions: 1
Precursor cost: 0.2
Atom economy: 98.84%
Target: CCOC(=O)c1ccc(N)cc1

=== Graph 11 ===
Depth: 1
Number of reactions: 1
Precursor cost: 14.23
Atom economy: 98.84%
Target: CCOC(=O)c1ccc(N)cc1

=== Graph 12 ===
Depth: 1
Number of reactions: 1
Precursor cost: 1.2
Atom economy: 82.11%
Target: CCOC(=O)c1ccc(N)cc1
"""

few_shot_assistant_2 = """[
  {
    "graph_id": 10,
    "atom_economy_percent": 98.84,
    "rank": 1
  },
  {
    "graph_id": 11,
    "atom_economy_percent": 98.84,
    "rank": 1
  },
  {
    "graph_id": 12,
    "atom_economy_percent": 82.11,
    "rank": 2
  }
]"""

messages = [
    {"role": "system", "content": system_prompt},

    # few-shot example 1
    #{"role": "user", "content": few_shot_user_1},
    #{"role": "assistant", "content": few_shot_assistant_1},

    # few-shot example 2
    #{"role": "user", "content": few_shot_user_2},
    #{"role": "assistant", "content": few_shot_assistant_2},

    # 真实任务：把整个 output.txt 丢进去
    {"role": "user", "content": retrosynthesis_text}
]

response = client.chat.completions.create(
    model="o4-mini",
    messages=messages,
)

print(response.choices[0].message.content)