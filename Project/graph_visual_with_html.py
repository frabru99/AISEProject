import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random

# Load the GraphML file
G = nx.read_graphml("./graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)


# Add colors and title to nodes
pos = nx.spring_layout(G, seed=42)
for node in net.nodes:
    node["x"], node["y"] = pos[node["id"]][0] * 1000, pos[node["id"]][1] * 1000
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

# Add title to edges
for edge in net.edges:
    edge["color"] = "gray"  # Puoi personalizzarlo
    edge["smooth"] = True  # Rende gli archi più curvi e leggibili
    edge["width"] = G[edge["from"]][edge["to"]].get("weight", 1) * 2  # Spessore basato sul peso
    if "description" in edge:
        edge["title"] = edge["description"]

# Save and display the network
net.show("knowledge_graph.html")
os.system("start knowledge_graph.html")
