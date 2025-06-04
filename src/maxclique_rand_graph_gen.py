import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt

# Parameters
p_counts = range(1, 10, 2)  # Edge probability range
node_counts = 4000
path = "../data/maxclique/random/"
os.makedirs(path, exist_ok=True)

for p_count in p_counts:
    p = p_count / 10
    n = node_counts
    G = nx.erdos_renyi_graph(n, p, seed=None)  # Generate random graph

    # Save as edge list
    L = G.number_of_edges()
    edge_list = np.zeros([L + 1, 2], dtype=np.int64)
    edge_list[0, :] = [n, L]  # First row: node and edge counts
    edge_list[1:, :] = list(G.edges)  # Edge list
    edge_list[1:, :] += 1  # Adjust to 1-based indices

    name = f'n{n}_random_p{int(p * 100)}.txt'
    np.savetxt(path + name, edge_list, fmt='%d')

    print(f"Generated graph with {n} nodes and {L} edges, saved to {path + name}")

# # Visualize graph (uncomment to enable)
# pos = nx.spring_layout(G)
# plt.figure(figsize=(8, 6))
# nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
# nx.draw_networkx_edges(G, pos, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")
# plt.title(f"Random Graph (n={n}, p={p})")
# plt.axis('off')
# plt.show()