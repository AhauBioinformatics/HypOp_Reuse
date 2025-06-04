import numpy as np
import networkx as nx
import random
import os

def ensure_even_sum(degrees):
    degrees = list(degrees)
    if sum(degrees) % 2 != 0:
        degrees[0] += 1
    return degrees

# ==========================
# 1. Graph generation from degree sequences
# ==========================

def generate_graph_from_degree_sequence(degree_seq):
    G = nx.configuration_model(degree_seq)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def generate_normal_degree_graph(n=100, mu=10, sigma=2):
    degrees = np.random.normal(loc=mu, scale=sigma, size=n).astype(int)
    degrees = [max(1, d) for d in degrees]
    degrees = ensure_even_sum(degrees)
    return generate_graph_from_degree_sequence(degrees)

def generate_exponential_degree_graph(n=100, scale=2.0):
    degrees = np.random.exponential(scale=scale, size=n).astype(int)
    degrees = [max(1, d) for d in degrees]
    degrees = ensure_even_sum(degrees)
    return generate_graph_from_degree_sequence(degrees)

def generate_uniform_degree_graph(n=100, low=5, high=15):
    degrees = np.random.randint(low=low, high=high + 1, size=n)
    degrees = ensure_even_sum(degrees)
    return generate_graph_from_degree_sequence(degrees)

# ==========================
# 2. Classic graph models
# ==========================

def generate_er_graph(n=100, p=0.05):
    return nx.erdos_renyi_graph(n=n, p=p)

def generate_ba_graph(n=100, m=3):
    return nx.barabasi_albert_graph(n=n, m=m)

def generate_ws_graph(n=100, k=6, p=0.1):
    return nx.watts_strogatz_graph(n=n, k=k, p=p)

# ==========================
# 3. Hypergraph generation
# ==========================

def generate_hypergraph(n_nodes=1000, n_edges=3000, edge_size_dist='uniform', **kwargs):
    hyperedges = []
    for _ in range(n_edges):
        if edge_size_dist == 'uniform':
            size = np.random.randint(kwargs.get('low', 2), kwargs.get('high', 6))
        elif edge_size_dist == 'normal':
            size = max(2, int(np.random.normal(kwargs.get('mu', 4), kwargs.get('sigma', 1))))
        elif edge_size_dist == 'exponential':
            size = max(2, int(np.random.exponential(kwargs.get('scale', 2))))
        else:
            raise ValueError('Unsupported distribution')
        edge = sorted(random.sample(range(1, n_nodes + 1), size))
        hyperedges.append(edge)
    return n_nodes, hyperedges

def generate_hypergraph_with_overlap(
    n_nodes=1000,
    n_edges=3000,
    edge_size_dist='uniform',
    overlap_t=2,    # minimum number of overlapping nodes required
    overlap_r=1,    # required number of existing hyperedges meeting the overlap_t condition
    **kwargs
):
    """
    Generate a hypergraph with overlap constraints:
      - Each edge size follows edge_size_dist and kwargs
      - A new edge is added only if it overlaps with at least overlap_r existing edges
        by at least overlap_t nodes
    """
    hyperedges = []

    # Helper: sample a single hyperedge size based on distribution
    def sample_size():
        if edge_size_dist == 'uniform':
            return np.random.randint(kwargs.get('low', 2), kwargs.get('high', 6))
        elif edge_size_dist == 'normal':
            return max(2, int(np.random.normal(kwargs.get('mu', 4), kwargs.get('sigma', 1))))
        elif edge_size_dist == 'exponential':
            return max(2, int(np.random.exponential(kwargs.get('scale', 2))))
        else:
            raise ValueError('Unsupported distribution')

    # Main loop: keep sampling candidate hyperedges until n_edges collected
    while len(hyperedges) < n_edges:
        size = sample_size()
        candidate = tuple(sorted(random.sample(range(1, n_nodes + 1), size)))

        # Count how many existing hyperedges overlap with candidate by at least overlap_t nodes
        overlap_count = sum(
            1
            for existing in hyperedges
            if len(set(existing) & set(candidate)) >= overlap_t
        )

        # Accept the candidate only if overlap_count >= overlap_r
        if overlap_count >= overlap_r:
            hyperedges.append(candidate)

    return n_nodes, hyperedges


# ==========================
# 4. Data saving functions
# ==========================

def save_graph_to_txt(G, filepath):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    mapping = {node: idx + 1 for idx, node in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    with open(filepath, 'w') as f:
        f.write(f'{num_nodes} {num_edges}\n')
        for u, v in G.edges():
            f.write(f'{u} {v} 1\n')


def save_hypergraph_to_txt(n_nodes, hyperedges, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f'{n_nodes} {len(hyperedges)}\n')
        for edge in hyperedges:
            f.write(' '.join(map(str, edge)) + '\n')


# ==========================
# 5. Main program
# ==========================

def main():
    n = 3000
    e = 30000
    graph_output_dir = 'data/synthetic_frustrat/graph/'
    hypergraph_output_dir = 'data/synthetic_frustrat/hypergraph/'
    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(hypergraph_output_dir, exist_ok=True)

    # Normal degree distribution graph
    G_normal = generate_normal_degree_graph(n)
    save_graph_to_txt(G_normal, os.path.join(graph_output_dir, 'normal_graph.txt'))
    n_nodes, hyperedges_normal = generate_hypergraph(n, e, edge_size_dist='normal', mu=4, sigma=1)
    save_hypergraph_to_txt(n_nodes, hyperedges_normal, os.path.join(hypergraph_output_dir, 'hyper_normal.txt'))

    # Exponential degree distribution graph
    G_exp = generate_exponential_degree_graph(n)
    save_graph_to_txt(G_exp, os.path.join(graph_output_dir, 'exp_graph.txt'))
    n_nodes, hyperedges_exp = generate_hypergraph(n, e, edge_size_dist='exponential', scale=2.0)
    save_hypergraph_to_txt(n_nodes, hyperedges_exp, os.path.join(hypergraph_output_dir, 'hyper_exponential.txt'))

    # Uniform degree distribution graph
    G_uniform = generate_uniform_degree_graph(n)
    save_graph_to_txt(G_uniform, os.path.join(graph_output_dir, 'uniform_graph.txt'))
    n_nodes, hyperedges_uniform = generate_hypergraph(n, e, edge_size_dist='uniform', low=2, high=6)
    save_hypergraph_to_txt(n_nodes, hyperedges_uniform, os.path.join(hypergraph_output_dir, 'hyper_uniform.txt'))

    # Erdős-Rényi graph
    G_er = generate_er_graph(n, p=0.05)
    save_graph_to_txt(G_er, os.path.join(graph_output_dir, 'er_graph.txt'))
    n_nodes, hyperedges_er = generate_hypergraph(n, e, edge_size_dist='uniform', low=2, high=6)
    save_hypergraph_to_txt(n_nodes, hyperedges_er, os.path.join(hypergraph_output_dir, 'hyper_er.txt'))

    # Barabási–Albert graph
    G_ba = generate_ba_graph(n, m=3)
    save_graph_to_txt(G_ba, os.path.join(graph_output_dir, 'ba_graph.txt'))
    n_nodes, hyperedges_ba = generate_hypergraph(n, e, edge_size_dist='exponential', scale=2.0)
    save_hypergraph_to_txt(n_nodes, hyperedges_ba, os.path.join(hypergraph_output_dir, 'hyper_ba.txt'))

    # Watts–Strogatz graph
    G_ws = generate_ws_graph(n, k=6, p=0.1)
    save_graph_to_txt(G_ws, os.path.join(graph_output_dir, 'ws_graph.txt'))
    n_nodes, hyperedges_ws = generate_hypergraph(n, e, edge_size_dist='normal', mu=4, sigma=1)
    save_hypergraph_to_txt(n_nodes, hyperedges_ws, os.path.join(hypergraph_output_dir, 'hyper_ws.txt'))

if __name__ == "__main__":
    main()
