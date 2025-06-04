import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_graph_from_degree_sequence(degree_seq):
    
    G = nx.configuration_model(degree_seq)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def plot_degree_distribution(G, title=''):
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=20, density=True, alpha=0.6, color='skyblue')
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def ensure_even_sum(degrees):
    degrees = list(degrees)
    if sum(degrees) % 2 != 0:
        degrees[0] += 1
    return degrees

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
    degrees = np.random.randint(low=low, high=high+1, size=n)
    degrees = ensure_even_sum(degrees)
    return generate_graph_from_degree_sequence(degrees)

if __name__ == "__main__":
    n = 10

    G_normal = generate_normal_degree_graph(n)
    plot_degree_distribution(G_normal, title="Degree Distribution (Normal)")

    G_exp = generate_exponential_degree_graph(n)
    plot_degree_distribution(G_exp, title="Degree Distribution (Exponential)")

    G_uniform = generate_uniform_degree_graph(n)
    plot_degree_distribution(G_uniform, title="Degree Distribution (Uniform)")
