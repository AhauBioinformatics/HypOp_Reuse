import json
import torch
from src.data_reading import read_stanford
import os
from src.utils import get_normalized_G_from_con,all_to_weights
import torch.nn as nn
from src.model import single_node_xavier
from itertools import chain
import copy
import random
import logging
import time
import numpy as np

def loss_maxclique_weighted(probs, C, dct, weights, p=4):
    """
    Loss function for the maximum clique:
    - probs: Probability of each node belonging to the clique
    - C: Set of edges, indicating which pairs of nodes are connected by edges
    - dct: Mapping dictionary that maps the nodes on the edges to node numbers
    - weights: Weight of each edge, used for weighting
    - p: Parameter used to adjust the weight of the loss term (can be adjusted)

    A smaller loss value indicates a larger clique and more edges between nodes.
    """
    x = probs.squeeze()
    loss = -sum(x)

    for c, w in zip(C, weights):
        temp = (p * w * x[dct[c[0]]] * x[dct[c[1]]])
        loss += temp

    return loss

def loss_maxclique_numpy(x, C, G, weights, penalty=100, hyper=False):
    x_values = np.array([x[i] for i in range(1, len(x) + 1)])
    clique_size = np.sum(x_values)

    missing_edges = 0
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if x[i+1] == 1 and x[j+1] == 1 and G[i, j] == 0:
                missing_edges += 1

    loss = penalty * missing_edges-clique_size

    return loss
def generate_neighbor(clique, vertices, G):
    new_clique = clique.copy()

    if random.random() < 0.5 and len(new_clique) < len(G):
        new_vertex = random.choice([v for v in vertices if v not in clique])
        new_clique.append(new_vertex)
    else:
        if len(clique) > 1:
            vertex_to_remove = random.choice(new_clique)
            new_clique.remove(vertex_to_remove)

    return new_clique

def mapping_distribution(best_outs, n,params, weights, constraints, G, vertices):
    _loss = loss_maxclique_numpy
    best_res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
    best_score = _loss(best_res, constraints, G, weights)
    prev_score = best_score
    t = params['t']
    min_temp = 0.01
    for rea in range(params['N_realize']):
        res = copy.deepcopy(best_res)
        for it in range(params['Niter_h']):
            print(f"Iteration {it}")

            clique = [vertex for vertex in res if res[vertex] == 1]

            new_clique = generate_neighbor(clique, vertices, G)

            new_res = {vertex: (1 if vertex in new_clique else 0) for vertex in res}

            lt = _loss(new_res, constraints, G, weights)
            l1 = _loss(res, constraints, G, weights)

            if lt < l1 or np.exp(-(lt - l1) / t) > np.random.uniform(0, 1):
                res = copy.deepcopy(new_res)

            t = t * 0.95
            if t < min_temp:
                break
            if (it + 1) % 100 == 0:
                score = _loss(res, constraints, G, weights)
                if score == prev_score:
                    print("Early stopping of SA")
                    break
                else:
                    prev_score = score
                    print(f"Score: {score}")

        score = _loss(res, constraints, G, weights)
        print(f"Final Score: {score}")
        if score < best_score:
            best_res = copy.deepcopy(res)
            best_score = score

    clique_size = sum(best_res.values())
    return best_res, clique_size


def tensor_to_networkx_graph(G_tensor):
    """
    Convert the Tensor adjacency matrix to a networkx graph.
    """
    G_np = G_tensor.numpy()

    G_nx = nx.from_numpy_array(G_np)
    return G_nx


import networkx as nx
import matplotlib.pyplot as plt

def visualize_clique(G, best_out, n):
    """
    Visualize the Maximum Clique: Highlight the nodes in the clique.

    Parameters:
    G: The original graph (networkx.Graph)
    best_out: A dictionary recording whether each node belongs to the maximum clique (key: node, value: 1 or 0)
    n: The number of nodes in the graph
    """
    pos = nx.spring_layout(G)

    graph_nodes = set(G.nodes())

    clique_nodes = [node - 1 for node, in_clique in best_out.items() if in_clique == 1 and (node - 1) in graph_nodes]
    non_clique_nodes = [node - 1 for node, in_clique in best_out.items() if in_clique == 0 and (node - 1) in graph_nodes]

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, node_color='red', node_size=500, alpha=0.8,
                           label="Clique Nodes")

    nx.draw_networkx_nodes(G, pos, nodelist=non_clique_nodes, node_color='lightgray', node_size=500, alpha=0.5,
                           label="Non-Clique Nodes")

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.6, edge_color='black')

    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")

    plt.title(f"Visualization of Clique (Clique size: {sum(best_out.values())})")
    plt.axis('off')
    plt.legend()
    plt.show()
def C2G(C,n):
    adj_matrix = np.zeros((n, n), dtype=int)
    for edge in C:
        i, j = edge[0] - 1, edge[1] - 1
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix


with open('configs/maxclique.json') as fi:
   params = json.load(fi)
folder_path = params['folder_path']
TORCH_DEVICE = torch.device('cpu')
TORCH_DTYPE = torch.float32
logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
log = logging.getLogger('main')

for k in range(params['K']):
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            path = folder_path + file_name
            constraints, header = read_stanford(path)
            n = header['num_nodes']
            print('num_nodes',n)
            f = int(np.sqrt(n))
            rounds = int(params['epoch'])
            prev_loss = 100
            patience = params['patience']
            count = 0
            best_loss = float('inf')

            info = {x + 1: [] for x in range(n)}
            for constraint in constraints:
                for node in constraint:
                    info[abs(node)].append(constraint)
            G = get_normalized_G_from_con(constraints, header)

            # model layers
            embed = nn.Embedding(n, f)
            embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
            conv1 = single_node_xavier(f, f // 2)
            conv2 = single_node_xavier(f // 2, 1)
            parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())

            optimizer = torch.optim.Adam(parameters, lr=params['lr'])

            # train steps
            start_time = time.time()
            for i in range(rounds):
                # 向前传播
                inputs = embed.weight
                temp = conv1(inputs)
                temp = G @ temp
                temp = torch.relu(temp)
                temp = conv2(temp)
                temp = G @ temp
                temp = torch.sigmoid(temp)

                loss = loss_maxclique_weighted(temp, constraints, {x + 1: x for x in range(n)},
                                               [1 for i in range(len(constraints))])
                print(f'num_nodes:{n}   Epoch:{i}  Loss:{loss.item():.4f}\n')

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if (abs(loss - prev_loss) <= params['tol']) | ((loss - prev_loss) > 0):
                    count += 1
                    if count >= params['patience']:
                        print(f'Stopping early on epoch {i} (patience: {patience})')
                        break
                else:
                    count = 0

                #### keep the best loss and result ####
                if loss < best_loss:
                    p = 0
                    best_loss = loss
                    best_out = temp
                    print(f'found better loss')
                else:
                    p += 1
                    if p > params['patience']:
                        print('Early Stopping')
                        break

            best_out = best_out.detach().numpy()
            best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}
            all_weights = [1.0 for c in constraints]
            weights = all_to_weights(all_weights, n, constraints)
            adj_matrix = C2G(constraints,n)
            res, clique_size = mapping_distribution(best_out, n,params, weights, constraints, adj_matrix,list(range(1,n+1)))

            end_time = time.time()
            log.info(f'{file_name}:, clique_size: {clique_size}, running_time: {end_time - start_time}')
            print(clique_size)

        # G_np = tensor_to_networkx_graph(G)
        # visualize_clique(G_np, res, n)