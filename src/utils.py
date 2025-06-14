import numpy as np
import torch
import timeit
from src.loss import loss_sat_numpy, loss_maxcut_numpy, loss_maxcut_numpy, loss_maxind_numpy, loss_maxind_QUBO, loss_task_numpy, loss_task_numpy_vec, loss_mincut_numpy, loss_partition_numpy
import random
import networkx as nx
import copy

from collections import OrderedDict, defaultdict

# Shared Functions
def generate_H_from_constraints(constraints, main, self_loop=False):
    """
    Generate the hypgraph incidence matrix H from hyper constriants list
    :param edges: Hyper edges. List of nodes that in that hyper edges.
    :n: number of nodes
    :self_loop: Whether need to add self_loops. 
    """
    H = []
    i = 1
    dct = {}
    new_constraints = []
    dct[main] = 0
    for c in constraints:
        temp = []
        for node in c:
            if abs(node) not in dct:
                dct[abs(node)] = i
                i += 1
            temp.append(dct[abs(node)])
        new_constraints.append(temp)
    n = len(dct)
    for c in new_constraints:
        temp = [0 for j in range(n)]
        for node in c:
            temp[node] = 1
        H.append(temp)
    if self_loop:
        # added self loop hyper edges
        for i in range(n):
            temp = [0 for j in range(n)]
            temp[i] = 1
            H.append(temp)
    return np.array(H, dtype=float).T, dct

def generate_H_from_edges(edges, n, self_loop=False):
    """
    Generate the hypgraph incidence matrix H from hyper edges list
    :param edges: Hyper edges. List of nodes that in that hyper edges.
    :n: number of nodes
    :self_loop: Whether need to add self_loops. 
    """
    H = []

    for edge in edges:
        temp = [0 for j in range(n)]
        for node in edge:
            temp[node] = 1
        H.append(temp)
    if self_loop:
        # added self loop hyper edges
        for i in range(n):
            temp = [0 for j in range(n)]
            temp[i] = 1
            H.append(temp)
    Ht=np.array(H, dtype=float).T
    return Ht

def _generate_G_from_H(H, variable_weight=False):
    """
    This function is implemented by Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong, Ji, Yue Gao from Xiamen University and Tsinghua University
    Originally github repo could be found here https://github.com/iMoonLab/HGNN
    Originally paper could be found here https://arxiv.org/abs/1809.09401
    
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    #H = np.array(H)
    n_edge = H.shape[1]
    n_node = H.shape[0]
    #Adjacency matrix of the graph with self loop
    A = get_adj(H, n_node, n_edge)
    DA=np.sum(A, axis=1)
    invDA = np.asmatrix(np.diag(np.power(DA, -0.5)))

    Ga=invDA @ A @ invDA

    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    DEm = DE-1
    inDEm = np.asmatrix(np.diag(np.power(DEm, -1)))
    inDEm = np.nan_to_num(inDEm, 0)
    invDE = np.asmatrix(np.diag(np.power(DE, -1)))


    DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
    W = np.asmatrix(np.diag(W))
    H = np.asmatrix(H)
    HT = H.T


    #G = H * W * invDE * HT
    Hp = H @ inDEm @ HT
    Hp = Hp - np.diag(np.diagonal(Hp))

    G = DV2 @ H @ W @ invDE @ HT @ DV2
    #G = DV2 @ H @ HT @ DV2
    Gp = DV2 @ Hp @ DV2

    return Gp

def get_normalized_G_from_con(constraints, header):
    n_nodes = header['num_nodes']
    G_matrix =  torch.zeros(n_nodes, n_nodes)
    indegree = [0] * n_nodes
    outdegree = [0] * n_nodes
    for (u, v) in constraints:
        indegree[u-1] += 1
        outdegree[u-1] += 1
        indegree[v-1] += 1
        outdegree[v-1] += 1
    for (u, v) in constraints:
        u_ = u - 1
        v_ = v - 1
        G_matrix[u_][v_] += 1 * (indegree[u_] ** (-0.5)) * (outdegree[u_] ** (-0.5))
        G_matrix[v_][u_] += 1 * (indegree[v_] ** (-0.5)) * (outdegree[v_] ** (-0.5))
    #for u in range(0, n_nodes):
    #    G_matrix[u][u] += 1
    return G_matrix

def get_adj(H, n_nodes, n_edges):
    A=np.zeros([n_nodes,n_nodes])
    for i in range(n_nodes):
        edges=np.argwhere(H[i, :] > 0).T[0]
        for e in edges:
            nodes=np.argwhere(H[:, e] > 0).T[0]
            for j in nodes:
                A[i,j]=1
    return A

def samples_to_Xs(sample, n, f):
    Xs = {}
    for i in range(n):
        a , b = i*f, (i+1)*f
        Xs[i+1] = torch.from_numpy(sample[a:b][None]).float()
    return Xs

def find_pres(out, tar):
    cur = out[tar]
    pres = []
    for node in out.keys():
        if node != tar:
            if out[node] > cur:
                pres.append(node)
    return pres

def all_to_weights(weights_all, n, C):
    weights = {x+1: [] for x in range(n)}
    for c, w in zip(C, weights_all):
        for node in c:
            weights[abs(node)].append(w)
    return weights

def all_to_weights_task(weights_all, n, C):
    weights = {x+1: [] for x in range(n)}
    for c, w in zip(C, weights_all):
        for node in c[:-1]:
            weights[abs(node)].append(w)
    return weights

# mapping functions
def mapping_algo(best_outs, weights, info, mode):
    finished = set()
    pres = {x: find_pres(best_outs, x) for x in best_outs.keys()}
    res = {x: 0 for x in best_outs.keys()}
    n = len(best_outs.keys())
    if mode == 'sat':
        _loss = loss_sat_numpy
    elif mode == 'maxcut':
        _loss = loss_maxcut_numpy
    while len(finished) < n:
        this_round = []
        for i in pres:
            if all([x in finished for x in pres[i]]):
                temp = res.copy()
                temp[i] = 1
                if _loss(temp, info[i], weights[i]) < _loss(res, info[i], weights[i]):
                    res = temp
                finished.add(i)
                this_round.append(i)
        for ele in this_round:
            del pres[ele]
    return res

def mapping_distribution(best_outs, params, n, info, weights, constraints, all_weights, inc, penalty, hyper):
    fine_tuning = params['fine_tuning']
    if fine_tuning == "SA":
        if params['random_init']=='one_half':
            best_outs= {x: 0.5 for x in best_outs.keys()}
        elif params['random_init']=='uniform':
            best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
        elif params['random_init'] == 'threshold':
            best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

        best_score = float('inf')
        lb = float('inf')
        if params['mode'] == 'sat':
            _loss = loss_sat_numpy
        elif params['mode'] == 'maxcut' or params['mode'] == 'QUBO_maxcut' or params['mode'] == 'maxcut_annea':
            _loss = loss_maxcut_numpy
        elif params['mode'] == 'maxind' or params['mode'] == 'QUBO':
            _loss = loss_maxind_numpy
        elif params['mode'] == 'task':
            _loss = loss_task_numpy
        elif params['mode'] == 'mincut':
            _loss = loss_mincut_numpy

        for rea in range(params['N_realize']):
            res = {x: random.choices(range(2), weights=[1 - best_outs[x], best_outs[x]], k=1)[0] for x in best_outs.keys()}
            best_score = _loss(res, constraints, all_weights, hyper=hyper)
            best_res = copy.deepcopy(res)
            t = params['t']
            # l1=best_score
            prev_score=best_score
            for it in range(params['Niter_h']):
                print(it)
                # temp = copy.deepcopy(res)
                ord = random.sample(range(1, n + 1), n)
                # j=0
                for i in ord:
                    # j+=1
                    temp = copy.deepcopy(res)
                    # temp = pr.copy()
                    if res[i] == 0:
                        temp[i] = 1
                    else:
                        temp[i] = 0
                    # if (j) % stepsize == 0:
                    # lt = _loss(temp, constraints, all_weights, penalty=penalty,  hyper=hyper)
                    # l1 = _loss(res, constraints, all_weights, penalty=penalty, hyper=hyper)
                    lt = _loss(temp, info[i], weights[i], penalty=penalty, hyper=hyper)
                    l1 = _loss(res, info[i], weights[i], penalty=penalty, hyper=hyper)
                    if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                        res = copy.deepcopy(temp)
                        # l1=lt
                        # print(l1)
                        # temp=copy.deepcopy(res)
                t = t * 0.95
                if (it+1)%100==0:
                    # score = _loss(res, constraints, all_weights, hyper=hyper)
                    score=l1
                    if score==prev_score:
                        print('early stopping of SA')
                        break
                    else:
                        prev_score = score
                        print(score)

            score = _loss(res, constraints, all_weights, hyper=hyper)
            print(score)
            if score < best_score:
                best_res =copy.deepcopy(res)
                best_score = score

    elif fine_tuning == "PSO":
        n = len(constraints)
        w = params.get('inertia_weight', 0.7)
        c1 = params.get('c1', 1.5)
        c2 = params.get('c2', 1.5)

        if params['random_init'] == 'one_half':
            best_outs = {x: 0.5 for x in range(n)}
        elif params['random_init'] == 'uniform':
            best_outs = {x: np.random.uniform(0, 1) for x in range(n)}
        elif params['random_init'] == 'threshold':
            best_outs = {x: 0 if np.random.uniform(0, 1) < 0.5 else 1 for x in range(n)}

        particles = [{x: np.random.choice([0, 1]) for x in best_outs.keys()} for _ in range(params['N_particles'])]
        velocities = [{x: np.random.uniform(-1, 1) for x in best_outs.keys()} for _ in
                    range(params['N_particles'])]
        p_best = copy.deepcopy(particles)
        best_res = None
        g_best_score = float('inf')

        if params['mode'] == 'sat':
            _loss = loss_sat_numpy
        elif params['mode'] in ['maxcut', 'QUBO_maxcut', 'maxcut_annea']:
            _loss = loss_maxcut_numpy
        elif params['mode'] in ['maxind', 'QUBO']:
            _loss = loss_maxind_numpy
        elif params['mode'] == 'task':
            _loss = loss_task_numpy
        elif params['mode'] == 'mincut':
            _loss = loss_mincut_numpy

        for i, particle in enumerate(particles):
            score = _loss(particle, constraints, all_weights, hyper=hyper)
            if score < g_best_score:
                best_res = copy.deepcopy(particle)
                g_best_score = score

        for it in range(params['Niter_h']):
            print(f"Iteration {it}")

            for i, particle in enumerate(particles):
                for x in particle.keys():
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    velocities[i][x] = (w * velocities[i][x] +
                                        c1 * r1 * (p_best[i][x] - particle[x]) +
                                        c2 * r2 * (best_res[x] - particle[x]))

                    particle[x] += velocities[i][x]
                    particle[x] = 1 if particle[x] > 0.5 else 0

                score = _loss(particle, constraints, all_weights, hyper=hyper)

                p_best_score = _loss(p_best[i], constraints, all_weights, hyper=hyper)
                if score < p_best_score:
                    p_best[i] = copy.deepcopy(particle)

                if score < g_best_score:
                    best_res = copy.deepcopy(particle)
                    g_best_score = score

            w = w * 0.95

            if (it + 1) % 100 == 0:
                print(f"Global best score at iteration {it + 1}: {g_best_score}")
                if g_best_score < params.get('early_stop_threshold', float('inf')):
                    print("Early stopping condition met.")
                    break
        
    elif fine_tuning == "ACO":
        if params['random_init'] == 'one_half':
            best_outs = {x: 0.5 for x in best_outs.keys()}
        elif params['random_init'] == 'uniform':
            best_outs = {x: np.random.uniform(0, 1) for x in best_outs.keys()}
        elif params['random_init'] == 'threshold':
            best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

        if params['mode'] == 'sat':
            loss_function = loss_sat_numpy
        elif params['mode'] in ['maxcut', 'QUBO_maxcut', 'maxcut_annea']:
            loss_function = loss_maxcut_numpy
        elif params['mode'] in ['maxind', 'QUBO']:
            loss_function = loss_maxind_numpy
        elif params['mode'] == 'task':
            loss_function = loss_task_numpy
        elif params['mode'] == 'mincut':
            loss_function = loss_mincut_numpy

        best_score = float('inf')
        pheromone = {x: 1.0 for x in best_outs.keys()}

        for rea in range(params['N_realize']):
            ants_solutions = []
            for ant in range(params['N_ants']):
                res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
                score = loss_function(res, constraints, all_weights, hyper)
                ants_solutions.append((res, score))

            for it in range(params['Niter_h']):
                for ant in ants_solutions:
                    res, score = ant

                    for i in range(n):

                        transition_probs = [pheromone[x] for x in best_outs.keys()]
                        chosen = np.random.choice(list(best_outs.keys()), p=transition_probs / np.sum(transition_probs))
                        temp = res.copy()
                        temp[chosen] = 1 - temp[chosen]

                        new_score = loss_function(temp, constraints, all_weights, hyper)
                        if new_score < score:
                            ants_solutions.remove(ant)
                            ants_solutions.append((temp, new_score))
                            pheromone[chosen] += 1
                            break

                for key in pheromone.keys():
                    pheromone[key] *= (1 - params['evaporation_rate'])

                print(f"Iteration {it}: Best score: {min(ants_solutions, key=lambda x: x[1])[1]}")

            current_best = min(ants_solutions, key=lambda x: x[1])
            if current_best[1] < best_score:
                best_score = current_best[1]
                best_res = current_best[0]

    elif fine_tuning == "GA":
        if params['random_init'] == 'one_half':
            best_outs = {x: 0.5 for x in best_outs.keys()}
        elif params['random_init'] == 'uniform':
            best_outs = {x: np.random.uniform(0, 1) for x in best_outs.keys()}
        elif params['random_init'] == 'threshold':
            best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

        best_score = float('inf')
        best_res = None

        # Create initial population
        population_size = params['population_size']
        population = [
            {x: np.random.choice([0, 1], p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
            for _ in range(population_size)
        ]

        # Define loss function based on mode
        if params['mode'] == 'sat':
            _loss = loss_sat_numpy
        elif params['mode'] in ['maxcut', 'QUBO_maxcut', 'maxcut_annea']:
            _loss = loss_maxcut_numpy
        elif params['mode'] in ['maxind', 'QUBO']:
            _loss = loss_maxind_numpy
        elif params['mode'] == 'task':
            _loss = loss_task_numpy
        elif params['mode'] == 'mincut':
            _loss = loss_mincut_numpy

        for rea in range(params['N_realize']):
            for generation in range(params['N_generations']):
                # Evaluate fitness of the population
                fitness_scores = [1 / (_loss(ind, constraints, all_weights, hyper=hyper) + 1e-6) for ind in population]

                # Select parents based on fitness
                parents_indices = np.random.choice(range(population_size), size=params['n_parents'],
                                                   p=fitness_scores / np.sum(fitness_scores))
                parents = [population[i] for i in parents_indices]

                # Crossover
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        parent1, parent2 = parents[i], parents[i + 1]
                        child1, child2 = {}, {}
                        for key in best_outs.keys():
                            # Single-point crossover
                            if random.random() < params['crossover_rate']:
                                child1[key] = parent1[key]
                                child2[key] = parent2[key]
                            else:
                                child1[key] = parent2[key]
                                child2[key] = parent1[key]
                        offspring.append(child1)
                        offspring.append(child2)
                # Mutation
                for ind in offspring:
                    for key in best_outs.keys():
                        if random.random() < params['mutation_rate']:
                            ind[key] = 1 - ind[key]  # Flip the bit

                # Replace the old population with the new offspring
                population = population + offspring
                population = sorted(population, key=lambda ind: _loss(ind, constraints, all_weights, hyper=hyper))[
                             :population_size]

                # Update best result
                for ind in population:
                    score = _loss(ind, constraints, all_weights, hyper=hyper)
                    if score < best_score:
                        best_res = copy.deepcopy(ind)
                        best_score = score

                print(f'Generation {generation + 1}/{params["N_generations"]}, Best Score: {best_score}')

    return best_res


def mapping_distribution_QUBO(best_outs, params, q_torch, n):
    #best_outs= {x: 0.5 for x in best_outs.keys()}
    #best_outs = {x: np.maxclique_data.uniform(0,1) for x in best_outs.keys()}
    best_score = float('inf')
    lb = float('inf')
    _loss = loss_maxind_QUBO
    for rea in range(params['N_realize']):
        res = {x: np.random.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
        ord = random.sample(range(1, n + 1), n)
        t = 0
        for it in range(params['Niter_h']):
            print(it)
            for i in ord:
                temp = res.copy()
                # temp = pr.copy()
                if res[i] == 0:
                    temp[i] = 1
                else:
                    temp[i] = 0
                lt = _loss(torch.Tensor(list(temp.values())), q_torch)
                l1 = _loss(torch.Tensor(list(res.values())), q_torch)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    res = temp
            t = t * 0.95
        score = _loss(torch.Tensor(list(res.values())), q_torch)
        if score < best_score:
            best_res =res
            best_score = score
    return best_res


def mapping_distribution_vec_task(best_outs, params, n, info, constraints, C_dic, all_weights, inc, lenc, leninfo, penalty, hyper):
    if params['random_init']=='one_half':
        best_outs= {x: 0.5 for x in best_outs.keys()}
    elif params['random_init']=='uniform':
        best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
    elif params['random_init'] == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}
    L=len(constraints)
    best_score = float('inf')
    lb = float('inf')

    if params['mode'] == 'task_vec':
        _loss = loss_task_numpy_vec

    for rea in range(params['N_realize']):
        res = {x: [np.random.choice(range(2), p=[1 - best_outs[x][i], best_outs[x][i]]) for i in range(L)] for x in best_outs.keys()}
        res_array = np.array(list(res.values()))
        # lbest = _loss(res, lenc, leninfo)
        lbest = _loss(res_array, lenc, leninfo)
        l1=lbest
        resbest = res.copy()
        # ord = maxclique_data.sample(range(1, n + 1), n)
        t = params['t']
        for it in range(params['Niter_h']):
            print(it)
            ord = random.sample(range(1, n + 1), n)
            for i in ord:
                #temp = copy.deepcopy(res)
                temp = copy.deepcopy(res_array)
                # temp = pr.copy()
                j=random.sample(range(L), 1)[0]
                # if res[i][j] == 0:
                #     temp[i][j] = 1
                # else:
                #     temp[i][j] = 0
                if res_array[i-1,j] == 0:
                    temp[i-1,j] = 1
                else:
                    temp[i-1,j] = 0
                lt = _loss(temp, lenc, leninfo)
                #l1 = _loss(res, lenc, leninfo)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    # res = copy.deepcopy(temp)
                    #res=temp.copy()
                    if res_array[i-1,j] == 0:
                        res_array[i-1,j] = 1
                    else:
                        res_array[i-1,j] = 0
                    l1=lt
                    if l1==0:
                        break
                    # if l1<=lbest:
                    #     lbest=l1
                    #     resbest=res.copy()
            if l1 == 0:
                break
            t = t * 0.95
        # score = _loss(res, lenc, leninfo)
        lbest=l1
        score = lbest
        print(score)
        if score <= best_score:
            #best_res =resbest.copy()
            best_res = copy.deepcopy(res_array)
            best_score = score
    return best_res


def mapping_distribution_vec(best_outs, params, n, info, weights, constraints, all_weights, inc, L, penalty,hyper):
    if params['random_init'] == 'one_half':
        best_outs = {x: 0.5 for x in best_outs.keys()}
    elif params['random_init'] == 'uniform':
        best_outs = {x: np.random.uniform(0, 1) for x in best_outs.keys()}
    elif params['random_init'] == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

    best_score = float('inf')
    lb = float('inf')

    if params['mode'] == 'partition':
        _loss = loss_partition_numpy

    for rea in range(params['N_realize']):
        # res={x:best_outs[x].argmax()}
        res={}
        for x in best_outs.keys():
            part=np.random.choice(range(params['n_partitions']), p=best_outs[x])
            res_x=[0 for _ in range(params['n_partitions'])]
            res_x[part]=1
            res[x]=res_x
        res_array = np.array(list(res.values()))
        lbest = _loss(res_array, constraints, weights, hyper)
        l1 = lbest
        resbest = res.copy()
        t = params['t']
        for it in range(params['Niter_h']):
            print(it)
            ord = random.sample(range(1, n + 1), n)
            for i in ord:
                temp = copy.deepcopy(res_array)
                temp[i-1,:]=[0 for _ in range(params['n_partitions'])]
                j = random.sample(range(L), 1)[0]
                temp[i - 1, j] = 1
                lt = _loss(temp, constraints, weights, hyper)
                if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
                    res_array[i-1,:] = [0 for _ in range(params['n_partitions'])]
                    res_array[i - 1, j] = 1
                    l1 = lt
            t = t * 0.95
        lbest = l1
        score = lbest
        print(score)
        if score <= best_score:
            best_res = copy.deepcopy(res_array)
            best_score = score
    return best_res

def Neighbors(n, info):
    Nei= {}
    for i in range(1,n+1):
        ne=[]
        for x in info[i]:
            for j in x:
                ne.append(j)
        ne=set(ne)
        ne.discard(i)
        Nei[i]=ne
    return Nei

import h5py
import pandas as pd
def analysis_res(path):
    with h5py.File(path, 'r') as f:
        names = []
        reses = []
        for key in f.keys():
            names.append(key)
            reses.append(f[key][:][0])
    res  = pd.DataFrame()
    res['File_name'] = names
    res['Result'] = reses        
    return res


def gen_q_mis(constraints, n_nodes, penalty=2 ,torch_dtype=None, torch_device=None):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.

    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_mat = torch.zeros(n_nodes, n_nodes)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for cons in constraints:
        Q_mat[cons[0]-1][cons[1]-1] = penalty
        Q_mat[cons[1] - 1][cons[0] - 1] = penalty
    # all diagonal terms get -1
    for u in range(n_nodes):
        Q_mat[u][u] = -1


    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)


    return Q_mat

def gen_q_maxcut(constraints, n_nodes,torch_dtype=None, torch_device=None):
    """
    Helper function to generate QUBO matrix for Maxcut as minimization problem.

    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_mat = torch.zeros(n_nodes, n_nodes)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for cons in constraints:
        Q_mat[cons[0]-1][cons[1]-1] = 1
        Q_mat[cons[1] - 1][cons[0] - 1] = 1
    # all diagonal terms get -1
    for u in range(n_nodes):
        Q_mat[u][u] = -1


    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)


    return Q_mat


def Maxind_postprocessing(res, constraints,n):
    res_copy=res
    graph_p = nx.Graph()
    graph_p.add_nodes_from(range(1,n+1))
    graph_p.add_edges_from(constraints)
    n=len(res)
    nei={}
    score={}
    for i in range(1,n+1):
        if res[i]==1:
            nei[i]=list(graph_p.neighbors(i))
            score[i]=sum([res[item] for item in nei[i]])
        else:
            score[i] = 0
    score_s=sorted(score.items(), key=lambda x: x[1], reverse=True)
    score_sd = {id: jd for (id, jd) in score_s}
    ss=0
    for cons in constraints:
        ss += res[cons[0]] * res[cons[1]]
    print(ss)

    while sum(score_sd.values())>0:
        nodes=list(score_sd.keys())
        res[nodes[0]]=0
        score[nodes[0]] = score[nodes[0]]-1
        score_s = sorted(score.items(), key=lambda x: x[1], reverse=True)
        score_sd = {id: jd for (id, jd) in score_s}

    return res_copy


def sparsify_graph(constraints, header,  info, spars_p):
    n=header['num_nodes']
    m=header['num_constraints']
    constraints2=copy.deepcopy(constraints)
    info2=copy.deepcopy(info)
    for edge in constraints:
        n1=edge[0]
        n2=edge[1]
        if len(info2[n1])>1 and len(info2[n2])>1:
            rnd=np.random.uniform(0, 1)
            if rnd<spars_p:
                constraints2.remove(edge)
                info2[n1].remove(edge)
                info2[n2].remove(edge)
    header2={}
    header2['num_nodes']=n
    header2['num_constraints']=len(constraints2)
    return constraints2, header2, info2


def generate_watermark(N, wat_len, wat_seed_value):
    # maxclique_data.seed(wat_seed_value)
    p=0.2
    selected_nodes=random.sample(range(1,N),wat_len)
    Gr = nx.erdos_renyi_graph(len(selected_nodes), p, seed=wat_seed_value, directed=False)

    mapping = {i: node for i, node in enumerate(selected_nodes)}
    Gr = nx.relabel_nodes(Gr, mapping)
    wat_G = np.zeros([len(Gr.edges)+1, 2]).astype(np.int64)
    wat_G[0,0]=[wat_len, len(Gr.edges)]
    wat_G[1:,:]=[list(edge) for edge in Gr.edges]

    return wat_G, selected_nodes

import pymetis
def parittion_metis(edges, n_nodes, n_parts):
    adjacency   = [[] for _ in range(n_nodes)]

    for u, v in edges:
        adjacency[u-1].append(v-1)
        adjacency[v-1].append(u-1)
    edgecuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency)

    return edgecuts, membership

def balanced_partition(partitions):
    # Count the number of nodes in each partition
    part_counts = {}
    for p in partitions:
        part_counts[p] = part_counts.get(p, 0) + 1
    
    # Find the maximum count among partitions
    max_count = max(part_counts.values()) if part_counts else 0
    
    # Balance partitions by adding nodes to partitions that have fewer nodes
    balanced = partitions.copy()
    for p in part_counts:
        while part_counts[p] < max_count:
            balanced.append(p)       # Append partition id to balance nodes count
            part_counts[p] += 1      # Update count
    
    return balanced
