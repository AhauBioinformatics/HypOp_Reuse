from src.data_reading import read_uf, read_stanford, read_huge, read_hypergraph, read_hypergraph_task, read_NDC, read_arxiv, convert_and_partition
from src.solver import centralized_solver, centralized_solver_for, centralized_solver_multi_gpu
from src.utils import parittion_metis, balanced_partition
import logging
import os
import h5py
import numpy as np
import json
import timeit
import multiprocessing
from torch import distributed as dist
import random
random.seed(12)

def exp_centralized(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    # with h5py.File(params['res_path'], 'w') as f:
    for K in range(int(params['K'])):
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg" or params['data'] == "bipartite" or  params['data'] == "cliquegraph":
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                elif params['data'] == "task":
                    constraints, header = read_hypergraph_task(path)
                elif params['data'] == "NDC":
                    constraints, header = read_NDC(path)


                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                res, res_th, outs, outs_th, probs, total_time, train_time, map_time = centralized_solver(constraints, header, params, file_name)

                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, training_time: {train_time}, mapping_time: {map_time}')

                if params['mode']=="partition":
                    print(sum(outs))
                    print(sum(outs_th))
                    group={l: [] for l in range(params['n_partitions'])}
                    for i in range(header['num_nodes']):
                        for l in range(params['n_partitions']):
                            if outs[i,l]==1:
                                group[l].append(i)
                    with open(params['res_path'], 'w') as f:
                        json.dump(group, f)



def exp_centralized_for(params):
    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')
    with h5py.File(params['res_path'], 'w') as f:
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                print(f'dealing {file_name}')
                path = folder_path + file_name
                temp_time = timeit.default_timer()
                if params['data'] == "uf":
                    constraints, header = read_uf(path)
                elif params['data'] == "stanford" or params['data'] == "random_reg":
                    constraints, header = read_stanford(path)
                elif params['data'] == "hypergraph":
                    constraints, header = read_hypergraph(path)
                else:
                    log.warning('Data mode does not exist. Only support uf, stanford, and hypergraph')

                res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header, params, file_name)

                time = timeit.default_timer() - temp_time
                log.info(f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))
                if params['mode']=='maxind':
                    N = 200
                    print((np.average(res)) / (N*0.45537))
                f.create_dataset(f"{file_name}", data = res)


import torch

import sys
def exp_centralized_for_multi(proc_id, devices, params):
    print("start to prepare for device")
    dev_id = devices[proc_id]
    torch.cuda.set_device(dev_id)
    TORCH_DEVICE = torch.device("cuda:" + str(dev_id))
    # print("start to initialize process")

    master_ip = "localhost"
    master_port = "29501"
    world_size = len(devices)  # Total number of processes
    rank = proc_id  # Rank of this process, set to 0 for master, 1 for worker

    print("start to initialize process")
    torch.distributed.init_process_group(backend="nccl", init_method=f'tcp://{master_ip}:{master_port}',
                                         world_size=world_size, rank=rank)

    # torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=len(devices), rank=proc_id)
    print("start to train")

    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')

    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "huge":
                constraints, header = read_huge(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "arxiv":
                constraints, header = read_arxiv()
            else:
                log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, arxiv, and NDC.')


            # split the nodes into different devices
            partition_strategy = params['partition_strategy']

            # partition_strategy = "parmetis"
            if partition_strategy == "random":
                print("partition")
                # Allocate in sequence
                total_nodes = header['num_nodes']
                cur_nodes = list(range(total_nodes * proc_id // len(devices), total_nodes * (proc_id + 1) // len(devices)))
                cur_nodes = [c + 1 for c in cur_nodes]
                # with open(f'res/original_cur_nodes_{rank}.txt', 'w') as f:
                #     for item in cur_nodes:
                #         f.write(str(item) + '\n')

            elif partition_strategy == "metis":
                # Use METIS
                total_nodes = header['num_nodes']
                if rank==0:
                    _, membership = parittion_metis(constraints, total_nodes, params['num_gpus'])
                    membership = torch.tensor(membership, dtype=torch.int64, device=f"cuda:{rank}")
                else:
                    membership = torch.zeros(total_nodes, dtype=torch.int64, device=f"cuda:{rank}")
                dist.broadcast(membership, src=0)
                dist.barrier()
                cur_nodes = [i+1 for i, p in enumerate(membership) if p == rank]
                # with open(f'res/metis_cur_nodes_{rank}.txt', 'w') as f:
                #     for item in cur_nodes:
                #         f.write(str(item) + '\n')
            elif partition_strategy == "parmetis":
                partitions = convert_and_partition(path, n_parts=world_size)
                balanced_partition = balant_partition(partitions)
                cur_nodes = [i+1 for i, p in enumerate(balanced_partition) if p == rank]
                # with open(f'res/parmetis_cur_nodes_{rank}.txt', 'w') as f:
                #     for item in cur_nodes:
                #         f.write(str(item) + '\n')
            elif partition_strategy == "extrem":
                with open("data/synthetic_frustrate/G22_partition.txt","r") as f:
                    content = f.read().strip()[1:-1]
                partitions = [int(x.strip()) for x in content.split(",")]
                balanced_partition = balant_partition(partitions)
                cur_nodes = [i+1 for i, p in enumerate(balanced_partition) if p == rank]
                # with open(f'res/extrem_cur_nodes_{rank}.txt', 'w') as f:
                #     for item in cur_nodes:
                #         f.write(str(item) + '\n')

                



            inner_constraint = []
            outer_constraint = []
            for i, c in enumerate(constraints):
                if c[0] in cur_nodes and c[1] in cur_nodes:
                    inner_constraint.append(c)
                elif (c[0] in cur_nodes and c[1] not in cur_nodes) or (c[0] not in cur_nodes and c[1] in cur_nodes):
                    outer_constraint.append(c)

                progress = int((i + 1) / len(constraints) * 50)
                sys.stdout.write("=" * progress + ">" + " " * (50 - progress) + "] " + f"{((i + 1) / len(constraints)) * 100:.2f}%\r")
                sys.stdout.flush()

            sys.stdout.write("\n")

            print("device", dev_id, "start to train")
            res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_for(constraints, header,
                                                                                                params, file_name,
                                                                                                cur_nodes,
                                                                                                inner_constraint,
                                                                                                outer_constraint,
                                                                                                proc_id)

            if res is not None:
                time = timeit.default_timer() - temp_time
                log.info(
                    f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))

                with h5py.File(params['res_path'], 'w') as f:
                    f.create_dataset(f"{file_name}", data=res)



def exp_centralized_for_multi_gpu(proc_id, devices, params):
    print("start to prepare for device")
    dev_id = devices[proc_id]
    torch.cuda.set_device(dev_id)
    TORCH_DEVICE = torch.device("cuda:" + str(dev_id))

    master_ip = "localhost"
    master_port = "29500"
    world_size = len(devices)  
    rank = proc_id 

    print("start to initialize process")
    torch.distributed.init_process_group(backend="nccl", init_method=f'tcp://{master_ip}:{master_port}', world_size=world_size, rank=rank)
    
    print("start to train")


    logging.basicConfig(filename=params['logging_path'], filemode='w', level=logging.INFO)
    log = logging.getLogger('main')
    folder_path = params['folder_path']
    folder_length = len(os.listdir(folder_path))
    print(f'Found {folder_length} files. Start experiments')

    for file_name in os.listdir(folder_path):
        if not file_name.startswith('.'):
            print(f'dealing {file_name}')
            path = folder_path + file_name
            temp_time = timeit.default_timer()
            if params['data'] == "uf":
                constraints, header = read_uf(path)
            elif params['data'] == "stanford" or params['data'] == "random_reg":
                constraints, header = read_stanford(path)
            elif params['data'] == "hypergraph":
                constraints, header = read_hypergraph(path)
            elif params['data'] == "arxiv":
                constraints, header = read_arxiv()
            else:
                log.warning('Data mode does not exist. Add the data mode. Current version only support uf, stanford, random_reg, hypergraph, arxiv, and NDC.')

            print("device", dev_id, "start to train")
            res, res2, res_th, probs, total_time, train_time, map_time = centralized_solver_multi_gpu(constraints, header, params, file_name, proc_id)

            if res is not None:
                time = timeit.default_timer() - temp_time
                log.info(
                    f'{file_name}:, running time: {time}, res: {res}, res_th: {res_th}, res2: {res2}, training_time: {train_time}, mapping_time: {map_time}')
                print(np.average(res))
                print(np.average(res_th))
                print("save res to", params['res_path'])
                with h5py.File(params['res_path'], 'w') as f:
                    f.create_dataset(f"{file_name}", data=res)
