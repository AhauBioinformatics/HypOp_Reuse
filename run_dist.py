from src.run_exp import exp_centralized, exp_centralized_for, exp_centralized_for_multi, exp_centralized_for_multi_gpu
from src.solver import QUBO_solver
import json
import torch.multiprocessing as mp
import argparse
import os
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MaxCut experiments')
    parser.add_argument('--test_mode', type=str, default='dist', choices=['dist', 'infer', 'multi_gpu'],
                        help='Test mode to run: dist, infer or multi_gpu')
    parser.add_argument('--dataset', type=str, default='stanford', choices=['stanford', 'arxiv'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, default='configs/dist_configs/maxcut_stanford_for_dist_2U.json',
                        help='Config file dir')
    # parser.add_argument('--gpu_num', type=int, default=4, choices=[1,2,4,8], help='GPU numbers used in training progress')
   #  parser.add_argument('--partition_strategy', type=str, default='original', choices=['original','metis','parmetis','extrem'],
   #                      help='Graph partition strategy')

    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count())

    test_mode = args.test_mode
    dataset = args.dataset
    config = args.config
    
    if test_mode == "infer":
        if dataset == "stanford":
            with open(config) as f:
                params = json.load(f)
        elif dataset == "arxiv":
            with open(config) as f:
                params = json.load(f)
        exp_centralized(params)

    elif test_mode == "dist":
        if dataset == "stanford":
            with open(config) as f:
                params = json.load(f)
        elif dataset == "arxiv":
            with open(config) as f:
                params = json.load(f)

        params["logging_path"] = params["logging_path"].split(".log")[0] + "_train.log"

        if params.get("multi_gpu", False):
            mp.spawn(exp_centralized_for_multi, args=(list(range(params["num_gpus"])), params), nprocs=params["num_gpus"])
        else:
            exp_centralized_for(params)

    elif test_mode == "multi_gpu":
        if dataset == "stanford":
            with open(config) as f:
                params = json.load(f)
        elif dataset == "arxiv":
            with open(config) as f:
                params = json.load(f)
        params["logging_path"] = params["logging_path"].split(".log")[0] + str(params.get("multi_gpu", "")) + "_" + params["data"] + "_test.log"

        mp.spawn(exp_centralized_for_multi_gpu, args=(list(range(params["num_gpus"])), params), nprocs=params["num_gpus"])
