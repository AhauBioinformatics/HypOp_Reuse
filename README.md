# HypOp-Reuse
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

## Contents
- [Overview](#overview)
- [Experimental Setup](#experimental-setup)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Reusability Experiments](#reusability-experiments)
- [Parameter Introductions](#parameter-introductions)
- [License](./LICENSE)


# Overview
This repository contains the Pytorch implementation of HypOp framework, as described in Nature Machine Intelligence paper [Distributed Constrained Combinatorial Optimization leveraging Hypergraph Neural Networks](https://www.nature.com/articles/s42256-024-00833-7). HypOp is a combinatorial optimization solver designed to efficiently address general problems with higher-order constraints. It leverages hypergraph neural networks to extend previous algorithms, enabling them to handle arbitrary cost functions. Additionally, it incorporates a distributed training architecture to efficiently manage large-scale tasks.

# Experimental Setup
Our experiment were conducted on an Intel Xeon 8358P CPU with a single NVIDIA RTX 4090 GPU or multiple GPUs, depending on the task.

---
# System Requirements
The source codes were developed in python 3.9 using Pytorch 2.3.1. The required python dependencies are given in `dependency.txt` file and have been tested on CentOS-7 with CUDA version 11.8.
```bash
conda create --name environment_name python=3.9
conda activate environment_name
pip install -r dependency.txt
```

# Quick Start
### Hypergraph MaxCut on CPU:

```bash
conda activate environment_name
python run run.py
```
### Distributed Hypergraph MaxCut on MULTI-GPUs
This section requires at least two GPUs on your system. We use the NCCL backend for distributed training. You can install NCCL by following the instructions below:
```bash
conda install nccl
```
#### Step1: Distributed Training

In `run_dist.py`, set `test_mode` variable to `dist`.

```python
python -m torch.distributed.launch run_dist.py
```

#### Step2: Postprocessing

In configs, set "load best out" to true, set "epoch" to 0.

In `run_dist.py`, set `test_mode` variable to `infer`.

```python
python -m torch.distributed.launch run_dist.py
```

---
# Reusability Experiments
Our experiments comprise three components. The configuration files for these experiments are detailed below:
### Part1: Reproduction of Reported Results
- **Experiments on CPU**: ( In `run.py`, set the path of config files to accommodate different experiments.)
    - Hypergraph MaxCut task
      - HypOp: `configs/Hypermaxcut_hypop.json`
      - Bipartite: `configs/Hypermaxcut_bipartite.json`
      - Simulated Annealing (SA): `configs/Hypermaxcut_sa.json`
      - ADAM: `configs/Hypermaxcut_adam.json`
      
    - Graph Maximum independent set (MIS) task
      - HypOp: `configs/MIS_hypop.json`
      - PI-GNN: `configs/MIS_pignn.json`
      

  - **Experiments on multi-GPUs**: ( Including two steps, use `run_dist.py`)
      - Hypergraph MaxCut task using HypOp in distributed experiments
        - Step1: Ditributed training, set `test_mode` in `run_dist.py` to `dist`, set config file to `configs/dist_configs/maxcut_stanford_for_dist.json`.
        - Step2: Fine-tuning infer, set `test_mode` in `run_dist.py` to `infer`, set config file to `configs/infer_configs/maxcut_stanford_for_dist.json`.
      - Hypergraph MaxCut task using HypOp in parallel multi-GPUs experiments
        - Step1: Parallel training, set `test_mode` in `run_dist.py` to `multi_gpu`, set config file to `configs/dist_configs/maxcut_stanford_for_paral`.
        - Step2: Fine-tuning infer, set `test_mode` in `run_dist.py` to `infer`, set config file to `configs/dist_configs/maxcut_stanford_for_paral`.
  
### Part2: Robustness Evaluation of HypOp
This section primarily evaluates the robustness of HypOp across various distributed training configurations.
- **Alternate the number of GPUs**
    - Hypergraph MaxCut task using HypOp: Set GPU number `'num_gpus'` in `'configs/dist_configs/maxcut_stanford_for_dist.json'` to `1,2,4,8`.
- **Alternate the graph partition strategy**
    - Hypergraph MaxCut task using HypOp: set `'partition_strategy'` in `'configs/dist_configs/maxcut_stanford_for_dist.json'` to `'original', 'round_robin', 'random'`.
- **Alternate fine-tuning algorithms**
    - Hypergraph MaxCut task using HypOp: set `'fine_tuning'` in `'configs/dist_configs/maxcut_stanford_for_dist.json'` to `'PSO', 'ACO', 'GA'`.

### Part3: Generalizability Evaluation
This part mainly uses HypOp to solve other combinatorial optimization problems.
- **Apply HypOp on Maximum Clique Problem**

    ```bash
    python maxclique_training.py
    ```
- **Apply HypOp on Quadratic Assignment Problem**
    ```bash
    python QAP_training.py
    ```
  
---
# Parameter Introductions
### Mode Parameters
    - data: stanford/hypergraph/NDC
    - mode: maxcut/maxind/QUBO
    - Adam: true/false; false default
### Training Parameters
    - lr: learning rate
    - epoch: number of training epochs
    - tol: training loss tolerace (used in Early Stop strategy)
    - patience: training patience (used in Early Stop strategy)
    - GD: false/true (true for direct optimization with gradient descent) 
    - load_G: false/true (true for when G is already computed and saved and want to load it)
    - sparsify: false/true (true for when the graph is too dense and need to sparsify it)
    - sparsify_p: the probability of removing an edge if sparsify is true
### Utils Parameters    
    - mapping: threshold/distribution
    - threshold: trivial mapping that maps numbers less than 0.5 to 0 and greater than 0.5 to 1
    - distribution: mapping using simulated annealing
    - N_realize: only used when mapping = distribution: number of realizations from the distribution
    - Niter_h: only used when mapping = distribution: number of simulated annealing iterations
    - t: simulated annealing initial temperature
    - random_init: initializing simulated annealing randomly (not with HyperGNN)
    - logging_path: path that the log file is saved
    - res_path: path that the result file is saved
    - folder_path: directory containing the data
### Transfer Learning Parameters
    - model_save_path: directory to save the model
    - model_load_path: directory to load the model
	- transfer: false/true (true for transfer learning)
	- initial_transfer: false/true (true for initializing the models with a pre-trained model)
