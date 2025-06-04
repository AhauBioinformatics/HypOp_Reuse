import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from src.dataset.data_loader import QAPDataset
import numpy as np
import copy
import random
from itertools import chain
import logging
# ------------------ model ------------------
class single_node_xavier(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(single_node_xavier, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x



def normalize_adj_matrix(A):
    degree = A.sum(dim=1)

    epsilon = 1e-9
    degree = degree + epsilon

    # D^(-1/2)
    D_inv_sqrt = torch.diag(degree.pow(-0.5))

    # D^(-1/2) * A * D^(-1/2)
    normalized_A = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)

    return normalized_A


# ------------------ loss function ------------------
def criterion_np(mapping, D, F):
    n = len(mapping)
    mapping = [mapping[i+1] for i in range(n)]
    mapping_tensor = torch.tensor(mapping, dtype=torch.long)

    mapped_F = F[mapping_tensor, :][:, mapping_tensor]
    total_cost = torch.sum(D * mapped_F)
    return total_cost

def criterion(mapping, D, H):

    H_mapped = torch.einsum('ij,jk,kl->il', mapping, H, mapping.T)

    total_cost = torch.sum(D * H_mapped)
    return total_cost
# -----------------fun-tuning------------------
def float2index(best_out):
    sorted_items = sorted(best_out.items(), key=lambda item: item[1])
    sorted_keys = [item[0] for item in sorted_items]

    n = len(best_out)
    integer_mapping = {sorted_keys[i]: i for i in range(n)}

    best_out_index = {k: integer_mapping[k] for k in best_out}

    return best_out_index
def swap_random(res):
    keys = list(res.keys())
    n = len(keys)

    if n == 0:
        return res

    num_to_swap = n // 2
    keys_to_swap = random.sample(keys, num_to_swap)

    half = num_to_swap // 2
    group1 = keys_to_swap[:half]
    group2 = keys_to_swap[half:]

    temp = copy.deepcopy(res)
    for key1, key2 in zip(group1, group2):
        temp[key1], temp[key2] = temp[key2], temp[key1]

    return temp
def mapping_distribution(best_outs, perm_mat,D, H, n,random_init="none", t=0.3, Niter_h=10):
    if random_init=='one_half':
        best_outs= {x: 0.5 for x in best_outs.keys()}
    elif random_init=='uniform':
        best_outs = {x: np.random.uniform(0,1) for x in best_outs.keys()}
    elif random_init == 'threshold':
        best_outs = {x: 0 if best_outs[x] < 0.5 else 1 for x in best_outs.keys()}

    best_loss = float('inf')
    _loss = criterion_np
    _gt_loss = criterion
    # res = {x: np.maxclique_data.choice(range(2), p=[1 - best_outs[x], best_outs[x]]) for x in best_outs.keys()}
    res = float2index(best_outs)

    # best_loss = _loss(res, D, H)
    best_res = copy.deepcopy(res)
    gt_loss = _gt_loss(perm_mat, D, H)
    print(gt_loss)

    for it in range(Niter_h):
        print('iter',it)
        temp = copy.deepcopy(res)
        temp = swap_random(temp)
        lt = _loss(temp, D,H)
        l1 = _loss(res, D,H)

        if lt < l1 or np.exp(- (lt - l1) / t) > np.random.uniform(0, 1):
            best_res = copy.deepcopy(temp)
            best_loss = copy.deepcopy(lt)
        t = t * 0.95
        # print(best_loss)
        rel_loss = (best_loss-gt_loss)/(gt_loss+0.001)
        print('rel_loss',rel_loss.item())

    return best_res,rel_loss
# ------------------ training ------------------
def train_hp_model(criterion, dataloader, num_epochs=25):

    for inputs in dataloader:

        name = inputs['name']
        print(name)
        TORCH_DEVICE = torch.device('cpu')
        dataset_size = len(dataloader.dataset)
        TORCH_DTYPE = torch.float32

        n = inputs['Fi'].shape[1]

        embed = nn.Embedding(n, n)
        embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
        conv1 = single_node_xavier(n, n//2)
        conv2 = single_node_xavier(n//2, 1)

        parameters = chain(conv1.parameters(), conv2.parameters(), embed.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.01)

        perm_mat = torch.squeeze(inputs['gt_perm_mat'])
        D = torch.squeeze(inputs['Fi'])
        H = torch.squeeze(inputs['Fj'])
        D_emb = normalize_adj_matrix(D)
        H_emb = normalize_adj_matrix(H)
        prev_loss = 100
        best_loss = float('inf')

        count = 0
        patience = 500
        p = 0

        for epoch in range(num_epochs):
            epoch_loss = 0

            # Forward pass
            inputs = embed.weight
            temp = conv1(inputs)
            temp = D_emb @ H_emb @ temp
            temp = torch.relu(temp)
            temp = conv2(temp)
            temp = D_emb @ H_emb @ temp
            # temp = torch.sigmoid(temp)
            temp = torch.softmax(temp / 0.1, dim=-1)
            loss = criterion(temp, D, H)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * perm_mat.size(0)
            gt_loss = criterion(perm_mat, D,H)
            rel_loss = (loss-gt_loss)/gt_loss

            if (epoch+1) % 100 == 0:
                print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
                print(f"rel_loss:{rel_loss}")

            if (abs(loss - prev_loss) <= 1e-5) | ((loss - prev_loss) > 0):
                count += 1
                if count >= patience:
                    print(f'Stopping early on epoch {epoch} (patience: {patience})')
                    break
            else:
                count = 0

            if loss < best_loss:
                p = 0
                best_loss = loss
                best_out = torch.squeeze(temp).reshape(-1, 1)

            else:
                p += 1
                if p > patience:
                    print('Early Stopping')
                    break

            prev_loss = loss
        best_out = best_out.detach().numpy()
        best_out = {i + 1: best_out[i][0] for i in range(len(best_out))}
        res,rel_loss = mapping_distribution(best_out,perm_mat,D, H,n, random_init="none", t=0.3, Niter_h=30)

        log.info(f'{name[0]}:, rel_loss: {rel_loss:.1f}')

        # break
    return res



# ------------------ main function ------------------
if __name__ == '__main__':
    logging.basicConfig(filename="log/qap.log", filemode='w', level=logging.INFO)

    torch.manual_seed(123)
    DATASET_FULL_NAME = 'QAPLIB'
    cls_list = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho','wil']

    for cls in cls_list:
        log = logging.getLogger(f"{cls}")
        print(cls)
        # ------------------- Read data -------------------
        dataset_len = {'train': 100, 'test': 120}
        qap_dataset = {
            x: QAPDataset(DATASET_FULL_NAME, dataset_len[x], cls if x == 'train' else None, sets=x,
                          fetch_online=False) for x in ('train', 'test')}
        train_dataloader = DataLoader(qap_dataset['train'], batch_size=1, shuffle=True)
        test_dataloader = DataLoader(qap_dataset['test'], batch_size=1, shuffle=False)

        # ------------------- Train the model -------------------
        res = train_hp_model(criterion, train_dataloader, num_epochs=4000)
        print(res)
