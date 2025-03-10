import numpy as np

import os

n = [1000, 2000, 3000, 5000, 7000, 10000]
m = [2 * item for item in n]
d = 10

for folder in range(9, 10):
    subfolder = f'new{folder}'

    for it in range(len(n)):
        B = np.random.choice(d, n[it], replace=True)

        I = np.zeros([n[it], m[it]])
        for i in range(n[it]):
            edges = np.random.choice(m[it], B[i], replace=False)
            I[i, edges] = 1

        k = np.sum(I, axis=0)
        I = np.copy(I[:, k > 1])

        k = np.sum(I, axis=1)
        I = np.copy(I[k > 0, :])

        nn, mm = I.shape

        constraints = []

        for i in range(mm):
            nodes = list(np.argwhere(I[:, i] > 0).T[0])
            constraints.append(nodes)

        header = [nn, mm]

        base_path = './data/hypergraph_data/synthetic/syn_random'
        file_name = f'Hyp_{nn}_{mm}.txt'
        full_path = os.path.join(base_path, subfolder, file_name)

        if not os.path.exists(os.path.join(base_path, subfolder)):
            os.makedirs(os.path.join(base_path, subfolder))

        with open(full_path, 'w') as f:
            f.write(str(header[0]))
            f.write(' ')
            f.write(str(header[1]))
            f.write('\n')

            i = 0

            for cons in constraints:
                if len(cons) > 1:
                    for ns in cons:
                        i += 1
                        f.write(str(ns + 1))
                        if i < len(cons):
                            f.write(' ')
                    f.write('\n')
                    i = 0