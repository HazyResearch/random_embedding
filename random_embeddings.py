import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import circulant

def create_random_circulant_embeddings(n, d, avg_embedding_norm=1):
    num_blocks = int(np.ceil(n/d))
    n_ceil = num_blocks * d
    rand_vec = np.random.randn(n_ceil)
    rand_signs = np.random.randint(2,size=n_ceil) * 2 - 1
    Xq = np.zeros((n_ceil, d))
    for i in range(num_blocks):
        a = i * d
        b = (i + 1) * d
        Xq[a:b,:] = circulant(rand_vec[a:b]) * rand_signs[a:b]
    Xq = Xq[:n,:]
    # normalize embeddings so that the average L2 norm of a row of the embedding matrix
    # is equal to avg_embedding_norm.
    curr_avg_embedding_norm = np.mean(np.sqrt(np.sum(Xq**2,axis=1)))
    Xq = Xq * avg_embedding_norm / curr_avg_embedding_norm
    return torch.tensor(Xq, dtype=torch.float32)

# Note: This is not yet implemented in a memory efficient manner
class RandomEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 avg_embedding_norm=1,
                 _weight=None):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            create_random_circulant_embeddings(
                num_embeddings, embedding_dim,
                avg_embedding_norm=avg_embedding_norm
            ),
            requires_grad=False
        )
