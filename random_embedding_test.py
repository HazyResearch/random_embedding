import unittest
import time
import torch
import numpy as np
from random_embedding import RandomEmbedding

class RandomEmbeddingTest(unittest.TestCase):
    def test_forward(self):
        for device in ['cuda','cpu']:
            if device=='cpu' or (device=='cuda' and torch.cuda.is_available()):
                print(''.format(device))
                t1 = time.perf_counter()
                for (n,d) in [(300,3),(301,3),(3,301),(30000,800)]:
                    avg_norm = 3
                    emb = RandomEmbedding(n,d,avg_embedding_norm=avg_norm)
                    if device == 'cuda':
                        emb.cuda()
                    x = torch.tensor(range(n), dtype=torch.int64, device=device)
                    out = emb(x)
                    self.check_embedding_output(emb,out,n,d,avg_norm)
                    x2 = torch.tensor([range(n),range(n)], dtype=torch.int64,
                                      device=device)
                    out = emb(x2)
                    self.assertTrue(out.shape == (2,n,d))
                    self.check_embedding_output(emb,out[0,:],n,d,avg_norm)
                    self.check_embedding_output(emb,out[1,:],n,d,avg_norm)
                t2 = time.perf_counter()
                print('Device: {}, time elapsed: {:.3f}s'.format(device, t2-t1))

    def check_embedding_output(self,emb,out,n,d,avg_norm):
        num_blocks = int(np.ceil(n/d))
        n_ceil = num_blocks * d

        # Test that shapes/dimensions are correct
        self.assertTrue(out.shape == (n,d))
        self.assertTrue(emb.embedding_dim == d and 
                        emb.num_embeddings == n)
        # Ensure that the (1) number of unique elements in the output 
        # matches what it should be for a circulant matrix (at most n_ceil),
        # and (2) that the average row norm of output tensor is equal to 
        # avg_norm.
        self.assertTrue(torch.abs(out).unique().numel() <= n_ceil)
        self.assertTrue(
            np.isclose(torch.mean(out.norm(dim=1)).item(), avg_norm)
        )
        # Check that in each of the d x d blocks of the output, that (1) the
        # diagonal always contains only a single unique absolute value, and
        # (2) that the # of unique absolute values in the block is <= d.
        for i in range(num_blocks):
            block = out[i*d:(i+1)*d,:]
            self.assertTrue(
                torch.abs(torch.diag(block)).unique().numel() == 1
            )
            self.assertTrue(torch.abs(block).unique().numel() <= d)

if __name__ == "__main__":
    unittest.main()
