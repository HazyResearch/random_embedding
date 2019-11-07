import unittest
import torch
import numpy as np
from random_embeddings import RandomEmbedding, create_random_circulant_embeddings

class RandomEmbeddingTest(unittest.TestCase):
    def test_forward(self):
        for (n,d) in [(300,3),(301,3),(3,301),(30000,800)]:
            avg_norm = 3
            num_blocks = int(np.ceil(n/d))
            n_ceil = num_blocks * d
            emb = RandomEmbedding(n,d,avg_embedding_norm=avg_norm)
            x = torch.tensor(range(n), dtype=torch.int64)
            # With the above 'x', the output of the embedding module should be equal to emb.weight
            out = emb(x)
            # Test that shapes/dimensions are correct
            self.assertTrue(list(out.shape) == [n,d])
            self.assertTrue(list(emb.weight.shape) == [n,d])
            self.assertTrue(emb.embedding_dim == d and emb.num_embeddings == n)
            # Ensure that the (1) number of unique elements in the output matches what it should be for a circulant matrix,
            # (2) that output exactly matches the emb.weight tensor, and (3) that the average row norm of output tensor is correct.
            if n_ceil * d < 1000:
                self.assertTrue(torch.abs(out).unique().numel() == n_ceil)
            else:
                # When sampling many i.i.d Gaussians, often get collisions, so we simply confirm the
                # number of unique entries is <= n_ceil.
                self.assertTrue(torch.abs(out).unique().numel() <= n_ceil)
            self.assertTrue(torch.all(torch.eq(out,emb.weight)).item())
            self.assertTrue(np.isclose(torch.mean(torch.sqrt(torch.sum(out**2,dim=1))).item(), avg_norm))
            # Check that in each of the d x d blocks of the output, that (1) the diagonal always contains only a single
            # unique absolute value, and (2) that the number of unique absolute values in the block is equal (or <=) to d.
            for i in range(num_blocks):
                block = out[i*d:(i+1)*d,:]
                self.assertTrue(torch.abs(torch.diag(block)).unique().numel() == 1)
                if n_ceil * d < 1000:
                    self.assertTrue(torch.abs(block).unique().numel() == d)
                else:
                    self.assertTrue(torch.abs(block).unique().numel() <= d)

if __name__ == "__main__":
    unittest.main()
