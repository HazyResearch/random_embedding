# Random Embedding Repository

This repository is for generating random circulant embedding matrices, which can
be used in the training of PyTorch models.
To use the code, simply clone the repo, and import the random_embedding module.

```
git clone https://github.com/HazyResearch/random_embedding.git
cd random_embedding
# If the following code runs without throwing an error, the module is being
# successfully imported.
python -c "import random_embedding"
# To run our unit tests locally, execute the following command:
python -m unittest random_embedding_test.RandomEmbeddingTest
```

In random_embedding.py, there is a module called ```RandomEmbedding```, which is a
subclass of ```torch.nn.Embedding```. This module is meant to serve as a drop-in
replacement for ```torch.nn.Embedding``` in the case where you want to use
*fixed* random embeddings during the training of a downstream model. Below is an
example of how to create a ```RandomEmbedding``` module for an embedding matrix
of size n-by-d, where the average norm of an embedding vector is equal to c (for
example, c can be chosen to match the average embedding vector norm of the 
pretrained embedding matrix being replaced); we additionally show how to pass
input to this embedding module:

```
import torch
from random_embedding import RandomEmbedding

n,d,c = 100,10,1
emb = RandomEmbedding(n,d,avg_embedding_norm=c)
# Extract the word embeddings for words corresponding to indices 0, 1, and 2.
word_indices = torch.tensor([0,1,2], dtype=torch.int64)
output = emb(word_indices)
print(output.shape)
# output shape will be 3 by 10
```

We have implemented the RandomEmbedding module in a memory-efficient manner,
such that for a vocabulary of size n, only 33n bits are used (32n bits for a
vector of n random floats parameterizing the circulant matrices, and an
additional n bits for a boolean random vector of length n).  For details about
exactly how we define the random circulant embeddings, please see our ACL paper
which we cite below.

Our implementation is tested under Python 3.6 and PyTorch 1.0 (for our tests,
see random_embedding_test.py).

## Citing this Repository

If you use this repository for your research, please cite the following paper:

```
@inproceedings{arora20contextual,
  title     = {Contextual Embeddings: When Are They Worth It?},
  author    = {Simran Arora and Avner May and Jian Zhang and Christopher Ré},
  booktitle = {{ACL}},
  publisher = {Association for Computational Linguistics},
  year      = {2020}
}
```
