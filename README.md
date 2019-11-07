# Random Embeddings Repository
This repository is for generating random circulant embedding matrices, which can be used in the training of PyTorch NLP models.
To use the code, simply clone the repo, and import the random_embeddings module.
```
git clone https://github.com/HazyResearch/random_embeddings.git
cd random_embeddings
# If the following code runs without throwing an error, the module is being successfully imported.
python -c "import random_embeddings"
```
In random_embeddings.py, there is module called ```RandomEmbedding```, which is a subclass of ```torch.nn.Embedding```.
This module is meant to serve as a drop-in replacement for ```torch.nn.Embedding``` in the case where you want to use *fixed* random embeddings during the training of a downstream model.
Below is an example of how to create a ```RandomEmbedding``` module for an embedding matrix of size n by d, where the average norm of an embedding vector is equal to c (c can be chosen to match the average embedding vector norm of the pre-trained embedding matrix being replaced); we additionally show how to pass input to this embedding module:
```
from random_embeddings import RandomEmbedding
n,d,c = 100,10,1
emb = RandomEmbedding(n,d,avg_embedding_norm=c)
# Extract the word embeddings for words corresponding to indices 0, 1, and 2.
word_indices = torch.tensor([0,1,2], dtype=torch.int64)
output = emb(word_indices)
print(output.shape)
# output shape will be 3 by 10
```
The RandomEmbedding module creates the random circulant embedding weight matrix (with ```requires_grad=False```), by calling the ```create_random_circulant_embeddings``` function, which is also in random_embeddings.py.
Note that the current implementation stores the entire circulant matrix as an n by d tensor, and so the memory footprint is the same as for an Embedding module of the same dimensions.
We plan to implement a memory-efficient version of this module, so that only the first column of the circulant embedding matrix is stored.

Our implementation is tested under Python 3.6 and PyTorch 1.0 (for our tests, see random_embeddings_test.py).
The code has a dependency on scipy, because scipy.linalg.circulant is used to create the circulant embedding matrix.
