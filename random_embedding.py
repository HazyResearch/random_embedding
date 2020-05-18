import numpy as np
import torch

class RandomEmbedding(torch.nn.Embedding):
    """A class used for efficiently storing random circulant embeddings.

    For a n-by-d embedding matrix, we let each d-by-d submatrix be a
    circulant matrix parameterized by a d-dimensional random vector.
    To add further variability between the rows of these circulant
    submatrices, we multiply each circulant submatrix by a diagonal
    matrix with random {+1,-1} values along the diagonal. This follows
    the convention of (Arora et al., 2020) and (Yu et al., 2017).
    Note that if d does not divide n evenly, than the final circulant
    submatrix will only be partially used.

    ...
    References
    ----------
    S. Arora, A. May, J. Zhang, C. Ré.
        Contextual Embeddings: When Are They Worth It? ACL 2020.

    F. Yu, A. Bhaskara, S. Kumar, Y. Gong, S. Chang.
        On Binary Embedding Using Circulant Matrices. JMLR 2017.

    ...
    Attributes
    ----------
    num_embeddings : int
        The size of the embedding vocabulary.
    embedding_dim : int
        The dimension of each embedding vector.
    avg_embedding_norm : float
        The average norm of a row in the embedding matrix (default 1).
    rand_weight : tensor (dtype = torch.float)
        A random and fixed float tensor storing the parameters of the
        circulant submatrices of the embedding matrix. Its shape is
        (b,embedding_dim), where b = ceil(num_embeddings/embedding_dim).
        Each row of rand_weight corresponds to the parameters for one of
        the circulant submatrices.
    rand_signs : tensor (dtype = torch.bool)
        A random and fixed boolean tensor which flips the signs of
        columns of the circulant matrix. Its shape is (b,embedding_dim),
        where b = ceil(num_embeddings/embedding_dim). For the i^th
        circulant submatrix, we multiply it by a diagonal matrix whose
        diagonal is given by the i^th row of rand_signs.
    ind : tensor (dtype = torch.long)
        A fixed tensor storing the indices [0,...,embedding_dim - 1],
        which is used for accessing a full row of the embedding matrix
        at a time in the forward method.

    ...
    Methods
    -------
    forward(input)
        Takes a tensor (dtype = torch.long) of indices as input, and
        returns the corresponding rows of the random embedding matrix.
    """

    def __init__(self, num_embeddings, embedding_dim, avg_embedding_norm=1):
        """Initializes the random circulant embedding matrix.

        Note that although RandomEmbedding is a subclass of
        nn.Embedding, this constructor ignores the padding_idx,
        norm_type, scale_grad_by_freq, sparse, and _weight arguments
        which can normally be passed to the constructor of the
        nn.Embedding class.

        Parameters
        ----------
        num_embeddings : int
            The size of the embedding vocabulary.
        embedding_dim : int
            The dimension of each embedding vector.
        avg_embedding_norm : float
            The desired average L2 norm of a row in the embedding matrix
            (default 1).
        """

        # we pass in a 0 for num_embeddings and embedding_dim to the superclass
        # constructor so that it doesn't instantiate a large embedding weight
        # matrix.
        super().__init__(0, 0)
        # Now we initialize num_embeddings and embedding_dim properly
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.avg_embedding_norm = avg_embedding_norm
        n, d = self.num_embeddings, self.embedding_dim
        # b is the number of different d-by-d circulant blocks in the matrix.
        b = int(np.ceil(n/d))
        # self.weight is a learnable parameter in nn.Embedding. We set it to 
        # None here because we don't need any learnable parameters.
        self.weight = None

        # Each of the b random d-dimensional rows of rand_weight represents
        # the parameters for one of the b circulant submatrices of the
        # random embedding matrix.
        rand_weight = torch.randn(b, d)

        # We now normalize rand_weight so that the average L2 row norm for 
        # the embedding matrix is equal to avg_embedding_norm. To compute the 
        # average norm of the rows of this circulant embedding matrix, we count
        # the L2 norm of each row of rand_weight[:b-1,:] d times (because
        # there are d rows in the embedding matrix that have the same norm as
        # each of these rows), and we count the L2 norm of 
        # rand_weight[b-1,:] (n-(b-1)*d) times. This is because when d does
        # not divide n evenly, the last row of rand_weight will only be
        # repeated this many times in the embedding matrix.
        curr_avg_norm = (d * torch.sum(rand_weight[:b-1,:].norm(dim=1)) + 
                        (n - (b-1) * d) * rand_weight[b-1,:].norm()) / n
        rand_weight *= avg_embedding_norm / curr_avg_norm.item()

        # ind is used to access a full row of the circulant embedding
        # matrix at a time.
        # rand_signs is used to randomly change the signs of the columns of
        # the rows of the embedding matrix.
        ind = torch.arange(d)
        rand_signs = torch.randint(2, (b,d), dtype=torch.bool)

        # Register these tensors as buffers, so they stay fixed during training.
        self.register_buffer('rand_weight', rand_weight)
        self.register_buffer('ind', ind)
        self.register_buffer('rand_signs', rand_signs)

    def forward(self, input):
        """Returns the requested rows of the embedding matrix.

        Parameters
        ----------
        input : torch.LongTensor
            A tensor of indices specifying which rows of the embedding
            matrix should be returned by the forward method. The values
            of input must all be between 0 and self.num_embeddings - 1.

        Returns
        -------
        tensor (dtype = torch.float)
            A tensor containing the rows of the embedding matrix
            specified by the indices in the input tensor. The returned
            tensor has shape (input.shape, self.embedding_dim).

        Raises
        ------
        TypeError
            If input tensor is not of type torch.long.
        ValueError
            If input tensor has any negative values, or values greater
            than self.num_embeddings - 1.
        """

        if input.dtype != torch.long:
            raise TypeError('Input must be of type torch.long')
        if (torch.sum(input >= self.num_embeddings).item() != 0 or 
                torch.sum(input < 0).item() != 0):
            raise ValueError('Entries of input tensor must all be non-negative '
                             'integers less than self.num_embeddings')
        d = self.embedding_dim
        input_us = input.unsqueeze(-1)
        # Given the input tensor of indices (of shape input.shape), we must
        # return the corresponding d-dimensional rows of the circulant random 
        # embedding matrix. Thus, the output of this forward
        # method will have shape (input.shape,d).
        # For each index in input, we first figure out what circulant block it
        # belongs to (input_us//d), and then access the corresponding row 
        # (x_0,...,x_{d-1}) of self.rand_weight in the order 
        # (x_i,x_{i-1},...,x_0,x_{d-1},x_{d-2}...x_{i+1}), where i is equal to
        # input_us % d.
        # After extracting this row, we multiply it entrywise by the row of the
        # rand_signs matrix corresponding to this circulant block.
        # Note that we index self.rand_weight with (input.shape,1) and
        # (input.shape,d) shaped index tensors, so the output has shape
        # (input.shape,d). Similarly, we index the first dimension of 
        # self.rand_signs with a tensor of shape (input.shape), so the output
        # is also fo shape (input.shape,d).
        return (self.rand_weight[input_us // d,  (input_us - self.ind) % d] * 
               (self.rand_signs[input // d, :] * 2.0 - 1.0))
