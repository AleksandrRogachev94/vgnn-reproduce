import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Single self-attention head.
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Weights matrix
        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_normal_(self.W.weight.data)
        # Attention weights
        self.a = nn.Parameter(torch.empty(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # Adapted from original code
    def attention(self, linear, a, N, data, edge):
        assert not torch.isnan(data).any()
        # Apply linear transformation to all nodes
        data = linear(data)
        # stack 2 tensors among which we will perform self-attention
        # 2 x number of connections x feature dim
        h = torch.stack((data[edge[0, :], :], data[edge[1, :], :]), dim=0)
        # concatenate each pair into 1 vector
        # edge_h: 2*feature dim x number of connections
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        # Calculate unnormalized self-attention coefficients
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.out_features))
        assert not torch.isnan(edge_e).any()

        # Apply softmax on sparse representation
        edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).to(device))
        # e_rowsum: N x 1
        row_check = e_rowsum == 0
        e_rowsum[row_check] = 1
        zero_idx = row_check.nonzero()[:, 0]
        edge_e = edge_e.add(
            torch.sparse.FloatTensor(zero_idx.repeat(2, 1), torch.ones(len(zero_idx)).to(device), torch.Size([N, N])))
        # edge_e: E
        h_prime = torch.sparse.mm(edge_e, data)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward(self, h, adj):
        return self.attention(self.W, self.a, h.shape[0], h, adj)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'