import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
# Graph Attention layers that can process differential graphs
class GraphAttention(nn.Module):

    def __init__(self, in_features, out_features, dropout, negative_slope, num_heads=1, bias=True, self_loop_type=2, average=False, normalize=False):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        if bias:
            if average:
                self.bias = nn.Parameter(torch.DoubleTensor(out_features))
            else:
                self.bias = nn.Parameter(torch.DoubleTensor(num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        # input size: N * in_features
        # self.W size: in_features * (num_heads*out_features)
        # h size: N * num_heads * out_features
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)
        N = h.size()[0]
    
        if self.self_loop_type == 0:
            adj = adj * (torch.ones(N, N) - torch.eye(N, N))
        elif self.self_loop_type == 1:
            adj = torch.eye(N, N) + adj * (torch.ones(N, N) - torch.eye(N, N))   
        else:
            pass
        
        e = []
        # a_i, a_j size: num_heads * out_features * 1
        for head in range(self.num_heads):
            # coeff_i, coeff_j size: N * 1
            coeff_i = torch.mm(h[:, head, :], self.a_i[head, :, :])
            coeff_j = torch.mm(h[:, head, :], self.a_j[head, :, :])
            # coeff size: N * N * 1
            coeff = coeff_i.expand(N, N) + coeff_j.transpose(0, 1).expand(N, N)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e size: N * N * num_heads
        e = self.leakyrelu(torch.cat(e, dim=-1)) 
            
        # adj size: N * N * num_heads
        adj = adj.unsqueeze(-1).expand(N, N, self.num_heads)
        # attention size: N * N * num_heads
        attention = e * adj
        attention = F.softmax(attention, dim=1)
        attention = attention * adj   
        # normalize
        if self.normalize:
            if self.self_loop_type != 1:
                attention += 1e-15
            attention = attention / attention.sum(dim=1).unsqueeze(dim=1).expand(N, N, self.num_heads)
            attention = attention * adj
            
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # output size: N * (num_heads*out_features)
        output = []
        for head in range(self.num_heads):
            h_prime = torch.matmul(attention[:, :, head], h[:, head, :])
            output.append(h_prime)
        if self.average:
            output = torch.mean(torch.stack(output, dim=-1), dim=-1)
        else:
            output = torch.cat(output, dim=-1)
        
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
# Revised from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py, revised GCN layers not used in MAGIC
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, self_loop_type=2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.self_loop_type = self_loop_type
        self.weight = nn.Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        N = input.size()[0]
        
        if self.self_loop_type == 0:
            adj = adj * (torch.ones(N, N) - torch.eye(N, N))
        elif self.self_loop_type == 1:
            adj = torch.eye(N, N) + adj * (torch.ones(N, N) - torch.eye(N, N))   
        else:
            pass
        
        degrees = adj.sum(dim=1)
        normalized_D = torch.diag_embed(degrees.pow(-0.5))
        normalized_adj = torch.mm(torch.mm(normalized_D, adj), normalized_D)
        support = torch.mm(input, self.weight)
        output = torch.mm(normalized_adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
