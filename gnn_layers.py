import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
# Graph Attention layers that can process differential graphs
class GraphAttention(nn.Module):

    def __init__(self, in_features, out_features, dropout, negative_slope, num_heads=1, bias=True, self_loop_type=2, average=False, normalize=False):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
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
    	"""
		Forward function for the graph attention layer used in MAGIC

		Arguments:
			input (tensor): input of the graph attention layer [N * in_features, N: number of agents]
			adj (tensor): the learned communication graph (adjancy matrix) by the sub-scheduler [N * N]
    	"""

        # self.W size: in_features * (num_heads*out_features)

        # perform linear transformation on the input, and generate multiple heads
        # h (tensor): the matrix after performing the linear transformation [N * num_heads * out_features]
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)
        N = h.size()[0]
    
        if self.self_loop_type == 0:
            adj = adj * (torch.ones(N, N) - torch.eye(N, N))
        elif self.self_loop_type == 1:
            adj = torch.eye(N, N) + adj * (torch.ones(N, N) - torch.eye(N, N))   
        else:
            pass
        
        e = []

        # compute the unnormalized coefficients
        # a_i, a_j (tensors): weight vectors to compute the unnormalized coefficients [num_heads * out_features * 1]
        for head in range(self.num_heads):
            # coeff_i, coeff_j (tensors): intermediate matrices to calculate unnormalized coefficients [N * 1]
            coeff_i = torch.mm(h[:, head, :], self.a_i[head, :, :])
            coeff_j = torch.mm(h[:, head, :], self.a_j[head, :, :])
            # coeff (tensor): the matrix of unnormalized coefficients for each head [N * N * 1]
            coeff = coeff_i.expand(N, N) + coeff_j.transpose(0, 1).expand(N, N)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e (tensor): the matrix of unnormalized coefficients for all heads [N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used to limit the large unnormalized coefficient values (TODO)
        e = self.leakyrelu(torch.cat(e, dim=-1)) 
            
        # adj: [N * N * num_heads]
        adj = adj.unsqueeze(-1).expand(N, N, self.num_heads)
        # attention (tensor): [N * N * num_heads]
        attention = e * adj
        attention = F.softmax(attention, dim=1)
        attention = attention * adj   
        # normalize
        if self.normalize:
            if self.self_loop_type != 1:
                attention += 1e-15
            attention = attention / attention.sum(dim=1).unsqueeze(dim=1).expand(N, N, self.num_heads)
            attention = attention * adj
        # dropout on the coefficients  
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
    
