import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention
from gnn_layers import GraphConvolution

class MAGIC(nn.Module):
    def __init__(self, args, num_inputs):
        super(MAGIC, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.recurrent = args.recurrent
        
        if args.gnn_type == 'gat':
            dropout = 0
            negative_slope = 0.2
            self.gconv1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
            self.gconv2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
            if args.use_gconv_encoder:
                self.gconv_encoder = GraphAttention(args.hid_size, args.gconv_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gconv_gat_normalize)
            
        if args.gnn_type == 'gcn':
            self.gconv1 = GraphConvolution(args.hid_size, args.hid_size, self_loop_type=args.self_loop_type1)
            self.gconv2 = GraphConvolution(args.hid_size, args.hid_size, self_loop_type=args.self_loop_type2)
        
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        self.encoder = nn.Linear(num_inputs, args.hid_size)

        self.init_hidden(args.batch_size)
        self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        if not args.first_graph_complete:
            if args.use_gconv_encoder:
                self.hard_attn1 = nn.Sequential(
                    nn.Linear(args.gconv_encoder_out_size*2, int(args.gconv_encoder_out_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(args.gconv_encoder_out_size/2), int(args.gconv_encoder_out_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(args.gconv_encoder_out_size/2), 2))
            else:
                self.hard_attn1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, int(self.hid_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(self.hid_size/2), int(self.hid_size/8)),
                    nn.ReLU(),
                    nn.Linear(int(self.hid_size/8), 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gconv_encoder:
                self.hard_attn2 = nn.Sequential(
                    nn.Linear(args.gconv_encoder_out_size*2, int(args.gconv_encoder_out_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(args.gconv_encoder_out_size/2), int(args.gconv_encoder_out_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(args.gconv_encoder_out_size/2), 2))
            else:
                self.hard_attn2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, int(self.hid_size/2)),
                    nn.ReLU(),
                    nn.Linear(int(self.hid_size/2), int(self.hid_size/8)),
                    nn.ReLU(),
                    nn.Linear(int(self.hid_size/8), 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.hard_attn1.apply(self.init_linear)
            if not args.second_graph_complete:
                self.hard_attn2.apply(self.init_linear)
                   
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        
        self.value_head = nn.Linear(2 * self.hid_size, 1)
        
        self.tanh = nn.Tanh()


    def forward(self, x, info={}):

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        hidden_state, cell_state = self.f_module(x.squeeze(), (hidden_state, cell_state))

        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # Mask communcation from dead agents (in TJ)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        if not self.args.first_graph_complete:
            if self.args.use_gconv_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gconv_encoder(comm, adj_complete)
                adj1 = self.get_adj_matrix(self.hard_attn1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.get_adj_matrix(self.hard_attn1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)
            
        if self.args.gnn_type == 'gat':
            comm = F.elu(self.gconv1(comm, adj1))
        else:
            comm = F.relu(self.gconv1(comm, adj1))
        
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gconv_encoder:
                encoded_state2 = encoded_state1
                adj2 = self.get_adj_matrix(self.hard_attn2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.get_adj_matrix(self.hard_attn2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        comm = self.gconv2(comm, adj2)
        
        # Mask communication to dead agents (in TJ)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        x, extras = x
        x = self.encoder(x)

        hidden_state, cell_state = extras

        return x, hidden_state, cell_state


    def init_linear(self, m):
        if type(m) == nn.Linear:
#             m.weight.data.normal_(0, self.init_std)
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def get_adj_matrix(self, hard_attn_model, hidden_state, agent_mask, directed=True):
        # hidden_state size: n * hid_size
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input size: n * n * (2*hid_size)
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output size: n * n * 2
        if directed:
            hard_attn_output = F.gumbel_softmax(hard_attn_model(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*hard_attn_model(hard_attn_input)+0.5*hard_attn_model(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output size: n * n * 1
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and its transpose size: n * n
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj size: n * n
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj
