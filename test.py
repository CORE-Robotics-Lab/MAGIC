import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
import data
from magic import MAGIC
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import gym

gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='MARL with Graph-Based Communication')

parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
parser.add_argument('--directed', action='store_true', default=False,
                    help='if the graph formed by the agents is directed')
parser.add_argument('--self_loop_type1', default=2, type=int,
                    help='self loop type in gnn (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=2, type=int,
                    help='self loop type in gnn (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gnn_type', default='gat', type=str,
                    help='type of gnn to use (gcn|gat)')
parser.add_argument('--gat_num_heads', default=1, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='if normalize first gat layer')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='if normilize second gat layer')
parser.add_argument('--gconv_gat_normalize', action='store_true', default=False,
                    help='if normilize gconv gat layer')
parser.add_argument('--use_gconv_encoder', action='store_true', default=False,
                    help='if use gconv encoder before learning the first graph')
parser.add_argument('--gconv_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gconv encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False,
                    help='if the first graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False,
                    help='if the second graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False,
                    help='if learn the second graph used in the second gnn layer')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='if use message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False,
                    help='if use message decoder')
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--mean_ratio', default=0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="grf",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other

parser.add_argument('--run_num', default=1, type=int,
                    help='load models in which run')
parser.add_argument('--ep_num', default=0, type=int,
                    help='load models saved from which epoch')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")


init_args_for_env(parser)
args = parser.parse_args()

args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False
env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

policy_net = MAGIC(args, num_inputs)


if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))


model_dir = Path('./saved') / args.env_name / args.gnn_type
if args.env_name == 'grf':
    model_dir = model_dir / args.scenario
curr_run = 'run' + str(args.run_num)
run_dir = model_dir / curr_run 

def run(num_epochs): 
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            batch, s = trainer.run_batch(ep)
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = ep + 1

        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward']/stat['num_episodes'], epoch_time
        ))

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.ep_num == 0:
    path = run_dir / 'model.pt'
else:
    path = run_dir / ('model_ep%i.pt' %(args.ep_num))
    
load(path)

run(args.num_epochs)
if args.display:
    env.end_display()

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)


