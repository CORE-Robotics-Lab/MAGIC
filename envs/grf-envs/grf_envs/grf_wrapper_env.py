import gym
import gfootball.env as grf_env
import numpy as np

class GRFWrapperEnv(gym.Env):
    
    def __init__(self,):
        self.env = None
        self.num_controlled_lagents = 0
        self.num_controlled_ragents = 0
        self.num_controlled_agents = 0
        self.num_lagents = 0
        self.num_ragents = 0
        self.action_space = None
        self.observation_space = None
        
    def init_args(self, parser):
        env = parser.add_argument_group('GRF')
        env.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper',
                         help="Scenario of the game")        
        env.add_argument('--num_controlled_lagents', type=int, default=3,
                         help="Number of controlled agents on the left side")
        env.add_argument('--num_controlled_ragents', type=int, default=0,
                         help="Number of controlled agents on the right side")  
        env.add_argument('--reward_type', type=str, default='scoring',
                         help="Reward type for training")
        env.add_argument('--render', action="store_true", default=False,
                         help="Render training or testing process")
        
    def multi_agent_init(self, args):
        self.env = grf_env.create_environment(
            env_name=args.scenario,
            stacked=False,
            representation='multiagent',
            rewards=args.reward_type,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=args.render,
            dump_frequency=0,
            logdir='/tmp/test',
            extra_players=None,
            number_of_left_players_agent_controls=args.num_controlled_lagents,
            number_of_right_players_agent_controls=args.num_controlled_ragents,
            channel_dimensions=(3, 3))
        self.num_controlled_lagents = args.num_controlled_lagents
        self.num_controlled_ragents = args.num_controlled_ragents
        self.num_controlled_agents = args.num_controlled_lagents + args.num_controlled_ragents
        self.num_lagents = self.env.num_lteam_players
        self.num_ragents = self.env.num_rteam_players
        if self.num_controlled_agents > 1:
            action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        else:
            action_space = self.env.action_space
        if self.num_controlled_agents > 1:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low[0],
                high=self.env.observation_space.high[0],
                dtype=self.env.observation_space.dtype)
        else:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                dtype=self.env.observation_space.dtype)
            
        # check spaces
        self.action_space = action_space
        self.observation_space = observation_space 
        return
   
    # check epoch arg
    def reset(self):
        self.stat = dict()
        obs = self.env.reset()
        if self.num_controlled_agents == 1:
            obs = obs.reshape(1, -1)
        return obs
    
    def step(self, actions):
        o, r, d, i = self.env.step(actions)
        if self.num_controlled_agents == 1:
            o = o.reshape(1, -1)
            r = r.reshape(1, -1)
        next_obs = o
        rewards = r
        dones = d
        infos = i
        self.stat['success'] = infos['score_reward']
        
        return next_obs, rewards, dones, infos
        
    def seed(self):
        return
    
    def render(self):
        self.env.render()
        
    def exit_render(self):
        self.env.disable_render()
        