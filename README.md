# MAGIC Implementation
This is the codebase for "[Multi-Agent Graph-Attention Communication and Teaming](http://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf)," which is published in [AAMAS 2021](https://aamas2021.soton.ac.uk/) (oral) and presented at [ICCV 2021 Mair2 Workshop](https://www.mair2.com/home) (best paper award). The presentation video of this work can be found [here](https://www.youtube.com/watch?v=g9sQyOjjoFY). A short video demo can be found [here](https://youtu.be/32zZdjC4-6A). The implementation is in three domains including Predator-Prey, Traffic-Junction, and Google Researh Football.

Authors: [Yaru Niu*](https://www.yaruniu.com/), [Rohan Paleja*](https://rohanpaleja.com/), [Matthew Gombolay](https://core-robotics.gatech.edu/people/matthew-gombolay/)

## Requirements
* OpenAI Gym
* PyTorch 1.5 (CPU)
* [visdom](https://github.com/facebookresearch/visdom)
* Predator-Prey and Traffic Junction [Environments](https://github.com/apsdehal/ic3net-envs)
* Fork from authors' version of [Google Research Football](https://github.com/chrisyrniu/football) 


## Testing Environment Setup
* Predator-Prey and Traffic Junction (from IC3Net)
  ```
  cd envs/ic3net-envs
  python setup.py develop
  ```
* Google Research Football  
  Install required apt packages with:  
  ```
  sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
  ```
  Install the game of author's version (added multi-agent observations and fixed some bugs):  
  ```
  git clone https://github.com/chrisyrniu/football.git
  cd football
  pip install .
  ```
  Install the multi-agent environment wrapper for GRF (each agent will receive a local observation in multi-agent settings)  
  ```
  cd envs/grf-envs
  python setup.py develop
  ```

## Training MAGIC
-Run `python main.py --help` to check all the options.  
-Use `--first_graph_complete` and `--second_graph_complete` to set the corresponding communication graph of the first round and second round to be complete (disable the sub-scheduler), respectively.  
-Use `--comm_mask_zero` to block the communication.
* Predator-Prey 5-agent scenario:
  `sh train_pp_medium.sh`
* Predator-Prey 10-agent scenario:
  `sh train_pp_hard.sh`
* Traffic-Junction 5-agent scenario:
  `sh train_tj_easy.sh`
* Traffic-Junction 10-agent scenario:
  `sh train_tj_medium.sh`
* Traffic-Junction 20-agent scenario:
  `sh train_tj_hard.sh`
* Google Research Football 3 vs. 2 (3-agent) scenario:
  `sh train_grf.sh`
  
## Training Baselines
-`cd baselines`  
-Run `python run_baselines.py --help` to check all the options.  
-Use `--comm_action_one` to force all agents to always communicate all (other) agents.  
-Use `--comm_mask_zero` to block the communication.  
-Use `--commnet` to enable CommNet, `--ic3net` to enable IC3Net, `--tarcomm` and `--ic3net` to enable TarMAC-IC3Net, and `--gacomm` to enable GA-Comm.  
-The learning rate for IC3Net in Google Research Football was adjusted as 0.0007, otherwise it was kept as 0.001.  
* Predator-Prey 5-agent scenario:
  `sh train_pp_medium.sh`
* Predator-Prey 10-agent scenario:
  `sh train_pp_hard.sh`
* Traffic-Junction 5-agent scenario:
  `sh train_tj_easy.sh`
* Traffic-Junction 10-agent scenario:
  `sh train_tj_medium.sh`
* Traffic-Junction 20-agent scenario:
  `sh train_tj_hard.sh`
* Google Research Football 3 vs. 2 (3-agent) scenario:
  `sh train_grf.sh`

## Visualization
* Check training progress  
  Use visdom with `--plot`. `--plot_env` should be followed by the name of this plotting environment. `--plot_port` should be followed by the port number you want to use.
* Plot with multiple log files  
  Use plot_script.py (log files in saved/):
  ```
  python plot.py saved/ title Reward
  python plot.py saved/ title Steps-Taken
  ```

## Citation
If you find our paper and repo helpful to your research, please consider citing the paper:
```
@inproceedings{niu2021multi,
  title={Multi-Agent Graph-Attention Communication and Teaming},
  author={Niu, Yaru and Paleja, Rohan and Gombolay, Matthew},
  booktitle={Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={964--973},
  year={2021}
}
```

## Reference
The training framework is adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
