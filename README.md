# MAGIC Implementation
This is the codebase for "[Multi-Agent Graph-Attention Communication and Teaming](https://chrisyrniu.github.io/files/aamas2021.pdf)," which is published in [AAMAS 2021](https://aamas2021.soton.ac.uk/). The implementation is in three domains including Predator-Prey, Traffic-Junction, and Google Researh Football.

## Requirements
* OpenAI Gym
* PyTorch 1.5
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
  Install the game of author's version (add multi-agent observations and fixed some bugs):  
  ```
  git clone git@github.com:chrisyrniu/football.git
  cd football
  pip install .
  ```
  Install the multi-agent environment wrapper for GRF (each agent will receive a local observation in multi-agent settings)  
  ```
  cd envs/grf-envs
  python setup.py develop
  ```

## Training MAGIC
Run `python main.py --help` to check all the options.
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

## Visualization
* Check training progress  
  Use visdom with `--plot`. `--plot_env` should be followed by the name of this plotting environment. `--plot_port` should be followed by the port number you want to use.
* Plot with multiple log files  
  Use plot_script.py (log files in saved/):
  ```
  python plot.py saved/ title Reward
  python plot.py saved/ title Steps-Taken
  ```

## Reference
The training framework is adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
