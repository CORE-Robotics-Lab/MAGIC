# MAGIC Implementation
This is the codebase for "[Multi-Agent Graph-Attention Communication and Teaming](https://chrisyrniu.github.io/files/aamas2021.pdf)," which is published in [AAMAS 2021](https://aamas2021.soton.ac.uk/). The implementation is in three domains including Predator-Prey, Traffic-Junction, and Google Researh Football.

## Requirements
* OpenAI Gym
* PyTorch 1.5
* [visdom](https://github.com/facebookresearch/visdom)
* Predator-Prey and Traffic Junction [Environments](https://github.com/apsdehal/ic3net-envs)
* Fork from authors' version of [Google Research Football](https://github.com/chrisyrniu/football) 


## Install Multi-Agent Environment Wrapper for GRF
  `cd grf-envs`

  `python setup.py develop`

## Run Training
  `sh train_pp_medium.sh`

  `sh train_pp_hard.sh`

  `sh train_tj_no_curriculum.sh`

  `sh train_grf.sh`

## Check Training Process and Results
* Use visdom
* Use plot_script.py and saved log file:

  `python plot.py saved/ name Reward`

  `python plot.py saved/ name Steps-Taken`

## Reference
The training method is adapted from [IC3Net](https://github.com/IC3Net/IC3Net)
