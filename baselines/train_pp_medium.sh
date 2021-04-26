#!/bin/bash
export OMP_NUM_THREADS=1

python -u run_baselines.py \
  --env_name predator_prey \
  --nagents 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --gacomm \
  --recurrent \
  --save \
  --seed 0 \
  | tee train_pp_medium.log


