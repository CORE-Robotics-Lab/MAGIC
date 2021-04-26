#!/bin/bash
export OMP_NUM_THREADS=1

python -u run_baselines.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --value_coeff 0.01 \
  --lrate 0.001 \
  --max_steps 80 \
  --gacomm \
  --recurrent \
  --save \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --save \
  --seed 0 \
  | tee train_grf.log

