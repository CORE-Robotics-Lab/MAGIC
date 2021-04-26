#!/bin/bash
export OMP_NUM_THREADS=1

python -u run_baselines.py \
  --env_name traffic_junction \
  --nagents 20 \
  --dim 18 \
  --max_steps 80 \
  --add_rate_min 0.05 \
  --add_rate_max 0.05 \
  --difficulty hard \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 4000 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --gacomm \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 0 \
  | tee train_tj_hard.log