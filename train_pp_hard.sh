#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name predator_prey \
  --nagents 10 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 1000 \
  --epoch_size 10 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.0003 \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --use_gat_encoder \
  --gat_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_decoder \
  --save \
  --seed 0 \
  | tee train_pp_hard.log
