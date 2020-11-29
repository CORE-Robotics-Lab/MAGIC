#!/bin/bash
export OMP_NUM_THREADS=1

python -u test.py \
  --env_name grf \
  --nagents 3 \
  --nprocesses 1 \
  --num_epochs 100 \
  --epoch_size 10 \
  --hid_size 128 \
  --max_steps 80 \
  --gnn_type gat \
  --directed \
  --gat_num_heads 8 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --use_gconv_encoder \
  --gconv_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --message_encoder \
  --message_decoder \
  --recurrent \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_lagents 3 \
  --num_controlled_ragents 0 \
  --reward_type scoring \
  --run_num 12 \
  --ep_num 0 \
  | tee test_grf.log
  
# Should revise according to the tested trained model
