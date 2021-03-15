#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 6 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 1500 \
  --epoch_size 10 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --gnn_type gat \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --self_loop_type1 1 \
  --self_loop_type2 1 \
  --first_graph_complete \
  --second_graph_complete \
  --message_decoder \
  --recurrent \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 0 \
  | tee train_tj_easy.log

  ## easy
  # --nagents 5 \
  # --dim 6 \
  # --max_steps 20 \
  # --add_rate_min 0.3 \
  # --add_rate_max 0.3 \
  # --difficulty easy \

  ## medium
  # --nagents 10 \
  # --dim 14 \
  # --max_steps 40 \
  # --add_rate_min 0.2 \
  # --add_rate_max 0.2 \
  # --difficulty medium \

  ## hard
  # --nagents 20 \
  # --dim 18 \
  # --max_steps 80 \
  # --add_rate_min 0.05 \
  # --add_rate_max 0.05 \
  # --difficulty hard \
