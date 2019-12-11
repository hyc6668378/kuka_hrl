#!/bin/bash
python sac.py \
   --experiment_name sac_kuka_explor_adv \
   --use_sample_dist \
   --use_adv &&

python sac.py \
   --experiment_name sac_kuka_explor \
   --use_sample_dist &&

python sac.py \
   --experiment_name sac_kuka_explor_adv_half_dete \
   --use_sample_dist \
   --use_adv  \
   --use_half_dete &&

python sac.py \
   --experiment_name sac_kuka_explor_half_dete \
   --use_sample_dist \
   --use_half_dete  &&

shutdown