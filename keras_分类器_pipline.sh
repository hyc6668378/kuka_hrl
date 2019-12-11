#!/bin/bash
#python train_keras_grasp_分类器模型+固化.py  --recoder _close_to_obj \
#--input_shape 128 --class_num 4 --test_acc_limit 0.9612&&
python train_keras_grasp_分类器模型+固化.py  --recoder whether_can_grasp_or_not \
--input_shape 32 --class_num 2 --test_acc_limit 0.96&&

echo "ok!"