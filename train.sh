#!/bin/bash

for dataset in "ten_class_dataset" "ten_class_dataset_half_1" "ten_class_dataset_half_2"

do

for model in "efficientnet_b6" "resnet101" "darknet53"

do

	python train.py --dataset=${dataset} --model=${model} > result_train_${model}_${dataset}.log

done

done
