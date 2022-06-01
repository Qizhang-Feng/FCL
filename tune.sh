#!/bin/bash
dataset_name="celeba"
sens_name="gender"
nohup python main_ray.py --dataset_name $dataset_name --sens_name $sens_name --conditional --debias --num_samples 20 > ${dataset_name}_${sens_name}.log 2>&1 &
#nohup python main_ray.py --dataset_name adult --sens_name age --conditional --debias --num_samples 100 > tune_100.log 2>&1 &