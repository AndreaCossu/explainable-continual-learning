#!/bin/bash

cuda_device=1

# MNIST
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --dataset mnist --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300

# CIFAR
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --dataset cifar --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300

# SPEECH
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --dataset speech --epochs 10 --lr 0.0001 --train_mb_size 64 --joint_epochs 10 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300
