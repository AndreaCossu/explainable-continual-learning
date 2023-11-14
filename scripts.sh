#!/bin/bash

cuda_device=0

# MNIST MLP
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset mnist --model mlp --hidden_size 256 --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300

# CIFAR RESNET
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset cifar --model cnn --epochs 30 --lr 0.001 --train_mb_size 64 --joint_epochs 50 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300

# SPEECH LSTM
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset speech --model rnn --hidden_size 256 --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300

# SPEECH ESN
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset speech --model esn --hidden_size 2000 --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --spectral_radius 0.9 --input_scaling 1 --replay_mem_size 300

# SPEECH RON
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset speech --model ron --hidden_size 500 --epochs 30 --lr 0.0001 --train_mb_size 64 --joint_epochs 10 --joint_lr 0.001 --joint_train_mb_size 128 --spectral_radius 0.99 --input_scaling 1 --dt 0.1 --gamma_min 0.5 --gamma_max 2.0 --epsilon_min 0.5 --epsilon_max 2.0 --replay_mem_size 300

# SPEECH CNN
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --explanator shap --dataset speech --model cnn --epochs 10 --lr 0.001 --train_mb_size 64 --joint_epochs 30 --joint_lr 0.001 --joint_train_mb_size 128 --replay_mem_size 300
