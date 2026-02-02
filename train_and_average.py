#!/usr/bin/env python3
"""
Combined training and model averaging script.

This script combines the functionality of multiprocess_training and model_average notebooks:
1. Trains multiple models (NN and CNN) using multiprocessing with different seeds and folds
2. Averages the model weights and evaluates the averaged models
"""

import os
import pickle
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold

from src.nn import *
from src.utils import *


def train_dataset(data_name, random_seed, learning_rate, num_epochs, num_folds, n_workers):
    """Train models for a specific dataset."""
    print("\n" + "=" * 80)
    print(f"Training on {data_name.upper()} dataset")
    print("=" * 80)
    
    # Load data
    print("\nLoading vehicle data...")
    train_data_path = f'data/vehicle_data_{data_name}_train.pkl'
    if not os.path.exists(train_data_path):
        print(f"Warning: Training data not found at {train_data_path}. Skipping {data_name}.")
        return
    
    with open(train_data_path, 'rb') as f:
        vehicle_data = pickle.load(f)
    print(f"Data loaded: {len(vehicle_data['data'])} samples")
    model_dir = f'models/{data_name}'
    
    # Initialize models and criterion
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    simple_nn_model = NeuralNetwork()
    conv_nn_model = ConvolutionalNeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    
    # ===================================================================
    # STEP 1: MULTIPROCESS TRAINING
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Training models with multiprocessing")
    print("=" * 80)
    
    # Prepare all training jobs
    jobs = []
    
    for i, (train_index, test_index) in enumerate(skf.split(vehicle_data['data'], vehicle_data['label'])):
        train_subset = {
            'data': vehicle_data['data'][train_index],
            'label': vehicle_data['label'][train_index],
        }
        test_subset = {
            'data': vehicle_data['data'][test_index],
            'label': vehicle_data['label'][test_index],
        }
    
        train_dl = get_dataloader(train_subset)
        test_dl = get_dataloader(test_subset)
    
        for seed in random_seed:
            # Neural Network job
            jobs.append((
                'nn',
                data_name,
                simple_nn_model,
                criterion,
                train_dl,
                test_dl,
                learning_rate,
                num_epochs,
                seed,
                i+1,
            ))
            
            # Convolutional Neural Network job
            jobs.append((
                'cnn',
                data_name,
                conv_nn_model,
                criterion,
                train_dl,
                test_dl,
                learning_rate,
                num_epochs,
                seed,
                i+1,
            ))
    
    print(f'\nPrepared {len(jobs)} training jobs')
    print(f'Running with {n_workers} parallel workers...\n')
    
    # Run training jobs with multiprocessing
    with Pool(processes=n_workers) as pool:
        pool.starmap(train_nn, jobs)
    
    print(f'\n{"=" * 80}')
    print(f'All {len(jobs)} training jobs completed!')
    print("=" * 80)
    
    # ===================================================================
    # STEP 2: MODEL AVERAGING
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Averaging model weights")
    print("=" * 80)
    
    # Load all trained model state dicts
    print("\nLoading trained models...")
    cnn_state_dict_list = []
    nn_state_dict_list = []
    
    for seed in random_seed:
        for fold in range(1, num_folds+1):
            cnn_path = f'{model_dir}/cnn/seed_{seed}/fold_{fold}/best_model.pth'
            nn_path = f'{model_dir}/nn/seed_{seed}/fold_{fold}/best_model.pth'
            cnn_state_dict_list.append(torch.load(cnn_path))
            nn_state_dict_list.append(torch.load(nn_path))
    
    print(f"Loaded {len(cnn_state_dict_list)} CNN models and {len(nn_state_dict_list)} NN models")
    
    # Average CNN model weights
    print("\nAveraging CNN model weights...")
    merged_cnn = ConvolutionalNeuralNetwork()
    merged_cnn_state_dict = merged_cnn.state_dict()
    
    for key in merged_cnn_state_dict.keys():
        cnn_params = [state_dict[key] for state_dict in cnn_state_dict_list]
        merged_cnn_state_dict[key] = torch.mean(torch.stack(cnn_params), dim=0)
    
    merged_cnn.load_state_dict(merged_cnn_state_dict)
    
    # Average NN model weights
    print("Averaging NN model weights...")
    merged_nn = NeuralNetwork()
    merged_nn_state_dict = merged_nn.state_dict()
    
    for key in merged_nn_state_dict.keys():
        nn_params = [state_dict[key] for state_dict in nn_state_dict_list]
        merged_nn_state_dict[key] = torch.mean(torch.stack(nn_params), dim=0)
    
    merged_nn.load_state_dict(merged_nn_state_dict)
    
    # ===================================================================
    # STEP 3: EVALUATE AVERAGED MODELS
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Evaluating averaged models")
    print("=" * 80 + "\n")
    
    # Evaluate on each fold
    for i, (_, test_index) in enumerate(skf.split(vehicle_data['data'], vehicle_data['label'])):
        test_subset = {
            'data': vehicle_data['data'][test_index],
            'label': vehicle_data['label'][test_index],
        }
    
        test_dl = get_dataloader(test_subset)
    
        merged_cnn.eval()
        merged_nn.eval()
    
        cnn_eval_metrics = {}
        nn_eval_metrics = {}
    
        cnn_true_label = []
        cnn_pred_label = []
    
        nn_true_label = []
        nn_pred_label = []
    
        with torch.no_grad():
            for inputs, labels in test_dl:
                cnn_outputs = merged_cnn(inputs)
                nn_outputs = merged_nn(inputs)
                _, cnn_preds = torch.max(cnn_outputs, 1)
                _, nn_preds = torch.max(nn_outputs, 1)
    
                cnn_true_label.extend(labels.cpu().numpy())
                cnn_pred_label.extend(cnn_preds.cpu().numpy())
    
                nn_true_label.extend(labels.cpu().numpy())
                nn_pred_label.extend(nn_preds.cpu().numpy())
    
            cnn_eval_metrics = calculate_metrics(cnn_true_label, cnn_pred_label)
            nn_eval_metrics = calculate_metrics(nn_true_label, nn_pred_label)
        
        print(f'Fold {i+1} CNN Evaluation Metrics: {cnn_eval_metrics}')
        print(f'Fold {i+1} NN Evaluation Metrics: {nn_eval_metrics}')
        print('-' * 80)
    
    # ===================================================================
    # STEP 4: SAVE AVERAGED MODELS
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Saving averaged models")
    print("=" * 80)
    
    cnn_dir = f'{model_dir}/cnn/averaged_model'
    nn_dir = f'{model_dir}/nn/averaged_model'
    create_dir(cnn_dir)
    create_dir(nn_dir)
    
    torch.save(merged_cnn.state_dict(), f'{cnn_dir}/best_model.pth')
    torch.save(merged_nn.state_dict(), f'{nn_dir}/best_model.pth')
    
    print(f"\nAveraged CNN model saved to: {cnn_dir}/best_model.pth")
    print(f"Averaged NN model saved to: {nn_dir}/best_model.pth")
    
    print("\n" + "=" * 80)
    print(f"{data_name.upper()} dataset training completed successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train vehicle classification models')
    parser.add_argument('--data', type=str, default='all', 
                       choices=['kaggle', 'collected', 'all'],
                       help='Dataset to train on (default: all)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 1234, 1337],
                       help='Random seeds for training (default: 0 42 1234 1337)')
    
    args = parser.parse_args()
    
    # Determine which datasets to train
    if args.data == 'all':
        datasets = ['kaggle', 'collected']
    else:
        datasets = [args.data]
    
    print("=" * 80)
    print("Vehicle Classification Training and Averaging Pipeline")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Random seeds: {args.seeds}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Folds: {args.folds}")
    print(f"Workers: {args.workers}")
    print("=" * 80)
    
    # Train each dataset
    for data_name in datasets:
        train_dataset(data_name, args.seeds, args.learning_rate, 
                     args.epochs, args.folds, args.workers)
    
    print("\n" + "=" * 80)
    print("All training completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
