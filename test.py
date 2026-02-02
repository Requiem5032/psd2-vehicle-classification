"""
Vehicle Classification Model Testing Script

This script tests trained Neural Network and CNN models on test datasets
and saves classification reports and confusion matrices to the results directory.
"""

import os
import pickle
import argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.nn import NeuralNetwork, ConvolutionalNeuralNetwork, get_dataloader


def test_model(model, dataloader, model_name, device='cpu'):
    """
    Test a model on the given dataloader.
    
    Args:
        model: The model to test
        dataloader: DataLoader containing test data
        model_name: Name of the model for logging
        device: Device to run inference on
    
    Returns:
        true_labels: List of true labels
        pred_labels: List of predicted labels
    """
    model.to(device)
    model.eval()
    
    true_labels = []
    pred_labels = []
    
    print(f"Testing {model_name}...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    return true_labels, pred_labels


def save_classification_report(true_labels, pred_labels, class_names, save_path):
    """
    Generate and save classification report to a text file.
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: Names of the classes
        save_path: Path to save the report
    """
    report = classification_report(true_labels, pred_labels, 
                                   target_names=class_names, 
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Classification Report\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write("\n")
    
    print(f"Classification report saved to {save_path}")


def save_confusion_matrix(true_labels, pred_labels, class_names, save_path, title):
    """
    Generate and save normalized confusion matrix plot.
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: Names of the classes
        save_path: Path to save the plot
        title: Title for the plot
    """
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, vmin=0, vmax=1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test vehicle classification models')
    parser.add_argument('--data', type=str, default='all', 
                       choices=['kaggle', 'collected', 'all'],
                       help='Dataset to test on (default: all)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['nn', 'cnn', 'all'],
                       help='Model type to test (default: all)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on (default: cpu)')
    
    args = parser.parse_args()
    
    # Determine which datasets to test
    if args.data == 'all':
        datasets = ['kaggle', 'collected']
    else:
        datasets = [args.data]
    
    # Determine which models to test
    if args.model == 'all':
        models_to_test = ['nn', 'cnn']
    else:
        models_to_test = [args.model]
    
    # Class names
    class_names = ['Bike', 'Car']
    
    print("="*60)
    print("Vehicle Classification Model Testing")
    print("="*60)
    print(f"Datasets: {datasets}")
    print(f"Models: {models_to_test}")
    print(f"Device: {args.device}")
    print("="*60)
    print()
    
    # Test each dataset
    for data_name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing on {data_name.upper()} dataset")
        print(f"{'='*60}\n")
        
        # Load test dataset
        test_data_path = f'data/vehicle_data_{data_name}_test.pkl'
        if not os.path.exists(test_data_path):
            print(f"Warning: Test data not found at {test_data_path}. Skipping {data_name}.")
            continue
        
        with open(test_data_path, 'rb') as f:
            test_dataset = pickle.load(f)
        
        print(f"Loaded dataset with {test_dataset['data'].shape[0]} samples")
        print(f"Data shape: {test_dataset['data'].shape}")
        print(f"Unique labels: {torch.unique(test_dataset['label']).tolist()}")
        print()
        
        # Create dataloader
        eval_dataloader = get_dataloader(test_dataset)
        
        # Create results directory for this dataset
        results_dir = f'results/{data_name}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Test Neural Network
        if 'nn' in models_to_test:
            print(f"\n{'-'*60}")
            print("Testing Neural Network")
            print(f"{'-'*60}")
            
            model_path = f'models/{data_name}/nn/averaged_model/best_model.pth'
            if not os.path.exists(model_path):
                print(f"Warning: Model not found at {model_path}. Skipping NN.")
            else:
                nn_model = NeuralNetwork()
                nn_state = torch.load(model_path, map_location=args.device)
                nn_model.load_state_dict(nn_state)
                
                nn_true, nn_pred = test_model(nn_model, eval_dataloader, 
                                             "Neural Network", args.device)
                
                # Save results
                nn_results_dir = os.path.join(results_dir, 'nn')
                os.makedirs(nn_results_dir, exist_ok=True)
                
                save_classification_report(
                    nn_true, nn_pred, class_names,
                    os.path.join(nn_results_dir, 'classification_report.txt')
                )
                
                save_confusion_matrix(
                    nn_true, nn_pred, class_names,
                    os.path.join(nn_results_dir, 'confusion_matrix.png'),
                    f'Neural Network - {data_name.capitalize()} Dataset'
                )
                
                print()
        
        # Test CNN
        if 'cnn' in models_to_test:
            print(f"\n{'-'*60}")
            print("Testing Convolutional Neural Network")
            print(f"{'-'*60}")
            
            model_path = f'models/{data_name}/cnn/averaged_model/best_model.pth'
            if not os.path.exists(model_path):
                print(f"Warning: Model not found at {model_path}. Skipping CNN.")
            else:
                cnn_model = ConvolutionalNeuralNetwork()
                cnn_state = torch.load(model_path, map_location=args.device)
                cnn_model.load_state_dict(cnn_state)
                
                cnn_true, cnn_pred = test_model(cnn_model, eval_dataloader,
                                               "Convolutional Neural Network", args.device)
                
                # Save results
                cnn_results_dir = os.path.join(results_dir, 'cnn')
                os.makedirs(cnn_results_dir, exist_ok=True)
                
                save_classification_report(
                    cnn_true, cnn_pred, class_names,
                    os.path.join(cnn_results_dir, 'classification_report.txt')
                )
                
                save_confusion_matrix(
                    cnn_true, cnn_pred, class_names,
                    os.path.join(cnn_results_dir, 'confusion_matrix.png'),
                    f'Convolutional Neural Network - {data_name.capitalize()} Dataset'
                )
                
                print()
    
    print("\n" + "="*60)
    print("Testing completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
