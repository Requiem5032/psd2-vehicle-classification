import pickle
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.nn import *
from src.utils import *
from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        add_help=True,
        description='Train and evaluate neural network models.',
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to the trained model.'
    )
    parser.add_argument(
        'data',
        type=str,
        help='Path to the prediction data csv.'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluate the model on the provided dataset.'
    )
    parser.print_help()
    args = parser.parse_args()

    model = ConvolutionalNeuralNetwork()
    model.eval()
    model_state = torch.load(args.model)
    model.load_state_dict(model_state)

    if args.eval:
        with open(args.data, 'rb') as f:
            vehicle_data = pickle.load(f)
            dataloader = get_dataloader(
                vehicle_data, deploy=False, shuffle=False)

        pred_label_list = []
        true_label_list = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                pred_label_list.extend(list(map(int, predicted.cpu().numpy())))
                true_label_list.extend(list(map(int, labels.cpu().numpy())))

        eval_metrics = calculate_metrics(true_label_list, pred_label_list)
        print('Results:')
        for key, value in eval_metrics.items():
            print(f'{key}: {value:.4f}')

        print('Prediction label distribution:')
        print(Counter(pred_label_list))

        print('True label distribution:')
        print(Counter(true_label_list))

        plt.figure(figsize=(10, 4))

        # Predicted labels
        plt.subplot(1, 2, 1)
        pred_counter = Counter(pred_label_list)
        plt.bar(pred_counter.keys(), pred_counter.values(), color='skyblue')
        plt.title('Predicted Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], ['Bike', 'Bus', 'Car'])

        # True labels
        plt.subplot(1, 2, 2)
        true_counter = Counter(true_label_list)
        plt.bar(true_counter.keys(), true_counter.values(), color='salmon')
        plt.title('True Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], ['Bike', 'Bus', 'Car'])

        plt.tight_layout()
        plt.show()


    # Predict single data point
    else:
        data_path = args.data
        data_df = pd.read_csv(data_path)
        processed_data = process_data(data_df)
        with torch.no_grad():
            outputs = model(processed_data)
            _, predicted = torch.max(outputs, 1)

        # Edit this part to give the output format you want (files, stdout, etc.)
        print(f'Predicted labels: {list(map(int, predicted.cpu().numpy()))}')
        print(processed_data.shape)


if __name__ == '__main__':
    main()
