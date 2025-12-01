import yaml
import copy
import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.utils import *
from src.nn import *
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class NeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(NeuralNetwork, self).__init__()
        self.hidden_activation = nn.ReLU()
        self.final_activation = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(input_channels*WINDOW_SIZE, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.hidden_activation(x)
        x = self.fc2(x)
        x = self.final_activation(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(LogisticRegression, self).__init__()
        self.final_activation = nn.Softmax(dim=-1)
        self.linear = nn.Linear(input_channels*WINDOW_SIZE, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.final_activation(x)
        return x


def train_model(
        model_name,
        ml_model,
        criterion,
        train_dataloader,
        test_dataloader,
        learning_rate,
        num_epochs,
        seed,
        fold_num,
):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = copy.deepcopy(ml_model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_metric = {}
        loss_history = []
        best_loss = float('inf')
        results_dir = f'results/{model_name}/seed_{seed}/fold_{fold_num}'
        create_dir(results_dir)

        for _ in range(num_epochs):
            model.train()
            true_label = []
            pred_label = []
            running_loss = 0.0

            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)

                true_label.extend(labels.cpu().numpy())
                pred_label.extend(preds.cpu().numpy())

            epoch_accuracy = accuracy_score(
                true_label,
                pred_label,
            )
            epoch_precision = precision_score(
                true_label,
                pred_label,
                average='macro',
                zero_division=0,
            )
            epoch_recall = recall_score(
                true_label,
                pred_label,
                average='macro',
                zero_division=0,
            )
            epoch_f1 = f1_score(
                true_label,
                pred_label,
                average='macro',
                zero_division=0,
            )

            epoch_loss = running_loss / len(train_dataloader)
            loss_history.append(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_metric = {
                    'accuracy': epoch_accuracy,
                    'precision': epoch_precision,
                    'recall': epoch_recall,
                    'f1_score': epoch_f1,
                }
                torch.save(
                    model.state_dict(),
                    f'{results_dir}/best_model.pth',
                )

        model.load_state_dict(
            torch.load(f'{results_dir}/best_model.pth'))
        model.eval()
        eval_metrics = {}
        true_label = []
        pred_label = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                true_label.extend(labels.cpu().numpy())
                pred_label.extend(preds.cpu().numpy())

        test_accuracy = accuracy_score(
            true_label,
            pred_label,
        )
        test_precision = precision_score(
            true_label,
            pred_label,
            average='macro',
            zero_division=0,
        )
        test_recall = recall_score(
            true_label,
            pred_label,
            average='macro',
            zero_division=0,
        )
        test_f1 = f1_score(
            true_label,
            pred_label,
            average='macro',
            zero_division=0,
        )
        eval_metrics = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
        }

        with open(f'{results_dir}/best_metric.yaml', 'w') as f:
            yaml.safe_dump(best_metric, f, sort_keys=False)
        with open(f'{results_dir}/eval_metrics.yaml', 'w') as f:
            yaml.safe_dump(eval_metrics, f, sort_keys=False)
        with open(f'{results_dir}/loss_history.yaml', 'w') as f:
            yaml.safe_dump(loss_history, f, sort_keys=False)

        print(
            f'Training completed for seed {seed}, fold {fold_num}', flush=True)
