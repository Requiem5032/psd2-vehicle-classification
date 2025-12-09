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

class NeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
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


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.hidden_activation = nn.ReLU()
        self.final_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.hidden_activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.hidden_activation(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.hidden_activation(x)
        x = self.fc2(x)
        x = self.final_activation(x)
        return x


def train_nn(
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

        results_dir = f'results/{model_name}/seed_{seed}/fold_{fold_num}'
        create_dir(results_dir)

        model = copy.deepcopy(ml_model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_metric = {}
        loss_history = []
        best_loss = float('inf')

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

            epoch_metrics = calculate_metrics(true_label, pred_label)

            epoch_loss = running_loss / len(train_dataloader)
            loss_history.append(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_metric = epoch_metrics
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

        eval_metrics = calculate_metrics(true_label, pred_label)

        with open(f'{results_dir}/best_metric.yaml', 'w') as f:
            yaml.safe_dump(best_metric, f, sort_keys=False)
        with open(f'{results_dir}/eval_metrics.yaml', 'w') as f:
            yaml.safe_dump(eval_metrics, f, sort_keys=False)
        with open(f'{results_dir}/loss_history.yaml', 'w') as f:
            yaml.safe_dump(loss_history, f, sort_keys=False)

        print(
            f'Training completed for seed {seed}, fold {fold_num}', flush=True)
