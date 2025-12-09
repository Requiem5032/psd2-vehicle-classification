import os
import torch
import numpy as np
import pandas as pd
from scipy import stats, fft

SENSOR_RATE = 25
WINDOW_SIZE = SENSOR_RATE*2
MAX_TIME = 150
TOTAL_POINTS = MAX_TIME * SENSOR_RATE
EVAL_TIME = np.linspace(0, MAX_TIME, TOTAL_POINTS)


def process_data(df, eval_time=EVAL_TIME):
    time = df['time'].to_numpy()
    ax = df['ax'].to_numpy()
    ay = df['ay'].to_numpy()
    az = df['az'].to_numpy()

    ax_interp = trim_to_multiple(np.interp(eval_time, time, ax), WINDOW_SIZE)
    ay_interp = trim_to_multiple(np.interp(eval_time, time, ay), WINDOW_SIZE)
    az_interp = trim_to_multiple(np.interp(eval_time, time, az), WINDOW_SIZE)
    processed_data = torch.stack(
        [
            torch.tensor(ax_interp, dtype=torch.float32),
            torch.tensor(ay_interp, dtype=torch.float32),
            torch.tensor(az_interp, dtype=torch.float32),
        ],
        dim=1,
    )

    reshaped_data = torch.reshape(
        processed_data, (-1, WINDOW_SIZE, 3)).transpose(1, 2).contiguous()
    return reshaped_data


def trim_to_multiple(arr, multiple):
    length = len(arr)
    trimmed_length = length - (length % multiple)
    return arr[:trimmed_length]


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
