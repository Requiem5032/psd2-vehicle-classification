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

    ax_interp = np.interp(eval_time, time, ax)
    ay_interp = np.interp(eval_time, time, ay)
    az_interp = np.interp(eval_time, time, az)
    processed_data = torch.stack(
        [
            torch.tensor(ax_interp, dtype=torch.float32),
            torch.tensor(ay_interp, dtype=torch.float32),
            torch.tensor(az_interp, dtype=torch.float32),
        ],
        dim=1,
    )

    reshaped_data = torch.reshape(processed_data, (-1, WINDOW_SIZE, 3))
    return reshaped_data
