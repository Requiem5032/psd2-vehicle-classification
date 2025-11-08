import numpy as np
import pandas as pd
from scipy import stats, fft

SENSOR_RATE = 25
MAX_TIME = 150
TOTAL_POINTS = MAX_TIME * SENSOR_RATE
EVAL_TIME = np.linspace(0, MAX_TIME, TOTAL_POINTS)


def interp_data(original_time, original_data):
    interp_data = np.interp(EVAL_TIME, original_time, original_data)
    return interp_data
