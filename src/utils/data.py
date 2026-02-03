import os
import pickle
import torch
import numpy as np
import pandas as pd
from scipy import stats, fft

SENSOR_RATE = 25
WINDOW_SIZE = SENSOR_RATE*2


def process_data(df, eval_time):
    time = df['time'].to_numpy()
    ax = df['ax'].to_numpy()
    ay = df['ay'].to_numpy()
    az = df['az'].to_numpy()

    ax_interp = trim_to_multiple(np.interp(eval_time, time, ax), WINDOW_SIZE)
    ay_interp = trim_to_multiple(np.interp(eval_time, time, ay), WINDOW_SIZE)
    az_interp = trim_to_multiple(np.interp(eval_time, time, az), WINDOW_SIZE)

    sensor_data = torch.stack(
        [
            torch.tensor(ax_interp, dtype=torch.float32),
            torch.tensor(ay_interp, dtype=torch.float32),
            torch.tensor(az_interp, dtype=torch.float32),
        ],
        dim=0,  # Shape: (3, time_points)
    )

    return sensor_data


def normalize_time(df):
    df['time'] = df['time'] - df['time'].min()
    return df


def create_sliding_windows(sensor_data, window_size=WINDOW_SIZE, overlap=0.5):
    """
    Create sliding windows from sensor data.
    
    Args:
        sensor_data: Tensor of shape (3, time_points) where 3 = [ax, ay, az]
        window_size: Size of each window
        overlap: Overlap ratio between windows (0 to 1)
    
    Returns:
        Tensor of shape (num_windows, 3, window_size)
    """
    step = int(window_size * (1 - overlap))
    windows = []
    
    num_time_points = sensor_data.shape[1]  # Get time dimension
    
    for i in range(0, num_time_points - window_size + 1, step):
        window = sensor_data[:, i:i+window_size]  # Shape: (3, window_size)
        windows.append(window)
    
    return torch.stack(windows, dim=0)  # Shape: (num_windows, 3, window_size)


def trim_to_multiple(arr, multiple):
    length = len(arr)
    trimmed_length = length - (length % multiple)
    return arr[:trimmed_length]


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def augment_with_noise(sensor_data, noise_level, seed=None):
    """
    Add Gaussian noise to sensor data for data augmentation.
    
    Args:
        sensor_data: Tensor of shape (num_windows, 3, window_size) or (3, window_size)
        noise_level: Standard deviation of Gaussian noise relative to signal std
        seed: Random seed for reproducibility
    
    Returns:
        Augmented tensor with same shape as input
    
    Example:
        >>> windows = torch.randn(100, 3, 50)  # 100 windows
        >>> augmented = augment_with_noise(windows, noise_level=0.05)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Calculate signal standard deviation
    signal_std = sensor_data.std()
    
    # Generate Gaussian noise
    noise = torch.randn_like(sensor_data) * signal_std * noise_level
    
    # Add noise to signal
    augmented_data = sensor_data + noise
    
    return augmented_data


def process_vehicle_dataset(dataframes, labels, output_path, label_mapping=None):
    """
    Process multiple vehicle DataFrames and create a consolidated dataset.
    
    Args:
        dataframes: List of pandas DataFrames containing sensor data (with columns: time, ax, ay, az)
        labels: List of label names corresponding to each DataFrame
        output_path: Path to save the processed dataset (pickle file)
        label_mapping: Dictionary mapping label names to integer values.
                      If None, will create mapping from labels list.
    
    Returns:
        Dictionary with 'data' and 'label' keys containing the processed dataset
    
    Example:
        >>> bike_df = pd.read_csv('data/bike.csv')
        >>> bus_df = pd.read_csv('data/bus.csv')
        >>> car_df = pd.read_csv('data/car-clear.csv')
        >>> dataframes = [bike_df, bus_df, car_df]
        >>> labels = ['bike', 'bus', 'car']
        >>> dataset = process_vehicle_dataset(dataframes, labels, 'data/vehicle_data.pkl')
    """
    if label_mapping is None:
        label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    
    all_windows = []
    all_labels = []
    
    for df, label in zip(dataframes, labels):
        # Extract time and prepare evaluation points
        df = normalize_time(df)
        time_data = df['time'].to_numpy()
        max_time = np.floor(np.max(time_data))
        total_points = int(max_time * SENSOR_RATE)
        eval_time = np.linspace(0, max_time, total_points)
        
        # Process the data
        processed_data = process_data(df, eval_time=eval_time)
        
        # Create sliding windows
        windows = create_sliding_windows(processed_data)
        
        # Create labels for these windows
        label_value = label_mapping[label]
        window_labels = torch.full((windows.shape[0],), label_value, dtype=torch.long)
        
        all_windows.append(windows)
        all_labels.append(window_labels)
        
        print(f'Processed {label}: {windows.shape[0]} windows with label value {label_value}')
    
    # Concatenate all data
    vehicle_data = torch.cat(all_windows, dim=0)
    vehicle_labels = torch.cat(all_labels, dim=0)
    
    # Create dataset dictionary
    dataset = {
        'data': vehicle_data,
        'label': vehicle_labels,
    }
    
    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f'\nDataset saved to {output_path}')
    print(f'Total samples: {vehicle_data.shape[0]}')
    print(f'Data shape: {vehicle_data.shape}')
    print(f'Label mapping: {label_mapping}')
    
    return dataset

