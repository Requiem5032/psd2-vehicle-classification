from torch.utils.data import DataLoader, Dataset


class VehicleSensorDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['data'])

    def __getitem__(self, idx):
        features = self.data_dict['data'][idx]
        label = self.data_dict['label'][idx]
        return features, label


class VehicleSensorDeployDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        features = self.data_tensor[idx]
        return features


def get_dataloader(data_dict, batch_size=16, deploy=False, shuffle=True):
    if deploy:
        dataset = VehicleSensorDeployDataset(data_dict)
    else:
        dataset = VehicleSensorDataset(data_dict)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return dataloader
