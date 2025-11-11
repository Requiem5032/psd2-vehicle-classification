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


def get_dataloader(data_dict, batch_size=16, shuffle=True):
    dataset = VehicleSensorDataset(data_dict)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return dataloader
