from torch.utils.data import TensorDataset, DataLoader, Dataset

class XAIDataSet(Dataset):

    def __init__(self, *data_tuple):
        self.dataset = TensorDataset(*data_tuple)

    def __getitem__(self, index):

        return self.dataset.__getitem__(index), index

    def __len__(self):
        return len(self.dataset)
    