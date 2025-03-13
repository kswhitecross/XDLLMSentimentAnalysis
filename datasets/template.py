from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """
    Just an example sample dataset that can be used for debugging.
    """
    def __init__(self, split='first'):
        super().__init__(self, SampleDataset)
        self.split = split
        if self.split == 'first':
            self.data = ['Hi, how are you?', 'New England is too cold']
        elif self.split == 'second':
            self.data = ['Goodbye, see you soon!', 'I love that cold New England weather!']
        else:
            raise ValueError(f"Unknown split {split}")

    def __getitem__(self, idx):
        return {"content": self.data[idx]}

    def __len__(self):
        return len(self.data)
