import torch
from torch.utils.data import Dataset, DataLoader
import pynvml
import psutil

class MultiDataLoader:
    def __init__(self, data_loaders):
        self.dataloaders=data_loaders
        self.data_loaders = [
            iter(data_loader) for data_loader in data_loaders
        ]
        self.num_loaders = len(data_loaders)
        self.max_loader_length = max(len(loader) for loader in data_loaders)
        self.current_batch_idx = 0


    def __iter__(self):
        return self
    
    def __len__(self):
        return self.max_loader_length

    def get_memory_usage(self):
        mem=psutil.virtual_memory()
        print('current available memory is' +' : '+ str(round(mem.used/1024**2)) +' MIB')
        return round(mem.used/1024**2)

    def reset(self):
        # delete the current data loaders and reinitialize them
        del self.data_loaders
        self.data_loaders = [
            iter(data_loader) for data_loader in self.dataloaders
        ]
        self.current_batch_idx = 0
        self.get_memory_usage()
    
    def __next__(self):
        if self.current_batch_idx >= self.max_loader_length:
            raise StopIteration
        self.loader_idx = self.current_batch_idx % self.num_loaders
        data_loader = self.data_loaders[self.loader_idx]
        try:
            batch = next(data_loader)
            self.current_batch_idx = self.current_batch_idx + 1
            return batch
        except StopIteration:
            self.current_batch_idx = self.current_batch_idx + 1
            return None

if __name__ == "__main__":

    class SubDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create some example datasets
    data1 = [torch.tensor([1]),]
    data2 = [torch.tensor([4]), torch.tensor([5]), torch.tensor([6])]

    # Create sub datasets and corresponding data loaders
    sub_dataset1 = SubDataset(data1)
    sub_dataset2 = SubDataset(data2)

    sub_data_loader1 = DataLoader(sub_dataset1, batch_size=1, shuffle=True)
    sub_data_loader2 = DataLoader(sub_dataset2, batch_size=1, shuffle=True)

    # Create the MultiDataLoader
    multi_data_loader = MultiDataLoader([sub_data_loader1, sub_data_loader2])

    # Iterate through batches
    print(len(multi_data_loader))
    for epoch in range(2):
        for batch_idx, batch in enumerate(multi_data_loader):
            print(f"Batch {batch_idx}: {batch}")
        multi_data_loader.reset()
