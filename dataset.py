import torch
from torch.utils.data import Dataset
from utils.checkdim import checkdim
import os

class InnoColorDataset(Dataset):
    def __init__(self, name, dir):
        """
        ALL DATA SHOULD BE TENSOR CHW
        """
        self.name = name
        self.x_list = []
        self.y_list = []
        self.m_info = []
        self.dir = dir
        self.carousel = None
    
    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index], self.m_info[index]
    
    def __len__(self):
        return len(self.m_info)
    
    def append(self, x, y, m_info):
        checkdim(x, 'CHW')
        checkdim(y, 'CHW')
        x = x.cpu()
        y = y.cpu()
        self.x_list.append(x)
        self.y_list.append(y)
        self.m_info.append(m_info)
    
    def update(self, index, x, y, m_info):
        if index < 0 or index >= len(self):
            print(f"[ERROR] Index {index} out of bounds for dataset {self.name}.")
            return
        self.x_list[index] = x.cpu()
        self.y_list[index] = y.cpu()
        self.m_info[index] = m_info
        print(f"[EVENT] Updated image at index {index} in dataset {self.name}.")

    def save(self):
        datadict = {
            "x": torch.stack([self.x_list[i] for i in range(len(self))]),
            "y": torch.stack([self.y_list[i] for i in range(len(self))]),
            "m_info": [self.m_info[i] for i in range(len(self))]
        }
        torch.save(datadict, self.dir)
        print(f"[EVENT] Dataset {self.name} saved to {self.dir}")
    
    def delete(self, index):
        if index < 0 or index >= len(self):
            print(f"[ERROR] Index {index} out of bounds for dataset {self.name}.")
            return
        del self.x_list[index]
        del self.y_list[index]
        del self.m_info[index]
        # Make sure carousel active index is updated
        if self.carousel and self.carousel.active_index >= index:
            self.carousel.active_index = max(0, self.carousel.active_index - 1)
        print(f"[EVENT] Deleted image at index {index} from dataset {self.name}.")

    def load(self):
        if not os.path.exists(self.dir):
            print(f"[ERROR] Dataset file {self.dir} does not exist.")
            return
        # Load the saved dataset
        loaded_data = torch.load(self.dir, weights_only=True)

        # Convert tensors back to lists for appending
        self.x_list = list(loaded_data["x"]) if isinstance(loaded_data["x"], torch.Tensor) else loaded_data["x"]
        self.y_list = list(loaded_data["y"]) if isinstance(loaded_data["y"], torch.Tensor) else loaded_data["y"]
        self.m_info = loaded_data["m_info"]  # Already a list of strings

        print(f"[EVENT] Dataset {self.name} loaded from {self.dir}")
