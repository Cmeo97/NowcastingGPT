import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np




class Collector: 

    def __init__(self):
        self.training_data = []
        self.testing_data = []
        self.validation_data = []
        self.testing_data_ext = []
    
    class CustomDataset(Dataset):
        def __init__(self, file_path):
            # Load the combined data of images and start times
            self.data = torch.tensor(np.load(file_path), dtype=torch.float32)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            # Retrieve image for the given index
            image = self.data[index]
            # Return as Tensors for the Dataloader (or in their original format)
            return image
        
    def collect_training_data(self, train_file_path, batch_size):
        train_dataset = self.CustomDataset(train_file_path)
        self.training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        length= len(train_dataset)

        return self.training_data, length
    
    def collect_testing_data(self, test_file_path, batch_size):
        test_dataset = self.CustomDataset(test_file_path)
        self.testing_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        length=len(test_dataset)
        print("Length", length)
        return self.testing_data, length
    
    def collect_validation_data(self, vali_file_path, batch_size):
        vali_dataset = self.CustomDataset(vali_file_path)
        self.validation_data = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)
        length= len(vali_dataset)
        return self.validation_data, length
    
    def collect_testing_data_ext(self, test_ext_file_path, batch_size):
        testing_dataset_ext = self.CustomDataset(test_ext_file_path)
        self.testing_data_ext = DataLoader(testing_dataset_ext, batch_size=batch_size, shuffle=False)
        length= len(testing_dataset_ext)
        return self.testing_data_ext, length

