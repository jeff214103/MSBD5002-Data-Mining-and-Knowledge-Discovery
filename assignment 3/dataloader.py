from torch.utils.data import Dataset
## train data
class dataloader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__ (self):
        return len(self.y)