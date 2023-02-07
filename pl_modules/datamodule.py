import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
class LADataset(Dataset):
    def __init__(self, path, a, n, train=True, boundary_type='D'):
        super().__init__()
        path = Path(path)
        if train:
            self.B = np.load(f'{path}/fd_B.npy')
            self.F = np.load(f'{path}/fd_F.npy')
            self.U = np.load(f'{path}/fd_X{boundary_type}.npy')
        else:
            self.B = np.load(f'{path}/fd_ValB.npy')
            self.F = np.load(f'{path}/fd_ValF.npy')
            self.U = np.load(f'{path}/fd_ValX{boundary_type}.npy')

        x = np.linspace(-a, a, n)
        y = np.linspace(-a, a, n)
        self.xx, self.yy = np.meshgrid(x, y)
        # self.xx, self.yy = torch.from_numpy(xx).float(), torch.from_numpy(yy).float()

    def __len__(self):
        return self.B.shape[0]

    def __getitem__(self, idx):
        f = self.F[idx, :]
        b = self.B[idx, :]
        u = self.U[idx, 1:-1, 1:-1]
        data = np.stack([self.xx, self.yy, f], axis=0)
        
        f = torch.from_numpy(f).float()
        b = torch.from_numpy(b).float()
        u = torch.from_numpy(u).float()
        data = torch.from_numpy(data).float()
        return  data, b, f[None, ...], u[None, ...]
    

class LADataModule(pl.LightningDataModule):
    
    def __init__(self, data_path, batch_size, a, n, boundary_type='D'):
        super().__init__()
        self.a ,self.n = a, n
        self.path = data_path
        self.type = boundary_type

        self.batch_size = batch_size

    def setup(self, stage):    
        if stage == 'fit' or stage is None:
            self.train_dataset = LADataset(self.path, self.a, self.n, True, self.type)
            self.val_dataset = LADataset(self.path, self.a, self.n, False, self.type)
        if stage == 'test':
            self.dataset = LADataset(self.path, self.a, self.n, False, self.type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=6)
        

if __name__ == '__main__':
    ds = LADataset('../data/64/mixed/', 1, 64)
    data, b, f, u = ds[0]
    print(data.to('cuda').type)
    print(b.dtype)

    
    print(data.shape)

    print(b.shape)
    print(f.shape)
    print(u.shape)
