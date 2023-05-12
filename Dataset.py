# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:28:17 2022

@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
from torch.utils.data import Dataset


class AE_dataset(Dataset):

    def __init__(self, data_array, idxs):

        """
        The shape of data_array can be (T, H, W) or (N, T, H, W)
        """
        super().__init__()
        self.data = data_array
        self.idxs = idxs

        if len(self.data.shape) == 3:
            # if shape is (T, H, W), make it (1, T, H, W)
            self.data = self.data.unsqueeze(0)

        # add channel dimension (N, T, C, H, W)
        self.data = self.data.unsqueeze(2)

        self.N = len(idxs)
        self.T = self.data.shape[1]

        self.hashmap = lambda i: (i//self.T, i%self.T)

    def __len__(self):
        return self.N*self.T

    def __getitem__(self, idx):
        n, t = self.hashmap(idx)
        return self.data[self.idxs[n], t]


class Dynamics_dataset(Dataset):

    def __init__(self, data_array, idxs, dt=1):

        """
        The shape of data_array can be (T, H, W) or (N, T, H, W)
        """
        super().__init__()
        self.data = data_array
        self.idxs = idxs

        if len(self.data.shape) == 3:
            # if shape is (T, H, W), make it (N, T, H, W) where N=1
            self.data = self.data.unsqueeze(0)

        # add channel dimension (N, T, C, H, W)
        self.data = self.data.unsqueeze(2)

        self.dt = dt
        self.N = len(idxs)
        self.T = self.data.shape[1]

        self.hashmap = lambda i: (i//(self.T-dt), i % (self.T-dt))

    def __len__(self):
        return self.N*(self.T-self.dt)

    def __getitem__(self, idx):
        n, t = self.hashmap(idx)
        x = self.data[self.idxs[n], t]
        y = self.data[self.idxs[n], t+self.dt]
        return x, y
