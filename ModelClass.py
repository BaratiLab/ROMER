# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:41:09 2022

@author: cmu
"""
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelClass():

    def __init__(self, name='model'):

        self.result_dir = 'Results/'+name
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    # %% Setting the Training and Validation Datasets
    def set_AE_dataset(self, train_dataset, val_dataset=None):

        self.ae_train_dataset = train_dataset
        self.ae_val_dataset = val_dataset
        self.ae_val = val_dataset is not None

    def set_dynamics_dataset(self, train_dataset, val_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val = val_dataset is not None

    # %% Setting the main model, and its optimizer and learning curves
    def set_AE(self, encoder, decoder, optimizer=optim.Adam,
               opt_kwargs={}, device=Device):

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.AE = nn.Sequential(self.encoder, self.decoder)
        try:
            self.optimizer_AE = optimizer(self.AE.parameters(), **opt_kwargs)
        except:
            print('AE not trainable')
        self.AE_train_loss = []
        self.AE_val_loss = []

    def set_model(self, model, optimizer=optim.Adam, opt_kwargs={},
                  device=Device):

        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters(), **opt_kwargs)
        self.train_loss = []
        self.val_loss = []

    # %% Saving and Loading the state_dict
    def save_state_dict(self, attrs=['encoder', 'decoder',
                                     'AE_train_loss', 'AE_val_loss',
                                     'model',
                                     'train_loss', 'val_loss'],
                        path=None):

        if path is None:
            path = self.result_dir+'/state_dict.pickle'
        state_dict = {attr: getattr(self, attr) for attr in attrs if attr in vars(self)}
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self, attrs=None, model_name=None):

        if model_name is None:
            path = self.result_dir + '/state_dict.pickle'
        else:
            path = 'Results/' + model_name + '/state_dict.pickle'
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        attrs = state_dict.keys() if attrs is None else attrs
        for key in attrs:
            setattr(self, key, state_dict[key])

# %% AutoEncoder
    def train_AE(self, criterion=nn.MSELoss(), epochs=50,
                 batch_size=64, device=Device):

        train_loader = DataLoader(self.ae_train_dataset,
                                   batch_size=batch_size, shuffle=True)
        if self.ae_val:
            val_loader = DataLoader(self.ae_val_dataset,
                                    batch_size=batch_size, shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_AE,
                                                         factor=0.2,
                                                         patience=5)
        self.AE.to(device)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            ###################################################################
            # Training:
            losses = []
            self.AE.requires_grad_().train()
            for x in tqdm(train_loader, position=0, leave=True):
                x = x.to(device)
                self.optimizer_AE.zero_grad()
                x_rec = self.AE(x)
                loss = criterion(x_rec, x)
                loss.backward()
                self.optimizer_AE.step()
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x, x_rec
            train_loss = np.mean(losses)
            self.AE_train_loss.append(train_loss)
            scheduler.step(train_loss)
            ###################################################################
            # Validation:
            if self.ae_val:
                self.AE.requires_grad_(False).eval()
                losses = []
                for x in tqdm(val_loader, position=0, leave=True):
                    x = x.to(device)
                    x_rec = self.AE(x)
                    loss = criterion(x_rec, x)
                    losses.append(loss.item())
                    torch.cuda.empty_cache()
                    del x, x_rec
                val_loss = np.mean(losses)
                self.AE_val_loss.append(val_loss)
            ###################################################################
            # Printing the results:
            print(f'Train Loss: {train_loss:.6f}')
            if self.ae_val:
                print(f'  Val Loss: {val_loss:.6f}')
            print('~'*60)

            if train_loss == min(self.AE_train_loss):
                self.save_state_dict()

    def AE_test_mode(self, device=Device):
        self.encoder.requires_grad_(False).eval().to(device)
        self.decoder.requires_grad_(False).eval().to(device)
        self.AE = nn.Sequential(self.encoder, self.decoder).eval()

    def AE_test_single(self, x, device=Device):
        return self.AE(x.to(device).unsqueeze(0)).squeeze(0)

    # %% The main model
    def train(self, criterion=nn.MSELoss(), epochs=50,
              batch_size=64, device=Device):

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True)
        if self.val:
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                    shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                         factor=0.2,
                                                         patience=5)
        self.model.to(device)
        self.AE_test_mode(device)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            ###################################################################
            # Training:
            losses = []
            self.model.requires_grad_().train()
            for x1, x2 in tqdm(train_loader, position=0, leave=True):
                x1, x2 = x1.to(device), x2.to(device)
                x1, x2 = self.encoder(x1), self.encoder(x2)
                self.optimizer.zero_grad()
                x2_pred = self.model(x1)
                loss = criterion(x2_pred, x2)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x1, x2, x2_pred
            train_loss = np.mean(losses)
            self.train_loss.append(train_loss)
            scheduler.step(train_loss)
            ###################################################################
            # Validation:
            if self.val:
                self.model.requires_grad_(False).eval()
                losses = []
                for x1, x2 in tqdm(val_loader, position=0, leave=True):
                    x1, x2 = x1.to(device), x2.to(device)
                    x1, x2 = self.encoder(x1), self.encoder(x2)
                    x2_pred = self.model(x1)
                    loss = criterion(x2_pred, x2)
                    losses.append(loss.item())
                    torch.cuda.empty_cache()
                    del x1, x2, x2_pred
                val_loss = np.mean(losses)
                self.val_loss.append(val_loss)
            ###################################################################
            # Printing the results:
            print(f'Train Loss: {train_loss:.6f}')
            if self.val:
                print(f'  Val Loss: {val_loss:.6f}')
            print('~'*60)
            if train_loss == min(self.train_loss):
                self.save_state_dict()

    def model_test_mode(self, device=Device):
        self.model.requires_grad_(False).eval().to(Device)

    def model_test_single(self, x, device=Device):
        encoded = self.encoder(x.to(device).unsqueeze(0))
        encoded_next = self.model(encoded)
        decoded_next = self.decoder(encoded_next).squeeze(0)
        return decoded_next

    # %% Using the model Autoregressively
    def check_training(self, n_start=0, t_start=0, t_end=None, device=Device):
        true = []
        pred = []
        T = self.train_dataset.T
        dt = self.train_dataset.dt
        if t_end is None:
            t_end = T
        for i in range(t_start, t_end, dt):
            i1 = n_start*(T-dt) + i
            i2 = n_start*T + i
            if i == t_start:
                x1 = self.train_dataset[i1][0].to(device).unsqueeze(0)
                x1_encoded = self.encoder(x1)
            else:
                true.append(self.ae_train_dataset[i2].squeeze().cpu().numpy())
                x1_encoded = x2_encoded
            x2_encoded = self.model(x1_encoded)
            x2_decoded = self.decoder(x2_encoded)
            pred.append(x2_decoded.squeeze().cpu().numpy())
        true, pred = np.stack(true), np.stack(pred)[:-1]
        return true, pred

    def validate(self, n_start=0, t_start=0, t_end=None, device=Device):
        true = []
        pred = []
        T = self.val_dataset.T
        dt = self.val_dataset.dt
        if t_end is None:
            t_end = T
        for i in range(t_start, t_end, dt):
            i1 = n_start*(T-dt) + i
            i2 = n_start*T + i
            if i == t_start:
                x1 = self.val_dataset[i1][0].to(device).unsqueeze(0)
                x1_encoded = self.encoder(x1)
            else:
                true.append(self.ae_val_dataset[i2].squeeze().cpu().numpy())
                x1_encoded = x2_encoded
            x2_encoded = self.model(x1_encoded)
            x2_decoded = self.decoder(x2_encoded)
            pred.append(x2_decoded.squeeze().cpu().numpy())
        true, pred = np.stack(true), np.stack(pred)[:-1]
        return true, pred
