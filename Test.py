#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
matplotlib.rcParams['figure.autolayout'] = True
plt.rcParams.update({'font.size': 18})

import warnings
warnings.filterwarnings('ignore')

import torch
from sklearn.model_selection import train_test_split

from utils import NS_load, count_params
from Dataset import AE_dataset, Dynamics_dataset
from ModelClass import ModelClass


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is', Device, '\n')

data_names = {1: 'NS_V1e-3', 2: 'NS_V1e-4', 3: 'NS_V1e-5'}
Ts = {1:50, 2:30, 3:20}

NS_models = sorted([name for name in os.listdir('Results') if name.startswith('NS')])
for i, model_name in enumerate(NS_models):
    print(f'model {i}: {model_name}')
    Model = ModelClass(model_name)
    Model.load_state_dict()

    dataset_idx = int(model_name[2])
    T = Ts[dataset_idx]

    a = model_name.find('dt')
    b = model_name.find('_', a)
    dt = int(model_name[a+2:b])

    a = model_name.find('(')
    b = model_name.find(')')
    N = int(model_name[a+1:b])


    # We set the random state for reproducability
    train_idx, val_idx = train_test_split(np.arange(N), test_size=1/6,
                                          random_state=0)

    # Validate
    model = ModelClass(model_name)
    model.load_state_dict()
    model.AE_test_mode()
    model.model_test_mode()

    n1 = count_params(model.AE)
    n2 = count_params(model.model)
    print(f"parameter count: CAE({n1}), transformer({n2}), total({n1+n2})")

    try:
        RMSEs = np.load(model.result_dir+'/RMSEs.npy')
        RMSEs_t = np.load(model.result_dir+'/RMSEs_t.npy')
        TRUEs = np.load(model.result_dir+'/TRUEs.npy')
        PREDs = np.load(model.result_dir+'/PREDs.npy')
    except:
        data_array = NS_load(data_names[dataset_idx], N, T)
        data_array /= max(data_array.max(), -data_array.min())

        model.set_AE_dataset(AE_dataset(data_array, train_idx),
                             AE_dataset(data_array, val_idx))
        model.set_dynamics_dataset(Dynamics_dataset(data_array, train_idx, dt),
                                   Dynamics_dataset(data_array, val_idx, dt))
        model.load_state_dict()
        model.AE_test_mode()
        model.model_test_mode()

        RMSEs = []
        RMSEs_t = []
        TRUEs = []
        PREDs = []
        np.random.seed(0)
        samples = np.random.choice(len(val_idx), 5, False)
        for i in range(len(val_idx)):
            n_start = i
            t_start = 9
            true, pred = model.validate(n_start, t_start)
            if i in samples:
                TRUEs.append(true)
                PREDs.append(pred)
            n = len(true)

            error = torch.as_tensor(true-pred, dtype=torch.float)
            target = torch.as_tensor(true, dtype=torch.float)

            error_norm_t = torch.norm(error, 2, (1, 2))
            target_norm_t = torch.norm(target, 2, (1, 2))
            rel_error_t = error_norm_t/target_norm_t
            RMSEs_t.append(np.array(rel_error_t))

            error_norm = torch.norm(error, 2)
            target_norm = torch.norm(target, 2)
            rel_error = error_norm/target_norm
            RMSEs.append(np.array(rel_error))

    RMSEs_t = np.stack(RMSEs_t)
    RMSEs = np.stack(RMSEs)
    TRUEs = np.stack(TRUEs)
    PREDs = np.stack(PREDs)
    np.save(model.result_dir+'/RMSEs_t.npy', RMSEs_t)
    np.save(model.result_dir+'/RMSEs.npy', RMSEs)
    np.save(model.result_dir+'/TRUEs.npy', TRUEs)
    np.save(model.result_dir+'/PREDs.npy', PREDs)

    plt.figure()
    plt.title(f'Test error (total {100*RMSEs.mean():.2f}%)')
    rmse_mean = RMSEs_t.mean(0)
    rmse_std = RMSEs_t.std(0)
    x_ = range(11, len(rmse_mean)+11)
    plt.plot(x_, 100*rmse_mean, marker='*', color='blue')
    plt.fill_between(x_, 100*(rmse_mean-rmse_std), 100*(rmse_mean+rmse_std),
                     alpha=0.15, color='blue')
    plt.grid(linestyle='--')
    plt.xlabel('time step')
    plt.ylabel('relative RMSE (%)')
    plt.xticks(np.array(range(10, 11+len(rmse_mean), 2 if dataset_idx==3 else 5)))
    plt.savefig(model.result_dir+'/Error.png')
    plt.close()

    plot_freqs = {1:8, 2:4, 3:2}
    plot_freq = plot_freqs[dataset_idx]
    fig = plt.figure(figsize=(10, 10))
    ncols = (TRUEs.shape[1]-1)//plot_freq + 1
    grid = ImageGrid(fig, 111, nrows_ncols=(2*len(TRUEs), ncols))
    for i, ax in enumerate(grid):
        n = i//ncols
        nt = i % ncols
        if n == 0:
            ax.set_title(f't={10+plot_freq*(nt+1)}')
        if n % 2 == 0:
            ax.pcolormesh(TRUEs[n//2][plot_freq*(nt+1)-1], cmap='bwr', vmin=-1, vmax=1)
            if nt == 0:
                ax.set_ylabel('CFD')
        else:
            ax.pcolormesh(PREDs[n//2][plot_freq*(nt+1)-1], cmap='bwr', vmin=-1, vmax=1)
            if nt == 0:
                ax.set_ylabel('ROMER')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(model.result_dir+'/samples.png')
    plt.close()

    print(f'Error: {100*RMSEs.mean():.2f}%\n')
