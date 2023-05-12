# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import torch
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams.update({'font.size': 18})


def what_is(x):
    print('It is', type(x))
    try:
        print('It has dtype', x.dtype)
    except:
        pass
    try:
        print('It has shape', x.shape)
    except:
        pass

def NS_load(data='NS_V1e-5', N=1200, T=20):
    name = f'data/{data}_N{N}_T{T}.npy'
    a = np.load(name)
    return torch.as_tensor(a, dtype=torch.float)


# %% getting the number of parameters of a pytorch model
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# %% Visualization
def viz_video(x, cmap='bwr', figsize=(12, 4), interval=10, vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    if vmax is None:
        vmax = max(x.max(), -x.min())
    im = ax.pcolormesh(x[0], cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im)

    def animate(i):
        im.set_array(x[i])
        ax.set_xlabel(f't = {i}')
        return im
    clip = FuncAnimation(fig, animate, frames=len(x),
                         interval=interval, repeat=False)
    plt.show()
    return clip


def viz_compare(true, pred, cmap='bwr', vmax=1, interval=100, dt=1):

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    im0 = ax[0].pcolormesh(true[0], cmap=cmap, vmin=-vmax, vmax=vmax)
    im1 = ax[1].pcolormesh(pred[0], cmap=cmap, vmin=-vmax, vmax=vmax)
    im2 = ax[2].pcolormesh(pred[0]-true[0], cmap=cmap, vmin=-vmax, vmax=vmax)

    ax[0].set_title('True')
    ax[1].set_title('Pred')
    ax[2].set_title('Error')

    ax[1].set_xlabel('t=0', fontsize=20)

    for axis in ax:
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])

    def animate(i):
        im0.set_array(true[i])
        im1.set_array(pred[i])
        im2.set_array(pred[i]-true[i])
        ax[1].set_xlabel(f't={i*dt}', fontsize=20)

    anim = animation.FuncAnimation(fig, animate, frames=len(true),
                                   interval=interval, blit=False, repeat=False)
    plt.show()
    return anim


def plot_curves(model, skip_epochs=4):
    ae_train_loss = model.AE_train_loss
    ae_val_loss = model.AE_val_loss
    train_loss = model.train_loss
    val_loss = model.val_loss

    epoch_ae = len(ae_train_loss)
    epoch = len(train_loss)
    plt.figure(figsize=(16,6))

    plt.subplot(121)
    plt.plot(range(skip_epochs+1, epoch_ae+1), ae_train_loss[skip_epochs:],
             label='train loss')
    plt.plot(range(skip_epochs+1, epoch_ae+1), ae_val_loss[skip_epochs:],
             label='val loss')
    plt.legend()
    plt.xlabel('Autoencoder learning curve')

    plt.subplot(122)
    plt.plot(range(skip_epochs+1, epoch+1), train_loss[skip_epochs:],
             label='train loss')
    plt.plot(range(skip_epochs+1, epoch+1), val_loss[skip_epochs:],
             label='val loss')
    plt.legend()
    plt.xlabel('Transformer learning curve')

    plt.show()
