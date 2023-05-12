import numpy as np
import matplotlib
import torch
from sklearn.model_selection import train_test_split
from utils import NS_load
from Dataset import AE_dataset, Dynamics_dataset
from Model import Encoder, Decoder, Transformer2D
from ModelClass import ModelClass
matplotlib.rcParams['figure.autolayout'] = True
import argparse


parser = argparse.ArgumentParser()
# Specifications about the dataset:
parser.add_argument('-data', '--dataset_idx', type=int, default=3,
                    choices = [0, 1, 2, 3, 4])
parser.add_argument('-N', '--N', type=int, default=1200,
                    choices=[1200, 4800, 9600])

# Specifications about the convolutional autoencoder
parser.add_argument('-down', '--down', type=int, default=4,
                    choices=[1, 2, 3, 4, 5])
parser.add_argument('-embed_dim', '--embed_dim', type=int, default=128,
                    choices=[32, 64, 128, 256, 512])

# Specifications about the transformer
parser.add_argument('-dt', '--dt', type=int, default=1,
                    choices=[1, 2, 4, 8])
parser.add_argument('-num_layers', '--num_layers', type=int, default=6)
parser.add_argument('-num_heads', '--num_heads', type=int, default=8,
                    choices=[1, 2, 4, 8])

# Specifications about training
parser.add_argument('-batch', '--batch_size', type=int, default=64,
                    choices=[32, 64, 128, 256])
parser.add_argument('-ae_epochs', '--ae_epochs', type=int, default=100)
parser.add_argument('-epochs', '--epochs', type=int, default=100)

args = parser.parse_args()

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is', Device, '\n')


dataset_idx = args.dataset_idx
N = args.N

down = args.down
embed_dim = args.embed_dim

dt = args.dt
num_layers = args.num_layers
num_heads = args.num_heads

batch_size = args.batch_size

data_names = {1: 'NS_V1e-3', 2: 'NS_V1e-4', 3: 'NS_V1e-5'}

Ts = {1: 50, 2:30, 3:20}
T = Ts[dataset_idx]

if N==4800: n=5000
elif N==9600: n=10000
data_array = NS_load(data_names[dataset_idx], n, T)[:N]

data_array /= max(data_array.max(), -data_array.min())

init_size = 64
embed_size = init_size//2**down

train_idx, val_idx = train_test_split(np.arange(N), test_size=1/6,
                                      random_state=0)

n_epochs_ae = args.ae_epochs
n_epochs_transformer = args.epochs


name = f'NS{dataset_idx}({N})_dt{dt}_down{down}_embed{embed_dim}'
name += f'_layers{num_layers}_heads{num_heads}'

hidden_dim = 2*embed_dim
channels = [embed_dim//2**i for i in range(down)]
channels.reverse()
channels = [1] + channels

print('model name: ', name)
Model = ModelClass(name)
Model.set_AE_dataset(AE_dataset(data_array, train_idx),
                     AE_dataset(data_array, val_idx))
Model.set_dynamics_dataset(Dynamics_dataset(data_array, train_idx, dt),
                           Dynamics_dataset(data_array, val_idx, dt))

encoder = Encoder(channels=channels, pool_mode='avg',
                  padding_mode='circular')
channels.reverse()
decoder = Decoder(channels=channels, up_mode='bilinear',
                  padding_mode='circular')
Model.set_AE(encoder, decoder)
print(f'Training autoencoder for {n_epochs_ae} epochs')
Model.train_AE(epochs=n_epochs_ae, batch_size=batch_size)

# set and train dynamic model
transformer = Transformer2D(shape=(embed_size, embed_size),
                            n_layers=num_layers,
                             MHA_kwargs=dict(embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             hidden_dim=hidden_dim)
                             )
Model.set_model(transformer)
print(f'Training transformer for {n_epochs_transformer} epochs')
Model.train(epochs=n_epochs_transformer, batch_size=batch_size)

print('Finished training', name, '\n')
