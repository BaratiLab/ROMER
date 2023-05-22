# ROMER
PyTorch Implementation of *"ROMER: Reduced Order Modeling of Fluid Flows with Transformers"* ([paper](https://doi.org/10.1063/5.0151515))

## Installation
Set up conda environment and clone the github repo:
```
# create a new environment
$ conda create --name ROMER python=3.7
$ conda activate ROMER

# install requirements
$ pip install -r requirements.txt

# clone the source code of ROMER
$ git clone https://github.com/BaratiLab/ROMER.git
$ cd ROMER
$ mkdir data
```
## Datasets
The datasets are available [here](https://doi.org/10.6084/m9.figshare.22806800). After download, unzip the data in the data folder.

## Training the model
Train and test the models from the paper:
```
chmod +x ./training.sh
./training.sh
```
Train and test the models in the hyperprameter study:
```
chmod +x ./parameter_search.sh
./parameter_search.sh
```
Train a model with your choice of hyperparameters:
```
python Train.py -data data -down down -embed_dim embed_dim -dt dt -num_layers num_layers -num_heads num_heads -batch batch -ae_epoch ae_epochs -epochs epochs
```
where

```data``` is the dataset number (1, 2 or 3) (default=3)

```down``` is the number of encoder/decoder blocks in the autoencoder (default=4)

```embed_dim``` is the feature dimension of the embedding tensor (default=128)

```dt``` is the number of time steps that the transformer learns to propagate the system (default=1)

```num_layers``` is the number of layers of the transformer (default=6)

```num_heads``` is the number of heads in the multi-head attention in the transformer (default=8)

```batch``` is the batch size used for training (default=64)

```ae_epoch``` is the number of epochs to train the autoencoder (default=100)

```epoch``` is the number of epochs to train the transformer (default=100)

You can find the recommended number of epochs for each dataset in ```training.sh```.

## Citation
If you find this code useful, please consider citing our work:
