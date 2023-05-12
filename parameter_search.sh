#!/bin/bash

python Train.py -data 3 -ae_epochs 300 -epochs 300 -embed 256
python Train.py -data 3 -ae_epochs 300 -epochs 300 -embed 64
python Train.py -data 3 -ae_epochs 300 -epochs 300 -down 3
python Train.py -data 3 -ae_epochs 300 -epochs 300 -down 2
python Train.py -data 3 -ae_epochs 300 -epochs 300 -num_heads 4
python Train.py -data 3 -ae_epochs 300 -epochs 300 -num_heads 2
python Train.py -data 3 -ae_epochs 300 -epochs 300 -num_layers 4
python Train.py -data 3 -ae_epochs 300 -epochs 300 -num_layers 8

python Test.py