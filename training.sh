#!/bin/bash

python Train.py -data 1 -ae_epochs 2 -epochs 2
python Train.py -data 2 -ae_epochs 2 -epochs 2
python Train.py -data 3 -ae_epochs 2 -epochs 2
python Test.py