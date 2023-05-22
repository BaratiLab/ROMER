#!/bin/bash
python Train.py -data 1 -ae_epochs 100 -epochs 100
python Train.py -data 2 -ae_epochs 200 -epochs 200
python Train.py -data 3 -ae_epochs 300 -epochs 300
python Test.py