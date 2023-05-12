#!/bin/bash

python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -embed 256
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -embed 64
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -down 3
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -down 2
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -num_heads 4
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -num_heads 2
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -num_layers 4
python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -num_layers 8

python AE_Transformer_NS.py -data 3 -ae_epochs 300 -epochs 300 -down 3 -embed 64

python AE_Transformer_NS_test.py