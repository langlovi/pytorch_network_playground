#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh
#python -u -m src.train.train -tn 'A(14)_seed102'

python -m src.train.train -tn "A(13)_seed101_confseed102" -mn "A13_seed101_confseed102" --seed 101 -tw 2.0
