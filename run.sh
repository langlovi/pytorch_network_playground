#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh

#python -m src.train.train -tn "B(21)_seed100" -mn "B21_seed100" --seed 100 -tw 1.0
#python -m src.train.train -tn "B(21)_seed101" -mn "B21_seed101" --seed 101 -tw 1.0
#python -m src.train.train -tn "B(21)_seed102" -mn "B21_seed102" --seed 102 -tw 1.0
#python -m src.train.train -tn "B(21)_seed103" -mn "B21_seed103" --seed 103 -tw 1.0

python -m src.train.train -tn "A(11)_seed101_batchnorm" -mn "A11_seed101_batchnorm" --seed 101 -tw 1.0


