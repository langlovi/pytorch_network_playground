#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh
python -u -m src.train.train -tn 'A(11)_workernode'
