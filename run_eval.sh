#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh

python -m src.utils.evaluate --src "B21_seed101_fold0.pt" --fold 0 -fp "B21_seed101.parquet" -e "training,validation,test" -bs 30000
python -m src.utils.evaluate --src "B21_seed102_fold0.pt" --fold 0 -fp "B21_seed102.parquet" -e "training,validation,test" -bs 30000
