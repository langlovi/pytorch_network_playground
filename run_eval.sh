#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh

python -m src.utils.evaluate --src "A13_seed101_confseed100_fold0.pt" --fold 0 -fp "A13_seed101_confseed100.parquet" -e "training,validation,test" -bs 30000
python -m src.utils.evaluate --src "A13_seed101_confseed102_fold0.pt" --fold 0 -fp "A13_seed101_confseed102.parquet" -e "training,validation,test" -bs 30000



