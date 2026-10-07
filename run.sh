#!/bin/bash
cd /afs/desy.de/user/l/langlovi/pytorch_network_playground
source setup.sh
#python -u -m src.train.train -tn 'A(14)_seed102'

echo "====================================="
echo "Starte Training 1"
echo "====================================="

python -m src.train.train -tn "A(12)_seed101" -mn "A12_seed101" --seed 101 -tw 1.5

echo "====================================="
echo "Starte Training 2"
echo "====================================="

python -m src.train.train -tn "A(12)_seed102" -mn "A12_seed102" --seed 102 -tw 1.5

echo "====================================="
echo "Starte Training 3"
echo "====================================="

python -m src.train.train -tn "A(12)_seed103" -mn "A12_seed103" --seed 103 -tw 1.5

echo "====================================="
echo "Starte Training 4"
echo "====================================="

python -m src.train.train -tn "A(13)_seed101" -mn "A13_seed101" --seed 101 -tw 2.0

echo "====================================="
echo "Starte Training 5"
echo "====================================="

python -m src.train.train -tn "A(13)_seed102" -mn "A13_seed102" --seed 102 -tw 2.0

echo "====================================="
echo "Starte Training 6"
echo "====================================="

python -m src.train.train -tn "A(13)_seed103" -mn "A13_seed103" --seed 103 -tw 2.0

echo "====================================="
echo "Starte Training 7"
echo "====================================="

python -m src.train.train -tn "A(14)_seed101" -mn "A14_seed101" --seed 101 -tw 5.0

echo "====================================="
echo "Starte Training 8"
echo "====================================="

python -m src.train.train -tn "A(14)_seed102" -mn "A14_seed102" --seed 102 -tw 5.0

echo "====================================="
echo "Starte Training 9"
echo "====================================="

python -m src.train.train -tn "A(14)_seed103" -mn "A14_seed103" --seed 103 -tw 5.0

echo "====================================="
echo "Alle Trainings beendet"
echo "====================================="
