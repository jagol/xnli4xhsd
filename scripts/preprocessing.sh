#!/bin/bash
# $1: path to data-dir
python3 src/preprocess.py -i all -d data
python3 src/split.py -d data -n BAS19_ES
python3 src/split.py -d data -n DYN21_EN
python3 src/split.py -d data -n FOR19_PT
python3 src/split.py -d data -n FOU18_EN
python3 src/split.py -d data -n HAS19_HI
python3 src/split.py -d data -n HAS20_HI
python3 src/split.py -d data -n HAS21_HI
python3 src/split.py -d data -n KEN20_EN
python3 src/split.py -d data -n OUS19_AR
python3 src/split.py -d data -n OUS19_FR
python3 src/split.py -d data -n SAN20_IT
