#!/bin/bash

json=$(cat paths.json)
configs_dir=$(echo "$json" | grep -o '"configs_dir": "[^"]*"' | cut -d'"' -f4)
results_dir=$(echo "$json" | grep -o '"output_dir": "[^"]*"' | cut -d'"' -f4)
data_dir=$(echo "$json" | grep -o '"data_dir": "[^"]*"' | cut -d'"' -f4)

configs_dir="/home/user/jgoldz/xnli-for-multilingual-hate-speech-detection"
results_dir="/srv/scratch1/jgoldz/xnli-for-multilingual-hate-speech-detection/results"

# ---- 0 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN1/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN1/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN2/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN2/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN3/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN3/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN4/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN4/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN5/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN5/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN6/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN6/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN7/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN7/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN8/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN8/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN9/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN9/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN10/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN10/baseline/MHC_ES_eval.json" --gpu 2
# ---- 20 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN1/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN1/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN2/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN2/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN3/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN3/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN4/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN4/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN5/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN5/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN6/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN6/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN7/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN7/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN8/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN8/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN9/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN9/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN10/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN10/baseline/MHC_ES_eval.json" --gpu 2
# ---- 200 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN1/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN1/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN2/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN2/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN3/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN3/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN4/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN4/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN5/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN5/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN6/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN6/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN7/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN7/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN8/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN8/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN9/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN9/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN10/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN10/baseline/MHC_ES_eval.json" --gpu 2
# ---- 2000 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN1/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN1/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN2/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN2/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN3/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN3/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN4/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN4/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN5/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN5/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN6/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN6/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN7/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN7/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN8/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN8/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN9/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN9/baseline/MHC_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN10/baseline/BAS19_ES_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN10/baseline/MHC_ES_eval.json" --gpu 2
# ---- 0 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN1/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN1/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN2/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN2/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN3/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN3/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN4/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN4/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN5/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN5/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN6/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN6/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN7/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN7/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN8/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN8/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN9/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN9/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN10/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN10/baseline/MHC_HI_eval.json" --gpu 2
# ---- 20 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN1/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN1/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN2/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN2/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN3/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN3/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN4/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN4/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN5/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN5/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN6/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN6/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN7/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN7/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN8/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN8/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN9/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN9/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN10/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN10/baseline/MHC_HI_eval.json" --gpu 2
# ---- 200 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN1/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN1/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN2/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN2/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN3/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN3/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN4/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN4/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN5/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN5/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN6/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN6/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN7/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN7/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN8/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN8/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN9/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN9/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN10/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN10/baseline/MHC_HI_eval.json" --gpu 2
# ---- 2000 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN1/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN1/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN2/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN2/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN3/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN3/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN4/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN4/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN5/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN5/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN6/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN6/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN7/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN7/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN8/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN8/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN9/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN9/baseline/MHC_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN10/baseline/HAS21_HI_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_2000/RUN10/baseline/MHC_HI_eval.json" --gpu 2
# ---- 0 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN1/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN1/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN2/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN2/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN3/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN3/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN4/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN4/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN5/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN5/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN6/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN6/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN7/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN7/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN8/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN8/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN9/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN9/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN10/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_0/RUN10/baseline/MHC_AR_eval.json" --gpu 2
# ---- 20 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN1/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN1/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN2/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN2/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN3/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN3/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN4/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN4/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN5/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN5/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN6/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN6/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN7/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN7/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN8/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN8/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN9/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN9/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN10/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_20/RUN10/baseline/MHC_AR_eval.json" --gpu 2
# ---- 200 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN1/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN1/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN2/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN2/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN3/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN3/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN4/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN4/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN5/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN5/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN6/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN6/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN7/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN7/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN8/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN8/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN9/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN9/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN10/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_200/RUN10/baseline/MHC_AR_eval.json" --gpu 2
# ---- 2000 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN1/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN1/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN2/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN2/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN3/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN3/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN4/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN4/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN5/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN5/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN6/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN6/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN7/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN7/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN8/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN8/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN9/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN9/baseline/MHC_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN10/baseline/OUS19_AR_eval.json" --gpu 2
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/OUS19_AR/examples_2000/RUN10/baseline/MHC_AR_eval.json" --gpu 2
