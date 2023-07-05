#!/bin/bash

json=$(cat paths.json)
configs_dir=$(echo "$json" | grep -o '"configs_dir": "[^"]*"' | cut -d'"' -f4)
results_dir=$(echo "$json" | grep -o '"output_dir": "[^"]*"' | cut -d'"' -f4)
data_dir=$(echo "$json" | grep -o '"data_dir": "[^"]*"' | cut -d'"' -f4)

configs_dir="/home/user/jgoldz/xnli-for-multilingual-hate-speech-detection"
results_dir="/srv/scratch1/jgoldz/xnli-for-multilingual-hate-speech-detection/results"

# ---- 0 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN1/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN1/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN2/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN2/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN3/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN3/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN4/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN4/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN5/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN5/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN6/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN6/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN7/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN7/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN8/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN8/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN9/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN9/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN10/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_0/RUN10/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
# ---- 20 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN1/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN1/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN2/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN2/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN3/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN3/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN4/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN4/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN5/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN5/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN6/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN6/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN7/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN7/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN8/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN8/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN9/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN9/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN10/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_20/RUN10/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
# ---- 200 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN1/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN1/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN2/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN2/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN3/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN3/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN4/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN4/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN5/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN5/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN6/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN6/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN7/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN7/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN8/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN8/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN9/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN9/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN10/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_200/RUN10/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
# ---- 2000 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN1/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN1/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN2/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN2/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN3/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN3/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN4/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN4/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN5/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN5/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN6/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN6/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN7/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN7/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN8/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN8/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN9/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN9/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN10/FBT_tc_FC_FRS/BAS19_ES_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/BAS19_ES/examples_2000/RUN10/FBT_tc_FC_FRS/MHC_ES_eval.json" --gpu 1
# ---- 0 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN1/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN1/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN2/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN2/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN3/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN3/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN4/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN4/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN5/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN5/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN6/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN6/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN7/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN7/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN8/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN8/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN9/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN9/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN10/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_0/RUN10/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
# ---- 20 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN1/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN1/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN2/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN2/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN3/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN3/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN4/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN4/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN5/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN5/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN6/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN6/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN7/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN7/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN8/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN8/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN9/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN9/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN10/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_20/RUN10/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
# ---- 200 ----
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN1/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN1/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN2/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN2/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN3/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN3/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN4/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN4/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN5/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN5/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN6/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN6/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN7/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN7/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN8/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN8/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN9/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN9/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN10/FBT_tc_FC_FRS/HAS21_HI_eval.json" --gpu 1
python3 src/evaluation.py --path_config "${configs_dir}/configs/M_mNLI/HAS21_HI/examples_200/RUN10/FBT_tc_FC_FRS/MHC_HI_eval.json" --gpu 1
