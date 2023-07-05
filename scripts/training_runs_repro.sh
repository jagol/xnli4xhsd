#!/bin/bash

json=$(cat paths.json)
results_dir=$(echo "$json" | grep -o '"output_dir": "[^"]*"' | cut -d'"' -f4)
data_dir=$(echo "$json" | grep -o '"data_dir": "[^"]*"' | cut -d'"' -f4)
results_dir="/srv/scratch1/jgoldz/xnli-for-multilingual-hate-speech-detection/results"

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name DYN21_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_DEN/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --max_length 128 --training_set ${data_dir}/processed/DYN21_EN/DYN21_EN_train_38644.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/DYN21_EN/DYN21_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  
  eval "$cmd"
done

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name KEN20_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_KEN/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --max_length 128 --training_set ${data_dir}/processed/KEN20_EN/KEN20_EN_train_*.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/KEN20_EN/KEN20_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  
  eval "$cmd"
done

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name FOU18_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_FEN/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --max_length 128 --training_set ${data_dir}/processed/FOU18_EN/FOU18_EN_train_*.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/FOU18_EN/FOU18_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  
  eval "$cmd"
done

# DYN21_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do 
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name DYN21_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_DEN/${dataset}/examples_${train_size}/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --checkpoint ${results_dir}/X_DEN/RUN${seed} --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# FOU18_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do 
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name FOU18_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_FEN/${dataset}/examples_${train_size}/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --checkpoint ${results_dir}/X_FEN/RUN${seed} --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      
      eval "$cmd"
    done
  done
done

# KEN20_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name KEN20_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_KEN/${dataset}/examples_${train_size}/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --checkpoint ${results_dir}/X_KEN/RUN${seed} --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# Direct fine-tuning on target language with M
datasets=("BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT")
model_names=('pysentimiento/robertuito-base-uncased' 'neuralmind/bert-base-portuguese-cased' 'neuralspace-reverie/indic-transformers-hi-bert' 'aubmindlab/bert-base-arabertv02' 'Musixmatch/umberto-commoncrawl-cased-v1')

IFS=$'\n'
combined=($(paste -d: <(printf "%s\n" "${datasets[@]}") <(printf "%s\n" "${model_names[@]}")))

for dataset_model in "${combined[@]}"; do
  IFS=":" read -ra arr <<< "$dataset_model"
  dataset="${arr[0]}"
  model_name="${arr[1]}"
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      echo "$cmd"
      cmd="CUDA_VISIBLE_DEVICES=7 python3 src/train.py --experiment_name MonolingTL_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/monoling_target_lang/${dataset}/examples_${train_size}/RUN${seed} --model_name ${model_name} --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# Direct fine-tuning on target language with X
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name X_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X/${dataset}/examples_${train_size}/RUN${seed} --model_name cardiffnlp/twitter-xlm-roberta-base --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 8 --gradient_accumulation 2 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done
