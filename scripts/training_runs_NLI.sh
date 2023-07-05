#!/bin/bash

json=$(cat paths.json)
results_dir=$(echo "$json" | grep -o '"output_dir": "[^"]*"' | cut -d'"' -f4)
data_dir=$(echo "$json" | grep -o '"data_dir": "[^"]*"' | cut -d'"' -f4)

results_dir="/srv/scratch1/jgoldz/xnli-for-multilingual-hate-speech-detection/results"

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_DYN21_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_DEN/RUN${seed} --model_name morit/xlm-t-full-xnli --max_length 150 --nli --training_set ${data_dir}/processed/DYN21_EN/DYN21_EN_train_38644.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/DYN21_EN/DYN21_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  
  eval "$cmd"
done

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_KEN20_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_KEN/RUN${seed} --model_name morit/xlm-t-full-xnli --max_length 150 --nli --training_set ${data_dir}/processed/KEN20_EN/KEN20_EN_train_*.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/KEN20_EN/KEN20_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  eval "$cmd"
done

for seed in {1..10}; do
  cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_FOU18_EN --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_FEN/RUN${seed} --model_name morit/xlm-t-full-xnli --max_length 150 --nli --training_set ${data_dir}/processed/FOU18_EN/FOU18_EN_train_*.jsonl --limit_training_set 20000 --validation_set ${data_dir}/processed/FOU18_EN/FOU18_EN_dev_500.jsonl --epochs 3 --batch_size 16 --gradient_accumulation 1 --log_interval 100 --seed ${seed}"
  eval "$cmd"
done

# DYN21_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_DYN21_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_DEN/${dataset}/examples_${train_size}/RUN${seed} --model_name morit/xlm-t-full-xnli --checkpoint ${results_dir}/X_NLI_DEN/RUN${seed} --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# KEN20_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  if [ "$dataset" != "HAS21_HI" ]; then
    continue
  fi
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_KEN20_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_KEN/${dataset}/examples_${train_size}/RUN${seed} --model_name morit/xlm-t-full-xnli --checkpoint ${results_dir}/X_NLI_KEN/RUN${seed} --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# FOU18_EN fine-tuning on target language
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  if [ "$dataset" != "HAS21_HI" ]; then
    continue
  fi
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_FOU18_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI_FEN/${dataset}/examples_${train_size}/RUN${seed} --model_name morit/xlm-t-full-xnli --checkpoint ${results_dir}/X_NLI_FEN/RUN${seed} --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done

# Fine-tuning on target language without English fine-tuning
for dataset in "BAS19_ES" "FOR19_PT" "HAS21_HI" "OUS19_AR" "SAN20_IT"; do
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name NLI_EN_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_NLI/${dataset}/examples_${train_size}/RUN${seed} --model_name morit/xlm-t-full-xnli --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
      eval "$cmd"
    done
  done
done


# Monolingual NLI with xlm-t: Spanish
dataset="BAS19_ES"
model_name="morit/spanish_xlm_xnli"
for train_size in 20 200 2000; do
  for seed in {1..10}; do
    cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name mNLI_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_mNLI/${dataset}/examples_${train_size}/RUN${seed} --model_name $model_name --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
    eval "$cmd"
  done
done


# Monolingual NLI with xlm-t: Hindi
dataset="HAS21_HI"
model_name="morit/hindi_xlm_xnli"
for train_size in 20 200 2000; do
  for seed in {1..10}; do
    cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name mNLI_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_mNLI/${dataset}/examples_${train_size}/RUN${seed} --model_name $model_name --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
    eval "$cmd"
  done
done

# Monolingual NLI with xlm-t: Hindi
dataset="OUS19_AR"
model_name="morit/arabic_xlm_xnli"
for train_size in 20 200 2000; do
  for seed in {1..10}; do
    cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name mNLI_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/X_mNLI/${dataset}/examples_${train_size}/RUN${seed} --model_name $model_name --max_length 150 --nli --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed}"
    eval "$cmd"
  done
done

# Fine-tuning on target language with M after training on mNLI
datasets=("BAS19_ES" "HAS21_HI" "OUS19_AR")
model_names=('pysentimiento/robertuito-base-uncased' 'neuralspace-reverie/indic-transformers-hi-bert' 'aubmindlab/bert-base-arabertv02')
checkpoints=('robertuito_xnli' 'hindi_bert_xnli' 'arabic_bert_xnli')

IFS=$'\n'
combined=($(paste -d: <(printf "%s\n" "${datasets[@]}") <(printf "%s\n" "${model_names[@]}") <(printf "%s\n" "${checkpoints[@]}")))

for dataset_model_checkpoint in "${combined[@]}"; do
  IFS=":" read -ra arr <<< "$dataset_model_checkpoint"
  dataset="${arr[0]}"
  model_name="${arr[1]}"
  checkpoint="${arr[2]}"
  for train_size in 20 200 2000; do
    for seed in {1..10}; do
      cmd="CUDA_VISIBLE_DEVICES=0 python3 src/train.py --experiment_name M_mNLI_${dataset} --run_name RUN${seed} --path_out_dir ${results_dir}/M_mNLI/${dataset}/examples_${train_size}/RUN${seed} --model_name ${model_name} --checkpoint ${results_dir}/M_mNLI/${checkpoint} --max_length 128 --training_set ${data_dir}/processed/${dataset}/${dataset}_train_*.jsonl --limit_training_set ${train_size} --validation_set ${data_dir}/processed/${dataset}/${dataset}_dev_*.jsonl --epochs 5 --batch_size 16 --gradient_accumulation 1 --log_interval 10 --seed ${seed} --nli"
      echo "$cmd"
      eval "$cmd"
    done
  done
done
