# Evaluating the Effectiveness of Natural Language Inference for Hate Speech Detection in Languages with Limited Labeled Data

This code accompanies the paper: [Evaluating the Effectiveness of Natural Language Inference for Hate Speech Detection in Languages with Limited Labeled Data](https://arxiv.org/abs/2306.03722).

## Setup

Create a file `paths.json` in the repository's root directory and write to it:
```json
{
  "data_dir": "path/to/the/data/directory",
  "output_dir": "path/to/the/output/directory",
  "configs_dir": "path/to/the/configs/directory"
}
```
Checkpoints, logs and results will be written to the output directory.

As an example:
- the `data_dir` could be `/home/user/projects/xnli-for-hate-speech-detection/data/` (in the following sections referenced as `<path-to-data-dir>`)
- `output_dir` could be `/home/user/projects/xnli-for-hate-speech-detection/output/` (in the following sections referenced as `<path-to-output-dir>`)
- and `configs_dir` could be `configs/` (in the following sections referenced as `<path-to-configs-dir>`).

Download all datasets:
```bash
bash scripts/download_data.sh
```

Create a python environment and install the required packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Preprocess the datasets:
```bash
bash scripts/preprocessing.sh
```

## Run Experiments

To reproduce the results from [Röttger et al. (2022), Data-Efficient Strategies for Expanding Hate Speech Detection into Under-Resourced Languages](https://aclanthology.org/2022.emnlp-main.383.pdf):

```bash
bash scripts/training_runs_repro.sh
bash scripts/run_eval_repro.sh
```

The baselines M and X are evaluated separately:
```bash
bash scripts/run_eval_M.sh
bash scripts/run_eval_X.sh
```

Fine-tune monolingual and multilingual models, which have been trained on (X)NLI:
```bash
bash scripts/training_runs_NLI.sh
```

Evaluate monolingual models trained on NLI:
```bash
bash scripts/run_eval_M_NLI.sh
```

Evaluate XLM-T models trained on NLI:
```bash
bash scripts/run_eval_X_NLI_baseline.sh
bash scripts/run_eval_X_NLI_strategies.sh
```

Evaluate models based on XLM-T and trained on XNLI:
```bash
bash scripts/run_eval_X_XNLI_baseline.sh
bash scripts/run_eval_X_XNLI_strategies.sh
```

Collect the results and write to one csv-file:
```bash
python3 src/parse_results_into_csv.py -i <path-to-output-dir> -o <path-to-csv-file>
```

To generate the plots execute the notebook `notebooks/plot_results.ipynb`.

Training runs are based on configs. To regenerate the configs execute the notebook `notebooks/generate_configs.ipynb`.