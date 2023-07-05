import argparse
import json
import logging
import os
import re
import random
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import sklearn
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, AutoTokenizer,
)
import wandb

from dataset import Dataset, DefaultDataset
from get_loggers import get_logger
import delete_checkpoints


# transformers.logging.set_verbosity_info()
train_logger = None
NLI: bool = False
eval_set: Optional[Dataset] = None
num_comp_metrics_out = 0
main_args: Optional[argparse.Namespace] = None


def get_model_checkpoint_path(model_checkpoint: Optional[str]) -> str:
    """If path is not a model checkpoint, search for a checkpoint in the given directory."""
    if model_checkpoint:
        if not re.fullmatch(delete_checkpoints.CHECKPOINT_REGEX, model_checkpoint.split('/')[-1]):
            train_logger.info(f'"{model_checkpoint}" is not a checkpoint. '
                             f'Search for checkpoint in child directories.')
            matches = [1 for name in os.listdir(model_checkpoint)
                       if re.fullmatch(delete_checkpoints.CHECKPOINT_REGEX, name) is not None]
            if sum(matches) > 1:
                raise Exception('Multiple checkpoints found.')
            elif sum(matches) < 1:
                raise Exception('No checkpoint found.')
            for name in os.listdir(model_checkpoint):
                if re.fullmatch(delete_checkpoints.CHECKPOINT_REGEX, name):
                    model_checkpoint = os.path.join(model_checkpoint, name)
                    train_logger.info(f'Found checkpoint "{name}", set checkpoint path to: {model_checkpoint}')
    return model_checkpoint


def filter_inputs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask'],
        'labels': batch['labels']
    }


def map_ternary_labels_to_binary(ternary_labels) -> List[int]:
    """Assumes that 0 references same class in ternary and binary.

    If a label is already binary, it is left unchanged.
    """
    out_labels = []
    for lbl in ternary_labels:
        if lbl == 0:
            out_labels.append(0)
        else:
            out_labels.append(1)
    return out_labels


def train(train_set: Dataset, dev_set: Dataset, model: AutoModelForSequenceClassification,
          tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    # for batch in train_set:
    #     break
    # print({k: v.shape for k, v in batch.items()})
    train_logger.info('Instantiate training args.')
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.path_out_dir,
        logging_dir=args.path_out_dir,
        logging_steps=args.log_interval,
        save_strategy=args.save_strategy,
        # debugging settings:
        # save_strategy='steps',
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        report_to="wandb" if args.wandb else None,
        no_cuda=args.no_cuda,
    )
    train_logger.info('Instantiate trainer.')
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_metrics=compute_metrics
    )
    train_logger.info('Start training.')
    trainer.train()
    train_logger.info('Finished training.')


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred

    if NLI:
        entail_contradiction_logits = torch.FloatTensor(logits[:, [0, 2]])
        predictions = [int(round(lb[1].item())) for lb in entail_contradiction_logits.softmax(dim=1)]
        labels_bin = map_ternary_labels_to_binary(labels.tolist())
    else:
        predictions = [int(p) for p in logits.argmax(axis=1)]
        labels_bin = labels.tolist()

    with open(os.path.join(main_args.path_out_dir, f'comp_metrics_results_{num_comp_metrics_out}.json'), 'w') as fout:
        json.dump({'labels_bin': labels_bin, 'predictions': predictions}, fout)

    return {
        'f1-macro': sklearn.metrics.f1_score(labels_bin, predictions, average='macro'),
        'accuracy': sklearn.metrics.accuracy_score(labels_bin, predictions),
        'precision': sklearn.metrics.precision_score(labels_bin, predictions),
        'recall': sklearn.metrics.recall_score(labels_bin, predictions),
        # 'roc_aux_score': sklearn.metrics.roc_auc_score(labels_bin, predictions),
    }


def ensure_valid_encoding(enc_dataset: Dataset) -> None:
    # test correct train-set encoding
    for item in enc_dataset:
        for key, val in item.items():
            if not isinstance(val, torch.Tensor):
                import pdb; pdb.set_trace()


def main(args: argparse.Namespace) -> None:
    global train_logger
    global NLI
    global eval_set
    global main_args
    main_args = args

    if args.nli:
        NLI = True
    if args.wandb:
        wandb.init(project='XNLI4XHS', name=args.run_name, tags=[args.experiment_name])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    path = '/'.join(args.path_out_dir.split('/'))
    if not os.path.exists(path):
        print(f'Output path "{path}" does not exist. Trying to create folder.')
        try:
            os.makedirs(path)
            print(f'Folder "{path}" created successfully.')
        except OSError:
            raise Exception(f'Creation of directory "{path}" failed.')
    if len(os.listdir(path)) > 0:
        raise Exception(f"Output directory '{path}' is not empty.")

    train_logger = get_logger('train')
    train_logger.info(f"CMD args: {json.dumps(args.__dict__, indent=4)}")
    train_logger.info(f'Load tokenizer: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    train_logger.info(f'Load trainset from: {args.training_set}')
    train_set = DefaultDataset(path=args.training_set, name='trainset')
    train_set.load(load_limit=args.limit_training_set, random_seed=args.seed)
    if args.nli:
        train_set.to_nli(args.hypothesis)
        train_set.nli_sanity_check()
    train_set.encode_dataset(tokenizer=tokenizer)
    ensure_valid_encoding(train_set)

    train_logger.info(f'Load trainset from: {args.validation_set}')
    eval_set = DefaultDataset(path=args.validation_set, name='eval_set')
    eval_set.load(load_limit=args.limit_validation_set)
    if args.nli:
        eval_set.to_nli(args.hypothesis)
        eval_set.nli_sanity_check()
    eval_set.encode_dataset(tokenizer=tokenizer)
    if args.checkpoint:
        model_to_load = get_model_checkpoint_path(args.checkpoint)
    else:
        model_to_load = args.model_name

    train_logger.info(f'Load Model from: {model_to_load}')

    if args.nli:
        train_logger.info(f'Set output layer to dimensionality to {3}')
        model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=3)
    else:
        train_logger.info(f'Set output layer to dimensionality: {2}')
        model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=2)
    train(train_set, eval_set, model, tokenizer, args)
    delete_checkpoints.main(argparse.Namespace(**{'path_out_dir': args.path_out_dir}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument('--experiment_name', required=False, help='Set experiment name. Entered as tag for wandb.')
    parser.add_argument('--run_name', help='Set run name for wandb.')
    parser.add_argument('-o', '--path_out_dir', help='Path to output directory.')
    parser.add_argument('-m', '--model_name',
                        help='Name of model to load. If checkpoint is given this option is still necessary in order to '
                             'load the correct tokenizer.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('-c', '--checkpoint', required=False,
                        help='Optional: provide checkpoint.')
    parser.add_argument('--no_cuda', action='store_true', help='Tell Trainer to not use cuda.')

    # task formulation
    parser.add_argument('-n', '--nli', action='store_true', help='If NLI formulation or not.')
    parser.add_argument('--hypothesis', default='This text is hate speech.')

    # data
    parser.add_argument('-t', '--training_set', help='Path to training data file.')
    parser.add_argument('-L', '--limit_training_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of randomly selected examples.')
    parser.add_argument('-v', '--validation_set', help='Path to development set data file.')
    parser.add_argument('--limit_validation_set', type=int, default=None,
                        help='Only encode and use <limit_dataset> number of randomly selected examples.')

    # hyperparams
    parser.add_argument('-E', '--epochs', type=float, default=5.0,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch-size to be used. Can only be set for training, '
                             'not for inference.')
    parser.add_argument('-a', '--gradient_accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps to perform. The effective '
                             'batch size is batch-size times gradient accumulation steps.')
    parser.add_argument('-l', '--learning_rate', type=float, default=3e-5,
                        help='Learning rate.')
    parser.add_argument('-w', '--warmup_steps', type=int, default=0,
                        help='Number of warmup steps.')

    # reporting and debugging
    parser.add_argument('-A', '--add_info', action='store_true', help='Load additional info into training loop.')
    parser.add_argument('-i', '--log_interval', type=int, default=1,
                        help='Interval batches for which the loss is reported.')

    # evaluation and reporting
    parser.add_argument('--evaluation_strategy', choices=['no', 'steps', 'epoch'], default='epoch')
    parser.add_argument('--eval_steps', type=int, default=None)

    # saving
    parser.add_argument('--save_strategy', default='epoch', choices=['epoch', 'steps', 'no'],
                        help='Analogous to the huggingface-transformers.Trainer-argument.')
    parser.add_argument('--save_steps', default=None, type=int,
                        help='Analogous to the huggingface-transformers.Trainer-argument.')

    # reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for torch, numpy and random. Set for reproducibility.')
    parser.add_argument('--wandb', type=bool, default=True)
    cmd_args = parser.parse_args()
    train_logger: Optional[logging.Logger] = None
    if not cmd_args.wandb:
        wandb.init(mode="disabled")
    main(cmd_args)
