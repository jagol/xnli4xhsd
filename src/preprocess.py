import os
import random
import re
import sys
import csv
import json
import argparse
import logging
from io import StringIO
from html import unescape
from typing import Dict, Any

import compute_data_stats


random.seed(5)


def get_logger() -> logging.Logger:
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    prepro_logger = logging.getLogger('preprocess')
    fname = os.path.join('../preprocessing_logs.txt')
    file_handler = logging.FileHandler(os.path.join('.', fname))
    file_handler.setFormatter(log_formatter)
    prepro_logger.addHandler(file_handler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(log_formatter)
    prepro_logger.addHandler(consoleHandler)
    prepro_logger.setLevel('INFO')
    return prepro_logger


class PreProcessor:

    def __init__(self, corpus_name: str, path_in: str, out_dir: str, dev_size: int, test_size: int) -> None:
        self._corpus_name = corpus_name
        self._path_in = path_in
        self._out_dir = out_dir
        self._test_size = test_size
        self._dev_size = dev_size
        self._items = []
        self._splits = {'train': [], 'dev': [], 'test': []}

    def preprocess(self) -> None:
        self._load()
        self._split()
        self._clean()
        for split in self._splits:
            if self._splits[split]:
                self._write_items_to_outfile(self._splits[split], split)
    
    def _load(self) -> None:
        raise NotImplementedError
    
    def _split(self) -> None:
        random.shuffle(self._items)        
        self._splits['test'] = self._items[:self._test_size]
        self._splits['dev'] = self._items[self._test_size:self._test_size + self._dev_size]
        self._splits['train'] = self._items[self._test_size + self._dev_size:]
        
    def _downsample_class(self, label: int, target_ratio: float) -> None:
        maj_class_items = [item for item in self._items if item['label'] == label]
        min_class_items = [item for item in self._items if item['label'] != label]
        target_maj_len = int(len(min_class_items) / target_ratio - len(min_class_items))
        maj_class_items_downsampled = random.sample(maj_class_items, k=target_maj_len)        
        self._items = min_class_items + maj_class_items_downsampled
        random.shuffle(self._items)

    def _clean(self) -> None:
        for split in  self._splits:
            for item in self._splits[split]:
                item['text'] = self._clean_text(item['text'])

    def _write_item_to_outfile(self, item: Dict[ str, Any ], outfile: StringIO) -> None:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _write_items_to_outfile(self, items, split: str) -> None:
        with open(os.path.join(self._out_dir, f'{self._corpus_name}_{split}_{len(items)}.jsonl'), 'w') as fout:
            for item in items:
                self._write_item_to_outfile(item, fout)
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Cleaning steps adopted from 
        https://github.com/paul-rottger/efficient-low-resource-hate-detection/blob/master/1_dataloading/1_wrangling_data.ipynb
        """
        text = unescape(text)
        text = re.sub(r"@[A-Za-z0-9_-]+",'@user', text) # format expected by XLM-T
        text = re.sub(r"http\S+",'http', text) # format expected by XLM-T
        text = re.sub(r"\n",' ', text)
        text = re.sub(r"\r",' ', text)  # not adopted
        text = re.sub(r"\t",' ', text)
        text = text.replace("[URL]", "http") # format expected by XLM-T
        text = text.strip()
        return text


class HateCheckPreProcessor(PreProcessor):
    
    labels = {
        'non-hateful': 0,
        'hateful': 1
    }

    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row[''],
                    'text': row['test_case'],
                    'label': self.labels[row['label_gold']],
                    'functionality': row['functionality'],
                    'corpus_name': self._corpus_name,
                })
    
    def _split(self) -> None:
        self._splits['test'] = self._items


class MHCPreProcessor(PreProcessor):
    
    labels = {
        'non-hateful': 0,
        'hateful': 1
    }

    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['mhc_case_id'],
                    'text': row['test_case'],
                    'label': self.labels[row['label_gold']],
                    'functionality': row['functionality'],
                    'corpus_name': self._corpus_name,
                })
    
    def _split(self) -> None:
        self._splits['test'] = self._items


class BAS19_ESPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['id'],
                    'text': row['text'],
                    'label': int(row['HS']),
                    'corpus_name': self._corpus_name,
                })


class DYN21_ENPreProcessor(PreProcessor):
    
    labels = {
        'nothate': 0,
        'hate': 1
    }
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row[''],
                    'text': row['text'],
                    'label': self.labels[row['label']],
                    'corpus_name': self._corpus_name,
                })


class FOR19_PTPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for i, row in enumerate(dreader):
                self._items.append({
                    'id': i,
                    'text': row['text'],
                    'label': int(row['hatespeech_comb']),
                    'corpus_name': self._corpus_name,
                })


class FOU18_ENPreProcessor(PreProcessor):
    
    labels = {
        'abusive': 0,
        'normal': 0,
        'spam': 0,
        'hateful': 1
    }
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for i, row in enumerate(dreader):
                self._items.append({
                    'id': i,
                    'text': row['text'],
                    'label': self.labels[row['label']],
                    'corpus_name': self._corpus_name,
                })
    
    def preprocess(self) -> None:
        self._load()
        self._downsample_class(0, 0.22)
        self._split()
        self._clean()
        for split in self._splits:
            if self._splits[split]:
                self._write_items_to_outfile(self._splits[split], split)


class HAS21_HIPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['text_id'],
                    'text': row['text'],
                    'label': 1 if row['task_2'] == 'HATE' else 0,
                    'corpus_name': self._corpus_name,
                })


class KEN20_ENPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['comment_id'],
                    'text': row['text'],
                    'label': int(row['label_hate_maj']),
                    'corpus_name': self._corpus_name,
                })

    def preprocess(self) -> None:
            self._load()
            self._downsample_class(0, 0.5)
            self._split()
            self._clean()
            for split in self._splits:
                if self._splits[split]:
                    self._write_items_to_outfile(self._splits[split], split)


class OUS19_ARPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['HITId'],
                    'text': row['tweet'],
                    'label': 1 if 'hateful' in row['sentiment'] else 0,
                    'corpus_name': self._corpus_name,
                })
    
    def _clean_text(self, text: str) -> str:
        text = super(OUS19_ARPreProcessor, self)._clean_text(text)
        text.replace("@url", "http")
        return text


class SAN20_ITPreProcessor(PreProcessor):
    
    def _load(self) -> None:
        with open(self._path_in) as fin:
            dreader = csv.DictReader(fin)
            for row in dreader:
                self._items.append({
                    'id': row['id'],
                    'text': row['text'],
                    'label': int(row['hs']),
                    'corpus_name': self._corpus_name,
                })


def main(args: argparse.Namespace) -> None:
    for corpus_name, path_in in CORPUS_PATHS.items():
        logger.info(f'Start processing {corpus_name}.')
        path_in = os.path.join(args.data_dir, path_in)
        if corpus_name.startswith('MHC'):
            out_dir = os.path.join(args.data_dir, 'processed', 'MHC')
            corpus_processor = PREPROCESSORS['MHC'](
                corpus_name, path_in, out_dir, None, None)
        elif corpus_name == 'HateCheck':
            out_dir = os.path.join(args.data_dir, 'processed', 'HateCheck')
            corpus_processor = PREPROCESSORS[corpus_name](
                corpus_name, path_in, out_dir, None, None)
        else:  
            out_dir = os.path.join(args.data_dir, 'processed', corpus_name)
            corpus_processor = PREPROCESSORS[corpus_name](
                corpus_name, path_in, out_dir,
                DEV_TEST_SIZES[corpus_name]['dev'], DEV_TEST_SIZES[corpus_name]['test']
            )
        corpus_processor.preprocess()
        logger.info(f'Finished processing {corpus_name}.')
        logger.info(f'Compute corpus stats.')
        try:
            train_fns = [fn for fn in os.listdir(out_dir) if 'train' in fn]
            dev_fns = [fn for fn in os.listdir(out_dir) if 'train' in fn]
            test_fns = [fn for fn in os.listdir(out_dir) if 'train' in fn]
        except:
            import pdb; pdb.set_trace()
        if train_fns:
            path_out_train = os.path.join(out_dir, [fn for fn in os.listdir(out_dir) if 'train' in fn][0])
            compute_data_stats.main(argparse.Namespace(**{'path_in': path_out_train, 'path_out': path_out_train.replace('.jsonl', '_stats.json')}))
        if dev_fns:
            path_out_dev = os.path.join(out_dir, [fn for fn in os.listdir(out_dir) if 'dev' in fn][0])
            compute_data_stats.main(argparse.Namespace(**{'path_in': path_out_dev, 'path_out': path_out_dev.replace('.jsonl', '_stats.json')}))
        if test_fns:
            path_out_test = os.path.join(out_dir, [fn for fn in os.listdir(out_dir) if 'test' in fn][0])
            compute_data_stats.main(argparse.Namespace(**{'path_in': path_out_test, 'path_out': path_out_test.replace('.jsonl', '_stats.json')}))
    logger.info('Preprocessing finished.')


class BinaryLabels:
    hate_speech = 1
    not_hate_speech = 0


class Splits:
    train = 'train'
    dev = 'dev'
    test = 'test'


PREPROCESSORS = {
    'HateCheck': HateCheckPreProcessor,
    'MHC': MHCPreProcessor,
    'BAS19_ES': BAS19_ESPreProcessor,
    'DYN21_EN': DYN21_ENPreProcessor,
    'FOR19_PT': FOR19_PTPreProcessor,
    'FOU18_EN': FOU18_ENPreProcessor,
    'HAS21_HI': HAS21_HIPreProcessor,
    'KEN20_EN': KEN20_ENPreProcessor,
    'OUS19_AR': OUS19_ARPreProcessor,
    # 'OUS19_FR': OUS19_FRPreProcessor,
    'SAN20_IT': SAN20_ITPreProcessor
}


DEV_TEST_SIZES = {
    'BAS19_ES': {'dev': 500, 'test': 2000},
    'DYN21_EN': {'dev': 500, 'test': 2000},
    'FOR19_PT': {'dev': 500, 'test': 2000},
    'FOU18_EN': {'dev': 500, 'test': 2000},
    'HAS21_HI': {'dev': 300, 'test': 500},
    'KEN20_EN': {'dev': 500, 'test': 2000},
    'OUS19_AR': {'dev': 300, 'test': 1000},
    'OUS19_FR': {'dev': 500, 'test': 1500},
    'SAN20_IT': {'dev': 500, 'test': 2000}
}


CORPUS_PATHS = {
    'BAS19_ES': 'raw/natural/bas19_es.csv',
    'DYN21_EN': 'raw/natural/dyn21_en.csv',
    'FOR19_PT': 'raw/natural/for19_pt.csv',
    'FOU18_EN': 'raw/natural/fou18_en.csv',
    'HAS21_HI': 'raw/natural/has21_hi.csv',
    'KEN20_EN': 'raw/natural/ken20_en.csv',
    'OUS19_AR': 'raw/natural/ous19_ar.csv',
    'OUS19_FR': 'raw/natural/ous19_fr.csv',
    'SAN20_IT': 'raw/natural/san20_it.csv',
    'MHC_AR': 'raw/MHC/hatecheck_cases_final_arabic.csv',
    'MHC_NL': 'raw/MHC/hatecheck_cases_final_dutch.csv',
    'MHC_FR': 'raw/MHC/hatecheck_cases_final_french.csv',
    'MHC_DE': 'raw/MHC/hatecheck_cases_final_german.csv',
    'MHC_HI': 'raw/MHC/hatecheck_cases_final_hindi.csv',
    'MHC_IT': 'raw/MHC/hatecheck_cases_final_italian.csv',
    'MHC_ZH': 'raw/MHC/hatecheck_cases_final_mandarin.csv',
    'MHC_PL': 'raw/MHC/hatecheck_cases_final_polish.csv',
    'MHC_PT': 'raw/MHC/hatecheck_cases_final_portuguese.csv',
    'MHC_ES': 'raw/MHC/hatecheck_cases_final_spanish.csv',
    'HateCheck': 'raw/MHC/HateCheck_test.csv',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='data/', help='Path to input data directory.')
    cmd_args = parser.parse_args()
    logger = get_logger()
    main(cmd_args)
