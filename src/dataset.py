import csv
import json
import random
from typing import Optional, Dict, List, Iterator, Any

from tqdm import tqdm
import torch


item_type = Dict[str, Any]


# label mapping from https://huggingface.co/morit/XLM-T-full-xnli/blob/main/config.json
NLI_LABEL2ID = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}


def get_numeric_label(item: item_type) -> torch.LongTensor:
    return torch.LongTensor([int(item['label'])])


class Dataset(torch.utils.data.IterableDataset):

    labels_str_to_num = None
    labels_num_to_str = None

    def __init__(self, path: str, name: str, **kwargs: Optional[Dict[str, str]]):
        """Initializer function.

        Args:
            path: Path to dataset file if the dataset is one file or
                otherwise to the directory containing the dataset files.
            name: Name of the data set.
            kwargs: Any additional keyword arguments.
        """
        self.path = path
        self.name = name
        self.kwargs = kwargs
        self._items = []
        self._encoded = False

    def load(self) -> List[Dict[str, str]]:
        raise NotImplementedError

    def get_num_items(self) -> int:
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[item_type]:
        if self._encoded:
            for item in self._items:
                if 'input_ids' in item:
                    try:
                        yield {'input_ids': item['input_ids'], 'token_type_ids': item['token_type_ids'],
                            'attention_mask': item['attention_mask'], 'labels': item['labels']}
                    except KeyError:
                        yield {'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'],
                            'labels': item['labels']}
                else:
                    yield item
        else:
            for item in self._items:
                yield item
    
    def _has_hypotheses(self) -> bool:
        if 'hypothesis' in self._items[0]:
            return True
        return False

    def get_num_label(self, str_label: int) -> int:
        """Given a string label, get the corresponding numeric label."""
        return self.labels_str_to_num[str_label]        
    
    def to_nli(self, hypothesis: str) -> None:
        """Add hypotheses and do hypothesis-augmentation."""
        for item in self._items:
            item['hypothesis'] = hypothesis
            # adjust label values for NLI
            if item['label'] == 0:
                item['label'] = NLI_LABEL2ID['contradiction']
            elif item['label'] == 1:
                item['label'] = NLI_LABEL2ID['entailment']
            else:
                raise Exception(f'Unexpected label value for item. label: {item["label"]}, item: {item}')
    
    def nli_sanity_check(self) -> None:
        for item in self._items:
            assert item['label'] in [0, 2]
    
    def encode_dataset(self, tokenizer) -> None:
        for item in tqdm(self._items):
            if self._has_hypotheses():
                enc = self._encode_item_with_hypotheses(tokenizer, text=item['text'], hypothesis=item['hypothesis'])
            else:
                enc = self._encode_item(tokenizer, item['text'])
            enc['labels'] = get_numeric_label(item).squeeze()
            enc['input_ids'] = enc['input_ids'].squeeze()
            try:
                enc['token_type_ids'] = enc['token_type_ids'].squeeze()
            except KeyError:
                pass
            enc['attention_mask'] = enc['attention_mask'].squeeze()
            item.update(enc)
        self._encoded = True
    
    @staticmethod
    def _encode_item(tokenizer, text: str) -> item_type:
        return tokenizer(text=text, return_tensors='pt', truncation=True, padding=True)
    
    @staticmethod
    def _encode_item_with_hypotheses(tokenizer, text: str, hypothesis: str) -> item_type:
        return tokenizer(text=text, text_pair=hypothesis, return_tensors='pt', truncation=True,
                         padding=True, return_token_type_ids=True)


class HateCheckDataset(Dataset):

    labels_str_to_num = {
        'hateful': 1,
        'non-hateful': 0
    }
    labels_num_to_str = {v: k for k, v in labels_str_to_num.items()}

    def load(self) -> None:
        with open(self.path) as fin:
            for line in fin:
                row = json.loads(line)
                self._items.append({
                    'id': row['id'],
                    'text': row['text'],
                    'label': row['label'],
                    'category': row['functionality']
                })

    def get_num_items(self) -> int:
        count = 0
        with open(self.path) as fin:
            reader = csv.DictReader(fin)
            for _ in reader:
                count += 1
        return count


class DefaultDataset(Dataset):
    """
    loads datasets in default format and structure: csv with id, text and label.
    """
    
    labels_str_to_num = {
        '1': 1,
        '0': 0
    }
    labels_num_to_str = {v: k for k, v in labels_str_to_num.items()}

    def load(self, load_limit: Optional[int] = None, random_seed: Optional[int] = None) -> None:
        with open(self.path) as fin:
            for line in fin:
                row = json.loads(line)
                self._items.append({
                    'id': row['id'],
                    'text': row['text'],
                    'label': row['label'],
                })
        if load_limit:
            if random_seed:
                random.seed(random_seed)
            random.shuffle(self._items)
            self._items = self._items[:load_limit]
            

    def get_num_items(self) -> int:
        count = 0
        with open(self.path) as fin:
            reader = csv.DictReader(fin)
            for _ in reader:
                count += 1
        return count
