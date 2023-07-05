import argparse
import json
import random
from collections import Counter
from typing import Dict, Any, List

from sklearn.metrics import f1_score, accuracy_score


random.seed(42)


def compute_label_frequencies(input_json: str) -> Dict[str, Dict[str, Any]]:
    labels = []
    with open(input_json, 'r') as fin:
        for line in fin:
            try:
                labels.append(json.loads(line)['label'])
            except:
                labels.append(json.loads(line)['label_gold'])

    label_counts = Counter(labels)
    total_labels = len(labels)

    absolute_frequencies = dict(label_counts)
    relative_frequencies = {label: count / total_labels for label, count in label_counts.items()}

    return {
        'absolute_label_frequencies': absolute_frequencies,
        'relative_label_frequencies': relative_frequencies
    }


def create_majority_class_predictions(labels: list, majority_class: str) -> list:
    return [majority_class] * len(labels)


def create_random_predictions(labels: list, unique_labels: list) -> list:
    return [random.choice(unique_labels) for _ in labels]


def accuracy_majority_class(relative_frequencies: Dict[str, float]) -> float:
    return max(relative_frequencies.values())


def accuracy_random_prediction(relative_frequencies: Dict[str, float]) -> float:
    return sum([freq**2 for freq in relative_frequencies.values()])


def macro_f1_random_prediction(true_labels: List[int]) -> float:
    P = sum(true_labels)
    N = len(true_labels) - P
    # Calculate F1 score for positive and negative classes
    F1_positive = 2 * ((P / (P + N)) * 0.5) / ((P / (P + N)) + 0.5)
    F1_negative = 2 * ((N / (P + N)) * 0.5) / ((N / (P + N)) + 0.5)
    # Calculate the macro-F1 score as the average of the two F1 scores
    macro_f1 = (F1_positive + F1_negative) / 2
    return macro_f1


def write_label_frequencies_to_json(output_json: str, frequencies: Dict[str, Dict[str, Any]], macro_f1_majority: float, macro_f1_random: float, accuracy_majority: float, accuracy_random: float, macro_f1_never_hate: float, macro_f1_always_hate: float, accuracy_never_hate: float, accuracy_always_hate: float) -> None:
    with open(output_json, 'w') as file:
        data = {
            'frequencies': frequencies,
            'macro_f1_majority_class': macro_f1_majority,
            'macro_f1_random_prediction': macro_f1_random,
            'accuracy_majority_class': accuracy_majority,
            'accuracy_random_prediction': accuracy_random,
            'macro_f1_never_hate': macro_f1_never_hate,
            'macro_f1_always_hate': macro_f1_always_hate,
            'accuracy_never_hate': accuracy_never_hate,
            'accuracy_always_hate': accuracy_always_hate
        }
        json.dump(data, file, indent=4)


def main(args: argparse.Namespace) -> None:
    frequencies = compute_label_frequencies(args.path_in)
    labels = list(frequencies['absolute_label_frequencies'].keys())
    true_labels = [label for label, count in frequencies['absolute_label_frequencies'].items() for _ in range(count)]
    majority_class = max(frequencies['absolute_label_frequencies'], key=frequencies['absolute_label_frequencies'].get)

    majority_class_predictions = create_majority_class_predictions(true_labels, majority_class)

    macro_f1_majority = f1_score(true_labels, majority_class_predictions, average='macro')
    macro_f1_random = macro_f1_random_prediction(true_labels)
    print(macro_f1_random)
    acc_majority = accuracy_majority_class(frequencies['relative_label_frequencies'])
    acc_random = accuracy_random_prediction(frequencies['relative_label_frequencies'])
    
    macro_f1_never_hate = f1_score(true_labels, len(true_labels)*[0], average='macro')
    macro_f1_always_hate = f1_score(true_labels, len(true_labels)*[1], average='macro')
    accuracy_never_hate = accuracy_score(true_labels, len(true_labels)*[0])
    accuracy_always_hate = accuracy_score(true_labels, len(true_labels)*[1])

    write_label_frequencies_to_json(args.path_out, frequencies, macro_f1_majority, macro_f1_random, acc_majority, acc_random, macro_f1_never_hate, macro_f1_always_hate, accuracy_never_hate, accuracy_always_hate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute label frequencies from a CSV file and output to a JSON file.')
    parser.add_argument('-i', '--path_in', help='Path to the input CSV file', required=True)
    parser.add_argument('-o', '--path_out', help='Path to the output JSON file', required=True)
    cmd_args = parser.parse_args()
    main(cmd_args)
