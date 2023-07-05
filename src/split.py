import argparse
import json


def main(args) -> None:
    with open(f'{args.data_dir}/{args.dataset_name}/{args.dataset_name}_preprocessed.jsonl') as fin:
        fout_train = open(f'{args.data_dir}/{args.dataset_name}/{args.dataset_name}_preprocessed_train.jsonl', 'w')
        fout_test = open(f'{args.data_dir}/{args.dataset_name}/{args.dataset_name}_preprocessed_test.jsonl', 'w')
        for line in fin:
            split = json.loads(line.strip('\n'))['split']
            if split == 'train':
                fout_train.write(line)
            elif split == 'dev':
                fout_test.write(line)
            elif split == 'test':
                fout_test.write(line)
            else:
                raise Exception(f'Unexpected split: {split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='Path to data directory.')
    parser.add_argument('-n', '--dataset_name', help='Name of the dataset to be processed')
    cmd_args = parser.parse_args()
    main(cmd_args)
