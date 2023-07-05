import argparse
import csv
from collections import defaultdict

import pandas as pd
from googletrans import Translator
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    translator = Translator()
    target_langs = ['ar', 'nl', 'fr', 'de', 'hi', 'it', 'pl', 'pt', 'es', 'zh-CN']
    df = pd.read_csv(args.path_in)
    fout = open(args.path_out, 'a')
    writer = csv.writer(fout)
    writer.writerow(['identifier', 'en'] + target_langs)
    for i, row in tqdm(df.iterrows()):
        if args.skip and i < args.skip:
            continue
        row_transl = [row['identifier'], row['English']]
        for tlang in target_langs:
            row_transl.append(translator.translate(row['English'], src='en', dest=tlang).text)
        writer.writerow(row_transl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to input file, csv that contains the hypotheses to translate.')
    parser.add_argument('-o', '--path_out', help='Path to output file, csv that contains all hypotheses translations.')
    parser.add_argument('-s', '--skip', type=int, help='Number of rows to skip from the beginning of the input file.')
    cmd_args = parser.parse_args()
    main(cmd_args)
