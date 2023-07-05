import argparse
import json
import os

import pandas as pd


def parse_results(results_dir: str) -> pd.DataFrame:
    results = []
    for eng_corpus in os.listdir(results_dir):
        if eng_corpus not in ['X_DEN', 'X_FEN', 'X_KEN', 'X_NLI_DEN', 'X_NLI_FEN', 'X_NLI_KEN', 'X_NLI', 'monoling_target_lang', 'X', 'X_mNLI', 'M_mNLI']:
            continue
        if 'NLI' in eng_corpus:
            model = 'morit/XLM-T-full-xnli'
        else:
            model = 'cardiffnlp/twitter-xlm-roberta-base'
        for target_lang_corpus in os.listdir(os.path.join(results_dir, eng_corpus)):
            if target_lang_corpus.startswith('RUN'):
                continue
            elif target_lang_corpus.endswith('xnli'):
                continue
            for fn_size in os.listdir(os.path.join(results_dir, eng_corpus, target_lang_corpus)):
                fn_size_num = int(fn_size.split('_')[1])
                for run in os.listdir(os.path.join(results_dir, eng_corpus, target_lang_corpus, fn_size)):
                    run_path = os.path.join(results_dir, eng_corpus, target_lang_corpus, fn_size, run)
                    run_num = int(run[3:])
                    for strategy in os.listdir(run_path):
                        if strategy.startswith('checkpoint'):
                            continue
                        if strategy.startswith('comp_metrics_results_0.json'):
                            continue
                        strat_path = os.path.join(run_path, strategy)
                        for fn in os.listdir(strat_path):
                            if not fn.endswith('eval.json'):
                                continue
                            test_set = '_'.join(fn.split('_')[:-1])
                            with open(os.path.join(strat_path, fn)) as fin:
                                d = json.load(fin)
                                if 'MHC' in fn or 'HateCheck' in fn:
                                    f1 = d['overall']['f1-macro']
                                    acc = d['overall']['acc']
                                else:
                                    f1 = d['metrics']['f1-macro']
                                    acc = d['metrics']['acc']
                                results.append([model, eng_corpus, target_lang_corpus, fn_size_num, strategy, run_num, test_set, f1, acc])
    pd_results = pd.DataFrame(results, columns=['model', 'english_corpus', 'target_lang_corpus', 'target_lang_examples', 
                                                'strategy', 'run', 'test_set', 'macro-f1', 'accuracy'])
    # filtered_df = pd_results[
    #     (pd_results['english_corpus'] == 'X_NLI_DEN') &
    #     (pd_results['target_lang_corpus'] == 'BAS19_ES') &
    #     (pd_results['target_lang_examples'] == 0) &
    #     (pd_results['strategy'] == 'baseline') &
    #     (pd_results['run'].isin(['RUN2', 'RUN3', 'RUN10']))
    # ]
    # import pdb; pdb.set_trace()
    return pd_results


def compute_run_avgs(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in df.columns if col not in ['run', 'macro-f1', 'accuracy']]
    groupby_obj = df.groupby(group_cols)
    df_out = groupby_obj.agg({'run': 'count', 'macro-f1': ['mean', 'std'], 'accuracy': 'mean'}).reset_index()
    df_out.columns = df_out.columns.map('_'.join).str.strip('_')
    df_out = df_out.rename(columns={'run_last': 'count', 'macro-f1_mean': 'mean-macro-f1', 'accuracy_mean': 'mean-accuracy', 'macro-f1_std': 'f1-stddev'})
    return df_out


def main(args: argparse.Namespace) -> None:
    df = parse_results(args.results_dir)
    df.to_csv(os.path.join(args.output_dir, 'raw_results.csv'))
    df_avg = compute_run_avgs(df)
    if len(df_avg[df_avg["run_count"]!=10]) != 0:
        import pdb; pdb.set_trace()
    df_avg.to_csv(os.path.join(args.output_dir, 'seed_avg_results.csv'))
    
    # drop acc 
    df_avg = df_avg.drop("mean-accuracy", axis=1)
    
    df_natural_only_avg = df_avg[~df_avg['test_set'].str.contains('MHC')]
    df_natural_only_avg.to_csv(os.path.join(args.output_dir, 'natural_only_avg_results.csv'))
    df_mhc_only_avg = df_avg[df_avg['test_set'].str.contains('MHC')]
    df_mhc_only_avg.to_csv(os.path.join(args.output_dir, 'mhc_only_avg_results.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--results_dir')
    parser.add_argument('-o', '--output_dir')
    cmd_args = parser.parse_args()
    main(cmd_args)
