import os, sys, json
sys.path.append(".")
sys.path.append("..")
from config import config_global as cg
from config import config_model as cm
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import concurrent.futures as cf
pd.set_option('display.max_rows', None)


def train_test_split(data, lang: str = None, test_size=0.05):

    # in data, rename code_v0 to input and code_v1 to target

    train, test = make_train_test_split_on_problem_id(data, test_size=test_size)
    while len(test) < 1000:
        test_size += 0.01
        print(f"Increasing test size to {test_size:.2f} to get 1000 test examples for {lang}")
        if test_size > 0.75:
            raise ValueError(f"Could not get 1000 test examples, even with test_size={test_size}")
        train, test = make_train_test_split_on_problem_id(data, test_size=test_size)

    # print the size of splits
    print(f"Lang: {lang}, Train: {len(train)}, Test: {len(test)}, Val: {len(test)}")

    train, val = make_train_test_split_on_problem_id(train)
    
    outpath = f"{cg.DATA_PATH}/{lang}_splits/"
    
    train.to_json(f"{outpath}train.jsonl", orient="records", lines=True)
    val.to_json(f"{outpath}val.jsonl", orient="records", lines=True)
    test.to_json(f"{outpath}test.jsonl", orient="records", lines=True)

    test = filter_test_dateset(test)
    test = test[test['is_same']==False]
    test_distinct_inputs = test.groupby(['problem_id','input']).apply(lambda x: x.sample(1)).reset_index(drop=True)
    test_1k = test_distinct_inputs.sample(n=1000, random_state=0)
    test_1k.to_json(f"{outpath}test-1k.jsonl", orient="records", lines=True)


def make_train_test_split_on_problem_id(df, test_size=0.05):
    """Make train/test split on problem_id."""
    np.random.seed(0)
    problem_ids = df["problem_id"].unique()
    test_problem_ids = np.random.choice(
        problem_ids, size=int(len(problem_ids) * test_size), replace=False
    )
    train = df[~df["problem_id"].isin(test_problem_ids)]
    test = df[df["problem_id"].isin(test_problem_ids)]
    return train, test


def get_submissions_by_problemid(problem_id):
    problem_path = os.path.join(cg.codenet_dataset, problem_id)
    problem_lang_path = os.path.join(problem_path, cg.LANG_DICT[cg.LANG])
    if os.path.exists(problem_lang_path):
        submission_files = glob(os.path.join(problem_lang_path, '*'))
        submission_ids = [os.path.basename(submission).split(".")[0] for submission in submission_files]
    
        problem_metadata_path = os.path.join(cg.codenet_metadata, f"{problem_id}.csv")
        problem_metadata_df = pd.read_csv(problem_metadata_path)
        problem_metadata_df = problem_metadata_df[problem_metadata_df['submission_id'].isin(submission_ids) & problem_metadata_df['status'].isin(cg.SUBMISSION_STATUS.keys())]

        return problem_metadata_df
    else:
        return None
    

def construct_bad_good_pairs(df):
    # get best candidates
    df_accepted = df[df['status']=='Accepted']
    if df_accepted.empty:
        return []

    df_sorted = df_accepted.sort_values(by=['cpu_time', 'memory', 'code_size', 'date'], ascending=[True, True, True, False])
    
    best_submission_id = df_sorted.iloc[0]['submission_id']
    best_cpu_time = df_sorted.iloc[0]['cpu_time']

    best_submission_path_pattern = os.path.join(cg.codenet_dataset, df_sorted.iloc[0]['problem_id'], cg.LANG_DICT[cg.LANG], f'{best_submission_id}.*')
    best_submission_path = glob(best_submission_path_pattern)[0]
    with open(best_submission_path, 'r') as fp:
        target_program = fp.read()

    pairs = []
    for index, row in df.iterrows():
        improvement_frac = (row['cpu_time'] - best_cpu_time) / row['cpu_time'] if row['cpu_time'] != 0 else 0
        if improvement_frac > 0.1: # note: we keep pairs pyi, yi`1q for which the relative time improvement is more than 10%
            pair_dict = dict()
            pair_dict['problem_id'] = row['problem_id']
            pair_dict['language'] = row['language']
            pair_dict['submission_id_v0'] = row['submission_id']
            pair_dict['submission_id_v1'] = best_submission_id
            pair_dict['cpu_time_v0'] = row['cpu_time']
            pair_dict['cpu_time_v1'] = df_sorted.iloc[0]['cpu_time']
            pair_dict['memory_v0'] = row['memory']
            pair_dict['memory_v1'] = df_sorted.iloc[0]['memory']
            pair_dict['status_v0'] = row['status']
            pair_dict['status_v1'] = df_sorted.iloc[0]['status']
            pair_dict['improvement_frac'] = improvement_frac

            input_path_pattern = os.path.join(cg.codenet_dataset, row['problem_id'], cg.LANG_DICT[cg.LANG], f"{row['submission_id']}.*")
            input_path = glob(input_path_pattern)[0]
            with open(input_path, 'r') as fp:
                input_program = fp.read()
            pair_dict['input'] = input_program
            pair_dict['target'] = target_program

            pair_dict['code_same'] = (pair_dict['input']==pair_dict['target']) # todo
            pair_dict['measured_runtime_v0'] = sys.maxsize # todo
            pair_dict['measured_runtime_v1'] = sys.maxsize # todo
            pair_dict['key']:[row['submission_id'], best_submission_id]
            pairs.append(pair_dict)

    # print(df_sorted)
    # df_accepted.sort_values(by='cpu_time', ascending=True, inplace=True)
    # print(df_accepted.head(10)['submission_id'])
    # return df_accepted
    return pairs


def process_problem(problem_path):
    problem_submisisons_df = get_submissions_by_problemid(os.path.basename(problem_path))
    if problem_submisisons_df is not None:
        problem_pairs = construct_bad_good_pairs(problem_submisisons_df)
        return problem_pairs
    return None


def get_clones_all():
    # problem_list_path = os.path.join(cg.DATA_PATH, 'Project_CodeNet', 'metadata', 'problem_list.csv')
    # problem_list_df = pd.read_csv(problem_list_path)
    problems = glob(os.path.join(cg.codenet_dataset, '*'))
    problem_df_list = []
    # for problem_path in problems: # problem_list_df['id']:
    #     problem_submisisons_df = get_submissions_by_problemid(os.path.basename(problem_path))
    #     if problem_submisisons_df is not None:
    #         construct_bad_good_pairs(problem_submisisons_df)
    #         problem_df_list.append(problem_submisisons_df)

    with cf.ThreadPoolExecutor() as executor:
        future_to_problem = {executor.submit(process_problem, problem_path): problem_path for problem_path in problems}
        for future in tqdm(cf.as_completed(future_to_problem), total=len(problems)):
            result = future.result()
            if result is not None:
                problem_df_list.extend(result)
    #         
    #         problem_metadata_path = os.path.join(cg.codenet_metadata, problem_id)
    #         problem_metadata_df = pd.read_csv(problem_metadata_path)
    #         problem_metadata_evaluated_df = problem_metadata_df[(problem_metadata_df['language'].str.lower() == cm.LANG.lower()) & (problem_metadata_df['status'] == 'Accepted')]
    #         # print(problem_metadata_evaluated_df.columns)
    #         # get_best_candidates(problem_metadata_evaluated_df)
    #         # print(problem_metadata_evaluated_df[problem_metadata_evaluated_df['cpu_time'] == 0])
    #         # print(problem_metadata_evaluated_df['cpu_time'].unique())
    #         get_programs_by_problemid(problem_id)
    df_all = pd.DataFrame(problem_df_list)
    return df_all
    # print(df_all)


def process_row(index_series_tuple):
    index, row = index_series_tuple
    try:
        if len(row['input']) < cm.count_tokens_max:
            return row
        else:
            # print(data)
            return None
    except Exception as e:
        return None


def filter_test_dateset(df):
    with cf.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row, df.iterrows()), total=len(df)))
    
    results_filtered = [result for result in results if result is not None]
    df_filtered = pd.DataFrame(results_filtered)
    return df_filtered


if __name__ == "__main__":
    df_all = get_clones_all()
    print(len(df_all))

    train_test_split(df_all, lang=cg.LANG)
