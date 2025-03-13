import os
import json
import sys
from collections import OrderedDict
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str)
parser.add_argument('--details', action='store_true')
args = parser.parse_args()


all_tasks = OrderedDict({
    "Classification": [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification", # 400000
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Retrieval": [
        "ArguAna",
        "ClimateFEVER", # 5418128
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "DBPedia", # 4636322
        "FEVER", # 5423234
        "FiQA2018",
        "HotpotQA", # 5240734
        "MSMARCO", # 8841866
        "NFCorpus",
        "NQ", # 2684920
        "QuoraRetrieval", # 532931
        "SCIDOCS",
        "SciFact",
        "Touche2020", # 382594
        "TRECCOVID", # 171382
    ],
    "Clustering": [
        "ArxivClusteringP2P", # 732723
        "ArxivClusteringS2S", # 732723
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "RedditClustering", # 420464
        "RedditClusteringP2P", # 459399
        "StackExchangeClustering", # 373850
        "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STS22",
        "STSBenchmark"
    ],
    "Summarization": [
        "SummEval"
    ]
})

df = pd.DataFrame({
    'task_type': [],
    'dataset': [],
    'performance': []
})

for task_type, datasets in all_tasks.items():
    for dataset in datasets:
        df.loc[len(df)] = {
            'task_type': task_type,
            'dataset': dataset,
            'performance': -1.00
        }

for i, row in df.iterrows():
    dataset = row['dataset']
    fname = os.path.join(args.input_path, f'{dataset}.json')
    if not os.path.exists(fname):
        continue
    with open(fname, 'r') as f:
        eval_split = "dev" if dataset == "MSMARCO" else "test"
        target_hf_subset_results = list(filter(lambda x: x['hf_subset'] in {'default', 'en', 'en-en'}, json.load(f)['scores'][eval_split]))
        assert len(target_hf_subset_results) == 1
        main_score = target_hf_subset_results[0]['main_score']
        df.at[i, 'performance'] = main_score

CQADupstackRetrieval_df = df[(df['performance'] != -1) & (df['dataset'].str.startswith('CQADupstack'))]
df = df[~df['dataset'].str.startswith('CQADupstack')]
df = df.reset_index(drop=True)
if len(CQADupstackRetrieval_df) == 12:
    df.loc[len(df)] = {
        'task_type': 'Retrieval',
        'dataset': 'CQADupstackRetrieval',
        'performance': CQADupstackRetrieval_df['performance'].mean()
    }
else:
    df.loc[len(df)] = {
        'task_type': 'Retrieval',
        'dataset': 'CQADupstackRetrieval',
        'performance': -1
    }
df = df[df['performance'] != -1]
df = df.reset_index(drop=True)

def round_performance(v):
    return round(v * 100, 2)

df['performance'] = df['performance'].apply(round_performance)


all_n_missings = 0
for task_type, datasets in all_tasks.items():
    task_df = df[df['task_type'] == task_type]
    if task_type == 'Retrieval':
        n_missings = len(all_tasks['Retrieval']) - 11 - len(task_df)
        n_tasks = len(all_tasks['Retrieval']) - 11
    else:
        n_missings = len(all_tasks[task_type]) - len(task_df)
        n_tasks = len(all_tasks[task_type])
    all_n_missings += n_missings

    task_type += f' ({n_tasks})'
    if n_missings != 0:
        task_type += f' (missing {n_missings})'

    if len(task_df) != 0:
        print(task_type, round(task_df['performance'].mean(), 2))
        
if all_n_missings > 0:
    print(f'all (missing {all_n_missings})', round(df['performance'].mean(), 2))
else:
    print('all', round(df['performance'].mean(), 2))

if args.details:
    # dataset_performance_pairs = zip(df['dataset'].tolist(), map(str, df['performance'].tolist()))
    # print('\n'.join(map(lambda x: f'{x[0]}\t{x[1]}', dataset_performance_pairs)))
    print('======= details ======')
    print('\n'.join(df['dataset'].tolist()))
    print()
    print('\n'.join(map(str, df['performance'].tolist())))
