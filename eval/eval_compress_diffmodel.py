import os
import sys
sys.path.append('../AutoRegEmbed')

import mteb
from mteb import MTEB_MAIN_EN
from sentence_transformers import SentenceTransformer, models
from src.modeling import CompressEmbeddingModel, CompressWithDiffTokenEmbeddingModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import json
import logging
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import math
from typing import *
import argparse
from collections import OrderedDict
import datasets

from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
# datasets.disable_caching()

# def get_tasks(split_id, n, fast=False):
all_tasks = OrderedDict({
    "Classification": [
        "AmazonCounterfactualClassification",
        # "AmazonPolarityClassification", # 400000
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
        # "ClimateFEVER", # 5418128
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
        # "DBPedia", # 4636322
        # "FEVER", # 5423234
        "FiQA2018",
        # "HotpotQA", # 5240734
        # "MSMARCO", # 8841866
        "NFCorpus",
        # "NQ", # 2684920
        # "QuoraRetrieval", # 532931
        "SCIDOCS",
        "SciFact",
        # "Touche2020", # 382594
        # "TRECCOVID", # 171382
    ],
    "Clustering": [
        # "ArxivClusteringP2P", # 732723
        # "ArxivClusteringS2S", # 732723
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        # "RedditClustering", # 420464
        # "RedditClusteringP2P", # 459399
        # "StackExchangeClustering", # 373850
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

def get_datasets(task_type='all', split_id=None, n_splits=None, path=None):
    if task_type == 'all':
        all_datasets = [datasets for _, datasets in all_tasks.items()]
        all_datasets = sum(all_datasets, [])
    else:
        all_datasets = all_tasks[task_type]
        # print(all_datasets)
    if path is not None and os.path.exists(path):
        ready_datasets = list(map(lambda x: x.replace('.json', ''), os.listdir(path)))
        for ready_dataset in ready_datasets:
            if ready_dataset != 'model_meta' and ready_dataset in all_datasets:
                all_datasets.remove(ready_dataset)
    if split_id is None:
        return all_datasets, None

    dataset_num = list()
    for dataset in all_datasets:
        eval_splits = ["dev"] if dataset == "MSMARCO" else ["test"]
        evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=[dataset], languages=["eng",]))
        task = evaluation.tasks[0]
        if task.metadata_dict['descriptive_stats']['n_samples'] is not None:
            num_total = task.metadata_dict['descriptive_stats']['n_samples'][eval_splits[0]]
        elif task.metadata_dict['descriptive_stats']['avg_character_length'] is not None:
            num_total = task.metadata_dict['descriptive_stats']['avg_character_length'][eval_splits[0]]['num_documents'] + task.metadata_dict['descriptive_stats']['avg_character_length'][eval_splits[0]]['num_queries']
        else:
            num_total = 0
        dataset_num.append((dataset, num_total))

    dataset_num.sort(key=lambda x: (x[1], x[0]), reverse=True)
    groups = [[] for _ in range(n_splits)]
    sums = [0] * n_splits
    for dataset, num_total in dataset_num:
        # 找到当前总和最小的组
        min_sum_index = sums.index(min(sums))
        # 将数字添加到该组，并更新总和
        groups[min_sum_index].append((dataset, num_total))
        sums[min_sum_index] += num_total
    
    datasets = sorted(groups[split_id], key=lambda x: x[1])
    total_size = sum([size for _, size in datasets])
    datasets = [dataset for dataset, _ in datasets]
    
    return datasets, total_size

class InstructedMultiprocessSentenceTransformerWrapper:

    def __init__(
        self,
        model_path, 
        proj=False, 
        mp_size=1,
        field_template=True,
        instructions=None,
        dtype='float16',
        max_length=512,
        num_compress_token=1
    ):
        self.model_path = model_path
        self.proj = proj
        self.mp_size = mp_size
        self.field_template = field_template
        self.dtype = dtype
        self.max_length = max_length
        self.num_compress_token = num_compress_token
        # with open(os.path.join(model_path, 'modules.json'), 'r') as f:
        #     if json.load(f)[-1]['type'] == 'sentence_transformers.models.Normalize':
        #         self.normalized = True
        #     else:
        #         self.normalized = False
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=InstructedMultiprocessSentenceTransformerWrapper._encode_per_process, 
                args=(
                    self.model_path,
                    self.dtype, 
                    rank, 
                    self.proj, 
                    self.input_queue, 
                    self.output_queue,
                    self.max_length,
                    self.num_compress_token
                )
            )
            p.start()
            self.processes.append(p)
        
        self.init_timer()
        self.instructions = instructions
        if instructions is None:
            self.instructions = None
        else:
            self.instructions = instructions
            # if self.proj and self.field_template:
            #     self.instructions = OrderedDict({dataset: {'text': f'<instruction>\n{instruction}\n</instruction>'} for dataset, instruction in instructions.items()})
            # else:
            #     self.instructions = OrderedDict({dataset: {'text': instruction} for dataset, instruction in instructions.items()})
            # if self.proj:
            #     instructions = []
            #     for i, (dataset, item) in enumerate(self.instructions.items()):
            #         instructions.append(item['text'])
            #     embeddings = self._encode(instructions, 64)
            #     for i, dataset in enumerate(self.instructions.keys()):
            #         self.instructions[dataset]['embedding'] = F.normalize(embeddings[i], dim=-1)

    def close(self):
        for p in self.processes:
            p.terminate()
        
        for p in self.processes:
            p.join()
            p.close()
        
        self.input_queue.close()
        self.output_queue.close()
    
    def init_timer(self):
        self.start_time = time.time()
        self.encoded_size = 0

    @staticmethod
    def _encode_per_process(
        model_path, 
        dtype,
        rank, 
        proj, 
        input_queue,
        output_queue,
        max_length,
        num_compress_token
    ):
        device = torch.device(f'cuda:{rank}')
        # model = AutoModel.from_pretrained(model_path,torch_dtype=dtype).to(device)
        # model = AutoModel.from_pretrained(
        #     model_path,
        #     attn_implementation='flash_attention_2',
        #     use_cache=False,
        #     trust_remote_code=True,
        #     torch_dtype=torch.bfloat16 
        # )
        if dtype == 'bfloat16':
            model = CompressWithDiffTokenEmbeddingModel(model_path,num_compress_token=num_compress_token).to(device)
        else:
            model = CompressWithDiffTokenEmbeddingModel(model_path,num_compress_token=num_compress_token, bf16=False).to(device)
        
        model.eval()
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # tokenizer.pad_token = tokenizer.eos_token
        model.tokenizer.max_length = max_length

        with torch.no_grad():
            while True:
                batch_id, sentences = input_queue.get()
                sentence_encoding = model.tokenizer(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=model.tokenizer.max_length,
                    add_special_tokens=True
                ).to(device)
                if not proj:
                    embeddings,_ = model._compress(
                        sentence_encoding['input_ids'],
                        sentence_encoding['attention_mask']
                    )
                    embeddings = torch.mean(embeddings, dim=1).cpu().float()
                    # embeddings = embeddings[:,-1,:].cpu().float()
                    embeddings = F.normalize(embeddings, dim=-1)
                else: # 
                    sentence_encoding = model[0](sentence_encoding)
                    embeddings = model[1](
                        sentence_encoding
                    )['sentence_embedding'].cpu().float()
                output_queue.put((batch_id, embeddings))

    def _encode(
        self,
        sentences: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False
    ):
        batch_size = min(batch_size, math.ceil(len(sentences) / self.mp_size))
        for start in range(0, len(sentences), batch_size):
            self.input_queue.put((start, sentences[start: start + batch_size]))
        if show_progress_bar:
            pbar = tqdm(total=len(sentences), desc=f'Encoded size: {self.encoded_size}, consumed time: {round(time.time() - self.start_time, 2)}s')
        id_embeddings = []
        for _ in range(0, len(sentences), batch_size):
            batch_id, embeddings = self.output_queue.get()
            id_embeddings.append((batch_id, embeddings))
            if show_progress_bar:
                pbar.update(embeddings.shape[0])
        if show_progress_bar:
            pbar.close()
        embeddings = torch.cat(list(map(lambda x: x[1], sorted(id_embeddings, key=lambda x: x[0]))), 0)
        self.encoded_size += len(sentences)
        return embeddings

    def encode(
        self, 
        sentences,
        prompt_name=None,
        prompt=None,
        batch_size=64,
        show_progress_bar=True,
        **kwargs
    ):
        if isinstance(sentences[0], dict):
            sentences = list(map(lambda d: '\n'.join([text for _, text in d.items()]).strip(), sentences))
        # print(sentences[0])
        if self.field_template:
            if self.proj:
                sentences = list(map(lambda sentence: f'<text>\n{sentence}\n</text>', sentences))
            else:
                instruction = self.instructions[prompt_name]
                if instruction['symmetry']:#\n\nThe following text is semantically similar to this text:
                    # STS template test
                    #template0
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>The text with semantic similarity to this text is\n\n" #
                    #template1
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>The text with semantic similarity to this text is as follows:" #
                    #template2
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>This sentence means in one word: “" #
                    #template3
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>This sentence means in one sentence: " #
                    #template4
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>This sentence means in one sentence: “" #
                    #template5
                    # prompt_template = "<CONTEXT>{text}<INSTRUCTION>Use one word to express the meaning of this text: “" #
                    #template6
                    # prompt_template = "<CONTEXT>Over 100 dead as typhoon slams central Philippines.\n This sentence means in one word: “Disaster<INSTRUCTION>{text}\nThis sentence means in one word: “" #
                    #template7
                    # prompt_template = "<CONTEXT>Over 100 dead as typhoon slams central Philippines.\n This sentence means in one word: “Disaster\n{text}<INSTRUCTION>This sentence means in one word: “" #
                    #template8
                    # prompt_template = "<CONTEXT>A jockey riding a horse.\nThis sentence means in one word: “Equestrian<INSTRUCTION>{text}\nThis sentence means in one word: “" #
                    #template9
                    # prompt_template = "<CONTEXT>A jockey riding a horse.<INSTRUCTION>This sentence means in one word: “Equestrian<CONTEXT>{text}<INSTRUCTION>This sentence means in one word: “" #
                    #template10
                    prompt_template = "{text}\nThis sentence means in one word: “" #
                    #template11
                    # prompt_template = "A jockey riding a horse.\nThis sentence means in one word: “Equestrian\n{text}\nThis sentence means in one word: “" #
                    #template12
                    # prompt_template = "{text}\nThis sentence means in one word: “\n" #
                    
                    # prompt_template = instruction['prompt_template']


                    # prompt = f'Instruct: {instruction}\nQuery: '
                    sentences = list(map(lambda sentence: prompt_template.format(text=sentence), sentences))
                    # pass
                else:
                    # prompt_template = instruction['prompt_template']['query']['next']
                    # prompt_template = "<CONTEXT>{query}<INSTRUCTION>The document that can answer this question is\n\n"
                    # template2
                    prompt_template = '''Query: "{query}". Use one word to represent the query in a retrieval task. The word is: “'''
                    sentences = list(map(lambda sentence: prompt_template.format(query=sentence), sentences))


        embeddings = self._encode(sentences, batch_size, show_progress_bar)
        if self.proj: 
            instruction_embedding = self.instructions[prompt_name]['embedding'].unsqueeze(-1) # (emb_dim, 1)
            embeddings -= torch.matmul(embeddings, instruction_embedding) * instruction_embedding.T
            if self.normalized:
                embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    encode_queries = encode
    
    def encode_corpus(
        self, 
        sentences,
        prompt_name=None,
        prompt=None,
        batch_size=64,
        show_progress_bar=True,
        **kwargs
    ):
        if isinstance(sentences[0], dict):
            sentences = list(map(lambda d: '\n'.join([text for _, text in d.items()]).strip(), sentences))
        if self.field_template and self.proj:
            sentences = list(map(lambda sentence: f'<text>\n{sentence}\n</text>', sentences))
        if self.field_template:
            instruction = self.instructions[prompt_name]
            if instruction['symmetry']:
                # prompt_template = instruction['prompt_template']
                prompt_template = "<CONTEXT>{text}<INSTRUCTION>The text with semantic similarity to this text is\n\n"
                # prompt = f'Instruct: {instruction}\nQuery: '
                sentences = list(map(lambda sentence: prompt_template.format(text=sentence), sentences))
            else:
                # prompt_template = instruction['prompt_template']['document']['self']
                # prompt_template = "<CONTEXT>{document}<INSTRUCTION>The text with semantic similarity to this text is\n\n"
                # template2
                prompt_template = '''Passage: "{document}". Use one word to represent the passage in a retrieval task. The word is: “'''
                sentences = list(map(lambda sentence: prompt_template.format(document=sentence), sentences))

        embeddings = self._encode(sentences, batch_size, show_progress_bar)        
        if self.proj:
            instruction_embedding = self.instructions[prompt_name]['embedding'].unsqueeze(-1) # (emb_dim, 1)
            embeddings -= torch.matmul(embeddings, instruction_embedding) * instruction_embedding.T
            if self.normalized:
                embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--instruction_file', type=str, default='mteb.json')
    parser.add_argument('--proj', action='store_true')
    parser.add_argument('--field_template', action='store_true')
    parser.add_argument('--mp_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--score_function', type=str, choices=['cos_sim', 'dot'], default='cos_sim')
    parser.add_argument('--split_id', type=int, default=None)
    parser.add_argument('--n_splits', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dtype', type=str, required=True, choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--task_type', type=str, default='all')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_compress_token', type=int, default=1)
    # parser.add_argument('--left', action='store_true')
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join(args.model_path, 'mteb_results')

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main")

    with open(args.instruction_file, 'r') as f:
        instructions = json.load(f)
        # for dataset, instruction in json.load(f).items():
        #     if isinstance(instruction, dict):
        #         instruction = instruction['query']
        #     instructions[dataset] = instruction

    if args.dataset is None:
        all_datasets, total_size = get_datasets(
            args.task_type, args.split_id, args.n_splits,
            os.path.join(args.save_path, 'no_model_name_available', 'no_revision_available')
        )
    else:
        all_datasets = args.dataset.split(',')
        total_size = None

    print(all_datasets)
    print(total_size)

    model = InstructedMultiprocessSentenceTransformerWrapper(
        model_path=args.model_path, 
        dtype=args.dtype,
        proj=args.proj, 
        mp_size=args.mp_size,
        field_template=args.field_template,
        max_length=args.max_length,
        instructions=instructions,
        num_compress_token = args.num_compress_token
    )
    start = time.time()
    # MTEB_MAIN_EN.tasks

    for dataset in all_datasets:
        model.init_timer()
        try:
            eval_splits = ["dev"] if dataset == "MSMARCO" else ["test"]
            evaluation = mteb.MTEB(
                tasks=mteb.get_tasks(tasks=[dataset], languages=["eng",])
            )
            eval_results = evaluation.run(
                model, 
                output_folder=args.save_path, 
                eval_splits=eval_splits,
                encode_kwargs={
                    "batch_size": args.batch_size
                },
                verbosity=2,
                score_function=args.score_function
            )
        except Exception as e:
            print(e)
            continue
    model.close()
    #     if task_type == 'Retrieval':
    #         cqadupstack_main_scores = []
    #         datasets_to_remove = []
    #         for dataset, main_score in results[task_type].items():
    #             if dataset.startswith('CQADupstack'):
    #                 datasets_to_remove.append(dataset)
    #                 cqadupstack_main_scores.append(main_score)
    #         for dataset in datasets_to_remove:
    #             del results[task_type][dataset]
    #         results[task_type]['CQADupstackRetrieval'] = sum(cqadupstack_main_scores) / len(cqadupstack_main_scores)

    #     main_scores = [main_score for _, main_score in results[task_type].items()]
    #     avg_main_score = sum(main_scores) / len(main_scores)
    #     logger.info(task_type + ': ' + str(avg_main_score))
    #     results[task_type]['overall'] = avg_main_score
    #     with open(os.path.join(args.save_path, 'all_results.json'), 'w') as f:
    #         json.dump(results, f, indent=2)
    
    # main_scores = []
    # for task_type, dataset_scores in results.items():
    #     for dataset, main_score in dataset_scores.items():
    #         if dataset != 'overall':
    #             main_scores.append(main_score)
    # avg_main_score = sum(main_scores) / len(main_scores)
    # logger.info('Overall: ' + str(avg_main_score))
    # results['overall'] = avg_main_score
    # with open(os.path.join(args.save_path, 'all_results.json'), 'w') as f:
    #     json.dump(results, f, indent=2)