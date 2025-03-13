import math
import os
import random
from tqdm import tqdm

from typing import List, Tuple
from collections import defaultdict
import copy
import json
import datasets
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.distributed as dist
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from typing import *
from .arguments import DataArguments

def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data


class TrainDatasetForBaseEmbedding(Dataset):
    def __init__(
        self, 
        args, 
        model
    ):
        self.tokenizer = model.tokenizer
        self.model = model
        self.dataset = []
        if not dist.is_initialized() or dist.get_rank() == 0:
            pbar = tqdm(desc='Loading data', smoothing=0)
        if args.train_data_path is not None:
            for example in self._load_data(args.train_data_path, args.num_hn):
                self.dataset.append(example)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    pbar.update(1)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pbar.close()

        self.args = args
        self.total_len = len(self.dataset)


    def _load_data(self, fname, num_hn):
            
        with open(fname, 'r') as f: 
            for line in f:
                line = json.loads(line)
                assert isinstance(line['query'], str)
                if isinstance(line['pos'], str):
                    line['pos'] = [line['pos']]
                assert isinstance(line['pos'], list)
                
                if isinstance(line['neg'], str):
                    line['neg'] = [line['neg']]
                elif isinstance(line['neg'], list):
                    line['neg'] = line['neg'][:num_hn]
                else:
                    raise ValueError('`neg` should be either str or list')
                
                # msmarco
                # query_template = "Web search query: {query}" 
                # instruction_template = "Answer document:"
                # q_template = query_template+instruction_template

                # dc_template = "{document}"
                # di_template = "Below is a paraphrase of this document:"
                # d_template = dc_template+di_template
                
                # sts
                query_template = "{query}\n" 
                instruction_template = "This sentence means in one word: “"
                q_template = query_template+instruction_template

                dc_template = "{document}\n"
                di_template = "This sentence means in one word: “"
                d_template = dc_template+di_template
                

                line['query'] = q_template.format(query=line['query'])
                line['pos'] = list(map(lambda x: d_template.format(document=x), line['pos']))
                line['neg'] = list(map(lambda x: d_template.format(document=x), line['neg']))
                

                yield line

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx) -> Tuple[str, List[str]]:
        query = self.dataset[idx]['query']

        passages = []

        pos = random.choice(self.dataset[idx]['pos']) 
        passages.append(pos)

        if len(self.dataset[idx]['neg']) < self.args.num_hn:
            # raise NotImplementedError()
            num = math.ceil((self.args.num_hn) / len(self.dataset[idx]['neg']))
            negs = random.sample(self.dataset[idx]['neg'] * num, self.args.num_hn)
        
        else:
            # negs = random.sample(self.dataset[idx]['neg'], self.args.train_group_size - 1)
            # negs = self.dataset[idx]['neg'][:self.args.train_group_size - 1]
            negs = self.dataset[idx]['neg']
        passages.extend(negs)

        return {
            'passages': passages,
            'query': query
        }


