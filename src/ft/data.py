import math
import os
import random
from tqdm import tqdm
# from dataclasses import dataclass
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

        pos = random.choice(self.dataset[idx]['pos']) # 随机选一个正例
        passages.append(pos)

        if len(self.dataset[idx]['neg']) < self.args.num_hn: # 如果负样本太少，则上采样负样本
            # raise NotImplementedError()
            num = math.ceil((self.args.num_hn) / len(self.dataset[idx]['neg']))
            negs = random.sample(self.dataset[idx]['neg'] * num, self.args.num_hn)
        ### 随机采样 暂不实现
        else:
            # negs = random.sample(self.dataset[idx]['neg'], self.args.train_group_size - 1)
            # negs = self.dataset[idx]['neg'][:self.args.train_group_size - 1]
            negs = self.dataset[idx]['neg']
        passages.extend(negs)

        return {
            'passages': passages,
            'query': query
        }


class TrainDatasetForEmbedding(Dataset):
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
                
                
                    
                        
                query_template = self.model.context_token+"{query}" 
                instruction_template = self.model.instruction_token+"The document that can answer this question is\n\n"
                q_template = query_template+instruction_template

                dc_template = self.model.context_token+"{document}"
                di_template = self.model.instruction_token+"The text with semantic similarity to this text is\n\n"
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

        pos = random.choice(self.dataset[idx]['pos']) # 随机选一个正例
        passages.append(pos)

        if len(self.dataset[idx]['neg']) < self.args.num_hn: # 如果负样本太少，则上采样负样本
            # raise NotImplementedError()
            num = math.ceil((self.args.num_hn) / len(self.dataset[idx]['neg']))
            negs = random.sample(self.dataset[idx]['neg'] * num, self.args.num_hn)
        ### 随机采样 暂不实现
        else:
            # negs = random.sample(self.dataset[idx]['neg'], self.args.train_group_size - 1)
            # negs = self.dataset[idx]['neg'][:self.args.train_group_size - 1]
            negs = self.dataset[idx]['neg']
        passages.extend(negs)

        return {
            'passages': passages,
            'query': query
        }

class EmbedCollator:
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __init__(
        self,
        tokenizer,
        query_max_len: int = 512,
        passage_max_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len


    def __call__(self, raw_batch):
        query = [f['query'] for f in raw_batch]
        passage = [f['passages'] for f in raw_batch]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        # print(query[0])
        # print(passage[0])

        query_ids = [self.tokenizer(
            i,
            padding=False,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )['input_ids'][0]
            for i in query]
        # query_ids = [torch.tensor(i, dtype=torch.long) for i in query_ids]
        # print(query_ids)
        query_ids = self.dynamic_padding(query_ids, self.tokenizer.pad_token_id)
        query_mask = torch.where(query_ids != self.tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

        passage_ids = [self.tokenizer(
            i,
            padding=False,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )['input_ids'][0]
            for i in passage]
        # passage_ids = [torch.tensor(i, dtype=torch.long) for i in passage_ids]
        passage_ids = self.dynamic_padding(passage_ids, self.tokenizer.pad_token_id)
        passage_mask = torch.where(passage_ids != self.tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))
        
        query = dict()
        query['input_ids'] = query_ids
        query['attention_mask'] = query_mask

        passage = dict()
        passage['input_ids'] = passage_ids
        passage['attention_mask'] = passage_mask


        
        
        batch = {
            "query": query,
            "passage": passage
        }
        
        return batch

    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) #获取sequences的最大长度
        
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)#先全部填充
        # print(padded_sequences.shape)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
