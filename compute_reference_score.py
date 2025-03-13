import json
from tqdm import tqdm
import random 
import torch
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)
from src.modeling.modeling_autoregembed import ConditionDistributionAlignmentModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pickle as pkl
import torch.multiprocessing as mp
from typing import *
import time
import math
from itertools import chain

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def ensure_list(obj):
    if not isinstance(obj, list):
        obj = [obj]
    return obj

def tokenize_function(examples, 
        model, 
        context_max_len=4096, 
        instruction_max_len=512, 
        output_max_len=4096,
        num_hn=1,
    ):
    examples['neg'] = ensure_list(examples['neg'])
    examples['neg'] = examples['neg'][:num_hn]
    if isinstance(examples['pos'], list):
        examples['pos'] = examples['pos'][0]


    query_ids = model.tokenizer(examples["query"]+'\n', max_length=context_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    instruction_ids = model.tokenizer(examples["instruction"]+'\n', max_length=instruction_max_len, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids = model.tokenizer(examples["pos"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    neg_output_ids = [model.tokenizer(i, max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids'] for i in examples["neg"]]


    pos_output = dict()
    neg_output = dict()

    input_ids = [model.tokenizer.bos_token_id] + query_ids + instruction_ids 
    
    pos_output['input_ids'] = input_ids
    neg_output['input_ids'] = input_ids

    pos_output['target_ids'] = pos_output_ids+[model.tokenizer.eos_token_id]
    neg_output['target_ids'] = [i+[model.tokenizer.eos_token_id] for i in neg_output_ids]

    pos_output['labels'] = [-100]*model.num_compress_token+pos_output_ids + [model.tokenizer.eos_token_id]
    neg_output['labels'] = [[-100]*model.num_compress_token+ i + [model.tokenizer.eos_token_id] for i in neg_output_ids]
    
    return pos_output,neg_output


def dynamic_padding_data(examples, tokenizer, device):      
    input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
    
    target_ids = [torch.tensor(example["target_ids"], dtype=torch.long) for example in examples]
    labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]

    input_ids = dynamic_padding(input_ids, fill_value=tokenizer.pad_token_id)
    attention_mask = torch.where(input_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    target_ids = dynamic_padding(target_ids, fill_value=tokenizer.pad_token_id)
    target_attention_mask = torch.where(target_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))
    
    labels = dynamic_padding(labels)

    batch = {"input_ids": input_ids.to(device),  
        "attention_mask": attention_mask.to(device),
        "target_ids": target_ids.to(device),
        "target_attention_mask": target_attention_mask.to(device),
        "labels": labels.to(device)}
    return batch

def dynamic_padding_data_neg(examples, tokenizer, device):      
    
    input_ids = list(chain.from_iterable([torch.tensor(example["input_ids"], dtype=torch.long)] * len(example['target_ids']) for example in examples))
    target_ids = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["target_ids"]]
    labels = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["labels"]]

    input_ids = dynamic_padding(input_ids, fill_value=tokenizer.pad_token_id)
    attention_mask = torch.where(input_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    target_ids = dynamic_padding(target_ids, fill_value=tokenizer.pad_token_id)
    target_attention_mask = torch.where(target_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))
    
    labels = dynamic_padding(labels)

    batch = {"input_ids": input_ids.to(device),  
        "attention_mask": attention_mask.to(device),
        "target_ids": target_ids.to(device),
        "target_attention_mask": target_attention_mask.to(device),
        "labels": labels.to(device)}
    return batch


def dynamic_padding(sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) #获取sequences的最大长度
        
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences



class InstructedMultiprocessSentenceTransformerWrapper:
    def __init__(
        self,
        model_path,
        decoder_path,
        num_compress_token,
        mp_size=8,
        dtype='float16',
        max_length=512,
    ):
        self.model_path = model_path
        self.decoder_path = decoder_path
        self.num_compress_token = num_compress_token
        self.mp_size = mp_size
        self.dtype = dtype
        self.max_length = max_length
        
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.processes = []
        for rank in range(mp_size):
            p = ctx.Process(
                target=InstructedMultiprocessSentenceTransformerWrapper._encode_per_process, 
                args=(
                    self.model_path,
                    self.decoder_path,
                    self.num_compress_token,
                    self.dtype, 
                    rank, 
                    self.input_queue, 
                    self.output_queue,
                    self.max_length,
                )
            )
            p.start()
            self.processes.append(p)
        
        self.init_timer()
            

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
        decoder_path,
        num_compress_token,
        dtype,
        rank, 
        input_queue,
        output_queue,
        max_length,

    ):
        device = torch.device(f'cuda:{rank}')
        if dtype == 'bfloat16':
            bf16 = True
        elif dtype == 'float16':
            bf16 = False
        

        model = ConditionDistributionAlignmentModel(model_path, decoder_path, bfloat16 = bf16,num_compress_token=num_compress_token).to(device)
        
        model.eval()
        
        model.tokenizer.max_length = max_length

        with torch.no_grad():
            while True:
                batch_id, sentences = input_queue.get()

                data = [tokenize_function(sentence, model) for sentence in sentences] 

                pos_data = [i[0] for i in data]
                neg_data = [i[1] for i in data]
                

                pos_batch = dynamic_padding_data(pos_data, model.tokenizer, device)
                neg_batch = dynamic_padding_data_neg(neg_data, model.tokenizer, device) 
                

                reference_chosen_logps = model(
                    pos_batch['input_ids'],
                    pos_batch['attention_mask'],
                    pos_batch['target_ids'],
                    pos_batch['target_attention_mask'],
                    pos_batch['labels'],
                    ).score.detach().cpu().float() #bs
                
                reference_chosen_logps = reference_chosen_logps.unsqueeze(-1) #bs:1
                
                assert len(neg_batch['input_ids']) == len(neg_batch['target_ids'])
                reference_rejected_logps = model(
                    neg_batch['input_ids'],
                    neg_batch['attention_mask'],
                    neg_batch['target_ids'],
                    neg_batch['target_attention_mask'],
                    neg_batch['labels'],
                    ).score.detach().cpu().float() #bs
                    
                reference_rejected_logps = reference_rejected_logps.view(len(reference_chosen_logps),-1) #bs:num_hn
                output_queue.put((batch_id, reference_chosen_logps, reference_rejected_logps))



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
            batch_id, reference_chosen_logps, reference_rejected_logps = self.output_queue.get()
            id_embeddings.append((batch_id, reference_chosen_logps, reference_rejected_logps))
            if show_progress_bar:
                pbar.update(reference_chosen_logps.shape[0])
        if show_progress_bar:
            pbar.close()
        reference_chosen_logps = torch.cat(list(map(lambda x: x[1], sorted(id_embeddings, key=lambda x: x[0]))), 0)
        reference_rejected_logps = torch.cat(list(map(lambda x: x[2], sorted(id_embeddings, key=lambda x: x[0]))), 0)
        self.encoded_size += len(sentences)
        return reference_chosen_logps, reference_rejected_logps

    def encode(
        self, 
        sentences,
        batch_size=64,
        show_progress_bar=True,
        **kwargs
    ):

        reference_chosen_logps, reference_rejected_logps = self._encode(sentences, batch_size, show_progress_bar)

        return reference_chosen_logps, reference_rejected_logps


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    data_path = './data/medi_nli.jsonl' # nli data path
    model_path = '' #compress_model_path
    decoder_path = 'llama2-7b-hf' #local model path

    data = read_jsonl(data_path)
    
    
    data_instruction = 'This sentence means in one word: “'
    for i in data:
        i['instruction'] = data_instruction

    model = InstructedMultiprocessSentenceTransformerWrapper(
        model_path = model_path,
        decoder_path=decoder_path,
        num_compress_token = 5,
        dtype='bfloat16',
        mp_size=4
    )

    BS = 50000
    num_step = len(data) // BS
    num_step = num_step + 1 if len(data) % BS > 0 else num_step
    reference_chosen_logps = []
    reference_rejected_logps = []
    for i in range(num_step):
        temp_data = data[i*BS:(i+1)*BS]
        reference_chosen_logp, reference_rejected_logp = model.encode(temp_data)
        reference_chosen_logps.append(reference_chosen_logp)
        reference_rejected_logps.append(reference_rejected_logp)

    model.close()
    reference_chosen_logps = torch.cat(reference_chosen_logps,0)#bs:1
    reference_rejected_logps = torch.cat(reference_rejected_logps,0)#bs:num_hn
    print(reference_chosen_logps.shape)
    print(reference_rejected_logps.shape)

    reference_logps = torch.cat([reference_chosen_logps,reference_rejected_logps], dim=1)
    print(reference_logps.size())

    torch.save(reference_logps, './reference_score/llama2-medi-reference-score.pth')

