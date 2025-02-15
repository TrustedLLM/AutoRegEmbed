import json
from tqdm import tqdm
import random 
import torch
import numpy as np
import sys
sys.path.append('../AutoRegembed')
from src.modeling import CompressEmbeddingModel, CompressWithDiffTokenEmbeddingModel, CompressWithDiffTokenReferenceDPOModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pickle as pkl
import torch.multiprocessing as mp
from typing import *
import time
import math
from itertools import chain

def ensure_list(obj):

    if not isinstance(obj, list):

        obj = [obj]
    return obj

def tokenize_function(examples, 
        model, 
        context_max_len=512, 
        instruction_max_len=512, 
        output_max_len=512,
        num_hn=5,
    ):
    examples['neg'] = ensure_list(examples['neg'])
    examples['neg'] = examples['neg'][:num_hn]
    if isinstance(examples['pos'], list):
        examples['pos'] = examples['pos'][0]


    context_ids = model.tokenizer(examples["query"]+'\n', max_length=context_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    instruction_ids = model.tokenizer(examples["instruction"]+'\n', max_length=instruction_max_len, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids = model.tokenizer(examples["pos"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids_self = model.tokenizer(examples["pos"]+'\n', max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    context_ids_self = model.tokenizer(examples["query"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']

    neg_output_ids = [model.tokenizer(i, max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids'] for i in examples["neg"]]


    pos_output = dict()
    neg_output = dict()

    input_ids = [model.tokenizer.bos_token_id] + context_ids + instruction_ids 
    input_ids_self = [model.tokenizer.bos_token_id] + pos_output_ids_self + instruction_ids 
    # print(text_output['input_ids'])
    # import pdb; pdb.set_trace()
    pos_output['input_ids'] = input_ids
    pos_output['input_ids_self'] = input_ids_self
    neg_output['input_ids'] = input_ids

    pos_output['target_ids'] = pos_output_ids+[model.tokenizer.eos_token_id]
    pos_output['target_ids_self'] = context_ids_self+[model.tokenizer.eos_token_id]
    neg_output['target_ids'] = [i+[model.tokenizer.eos_token_id] for i in neg_output_ids]

    pos_output['labels'] = [-100]*model.num_compress_token+pos_output_ids + [model.tokenizer.eos_token_id]
    pos_output['labels_self'] = [-100]*model.num_compress_token+context_ids_self + [model.tokenizer.eos_token_id]
    neg_output['labels'] = [[-100]*model.num_compress_token+ i + [model.tokenizer.eos_token_id] for i in neg_output_ids]
    
    return pos_output,neg_output


def dynamic_padding_data(examples, tokenizer, device):      
    input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
    input_ids_self = [torch.tensor(example["input_ids_self"], dtype=torch.long) for example in examples]
    
    target_ids = [torch.tensor(example["target_ids"], dtype=torch.long) for example in examples]
    labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
    target_ids_self = [torch.tensor(example["target_ids_self"], dtype=torch.long) for example in examples]
    labels_self = [torch.tensor(example["labels_self"], dtype=torch.long) for example in examples]

    input_ids = dynamic_padding(input_ids, fill_value=tokenizer.pad_token_id)
    attention_mask = torch.where(input_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    input_ids_self = dynamic_padding(input_ids_self, fill_value=tokenizer.pad_token_id)
    attention_mask_self = torch.where(input_ids_self != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    target_ids = dynamic_padding(target_ids, fill_value=tokenizer.pad_token_id)
    target_attention_mask = torch.where(target_ids != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))

    target_ids_self = dynamic_padding(target_ids_self, fill_value=tokenizer.pad_token_id)
    target_attention_mask_self = torch.where(target_ids_self != tokenizer.pad_token_id, torch.tensor(1), torch.tensor(0))
    
    labels = dynamic_padding(labels)
    labels_self = dynamic_padding(labels_self)

    batch = {"input_ids": input_ids.to(device),  
        "attention_mask": attention_mask.to(device),
        "target_ids": target_ids.to(device),
        "target_attention_mask": target_attention_mask.to(device),
        "labels": labels.to(device),
        "target_ids_self": target_ids_self.to(device),
        "target_attention_mask_self": target_attention_mask_self.to(device),
        "labels_self": labels_self.to(device),
        "input_ids_self": input_ids_self.to(device),  
        "attention_mask": attention_mask_self.to(device)}
    return batch

def dynamic_padding_data_neg(examples, tokenizer, device):      
    # input_ids = [torch.tensor(example["input_ids"], dtype=torch.long)*len(example['target_ids']) for example in examples ]
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
        model_type, #base, compress 
        num_compress_token,
        mp_size=8,
        dtype='float16',
        max_length=512,
    ):
        self.model_path = model_path
        self.decoder_path = decoder_path
        self.model_type = model_type
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
                    self.model_type,
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
        model_type,
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
        

        model = CompressWithDiffTokenReferenceDPOModel(model_path, decoder_path, bfloat16 = bf16,num_compress_token=num_compress_token).to(device)
        # if model_type == 'base':
        #     if attention_method == 'causal':
        #         model = EmbeddingModel(model_path,normalized = normalized,sentence_pooling_method=sentence_pooling_method,bf16=bf16).to(device)
        #     elif attention_method == 'Bi':
        #         model = BiEmbeddingModel(model_path,normalized = normalized,sentence_pooling_method=sentence_pooling_method,bf16=bf16).to(device)
        # elif model_type == 'compress':
        #     model = CompressWithDiffTokenEmbeddingModel(model_path,num_compress_token=num_compress_token).to(device)
        
        model.eval()
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # tokenizer.pad_token = tokenizer.eos_token
        model.tokenizer.max_length = max_length

        with torch.no_grad():
            while True:
                batch_id, sentences = input_queue.get()

                data = [tokenize_function(sentence, model) for sentence in sentences] #bs

                pos_data = [i[0] for i in data]
                neg_data = [i[1] for i in data]
                # print(pos_data[0])

                pos_batch = dynamic_padding_data(pos_data, model.tokenizer, device)
                neg_batch = dynamic_padding_data_neg(neg_data, model.tokenizer, device) 
                # print(pos_batch['input_ids'])

                reference_chosen_logps = model(
                    pos_batch['input_ids'],
                    pos_batch['attention_mask'],
                    pos_batch['target_ids'],
                    pos_batch['target_attention_mask'],
                    pos_batch['labels'],
                    ).score.detach().cpu().float() #bs
                reference_chosen_logps_self = model(
                    pos_batch['input_ids'],
                    pos_batch['attention_mask'],
                    pos_batch['target_ids_self'],
                    pos_batch['target_attention_mask_self'],
                    pos_batch['labels_self'],
                    ).score.detach().cpu().float() #bs
                reference_chosen_logps = torch.stack([reference_chosen_logps,reference_chosen_logps_self],dim=1)#bs:2
                # reference_rejected_logps = []
                # for i in range(len(neg_batch['target_ids'])):
                # print(neg_batch['input_ids'].shape)
                # print(neg_batch['target_ids'].shape)
                # print(neg_batch['input_ids'])
                # print(neg_batch['target_ids'])
                assert len(neg_batch['input_ids']) == len(neg_batch['target_ids'])
                reference_rejected_logps = model(
                    neg_batch['input_ids'],
                    neg_batch['attention_mask'],
                    neg_batch['target_ids'],
                    neg_batch['target_attention_mask'],
                    neg_batch['labels'],
                    ).score.detach().cpu().float() #bs
                    
                # reference_rejected_logps = torch.stack(reference_rejected_logps, dim=0) #bs*num_hn
                reference_rejected_logps = reference_rejected_logps.view(len(reference_chosen_logps),-1)
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
    data_path = 'medi_ALLNLI_default.jsonl'
 
    model_path = 'llama2-7b-pretrain-pwcunique-5token-diff-nospecial/hf'
    decoder_path = 'llama2-7b-hf'

    data = read_jsonl(data_path)
    
    
    data_instruction = 'This sentence means in one word: “'
    for i in data:
        i['instruction'] = data_instruction
    # data = [f"<CONTEXT>{i['text']}<INSTRUCTION>The text with semantic similarity to this text is\n\n" for i in data]
    # data = [i['text'] for i in data]
    # data = [i['text']+data_template for i in data]
    # data_cor = [f"<CONTEXT>{i['text']}<INSTRUCTION>The text with semantic similarity to this text is\n\n" for i in data_cor]
    # data_query = [f"<CONTEXT>{i['text']}<INSTRUCTION>The document that can answer this question is\n\n" for i in data_query]
    


    model = InstructedMultiprocessSentenceTransformerWrapper(
        model_path = model_path,
        decoder_path=decoder_path,
        model_type = 'compress',
        num_compress_token = 5,
        dtype='bfloat16'
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
    reference_chosen_logps = torch.cat(reference_chosen_logps,0)#bs:2
    reference_rejected_logps = torch.cat(reference_rejected_logps,0)#bs:num
    print(reference_chosen_logps.shape)
    print(reference_rejected_logps.shape)

    reference_logps = torch.cat([reference_chosen_logps,reference_rejected_logps], dim=1)
    print(reference_logps.size())

    torch.save(reference_logps, '../AutoRegEmbed/data/dpo_score/llama2-pwcunique_medi_ALLNLI_default.pth')

