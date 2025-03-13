import json
import torch
from torch.utils.data import Dataset
from itertools import chain

def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data

def ensure_list(obj):
    if not isinstance(obj, list):
        obj = [obj]

    return obj


class ConditionDistributionAligmentDataset(Dataset):
    def __init__(
        self, 
        path,
        instruction,
        reference_score_path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)

        reference_score = torch.load(reference_score_path)

        self.args = args
        self.model = model
        self.total_len = len(self.data)
        self.reference_score = reference_score
        self.instruciton = instruction


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        return pretrain_tokenize_function(
            examples = self.data[idx],
            reference_score = self.reference_score[idx],
            instruction = self.instruciton,
            model = self.model,
            context_max_len = self.args.context_maxlen,
            instruction_max_len = self.args.instruction_maxlen,
            output_max_len = self.args.output_maxlen,
            instruction_left = self.args.instruction_left,
            num_hn = self.args.num_hn
        )


def pretrain_tokenize_function(examples, 
        reference_score,
        instruction,
        model, 
        context_max_len, 
        instruction_max_len, 
        output_max_len,
        instruction_left,
        num_hn
    ):

    examples['neg'] = ensure_list(examples['neg'])
    examples['neg'] = examples['neg'][:num_hn]
    if isinstance(examples['pos'], list):
        examples['pos'] = examples['pos'][0]
    query_input_ids = model.tokenizer(examples["query"]+'\n', max_length=context_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    instruction_ids = model.tokenizer(instruction+'\n', max_length=instruction_max_len, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids = model.tokenizer(examples["pos"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_input_ids = model.tokenizer(examples["pos"]+'\n', max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    
    neg_output_ids = [model.tokenizer(i, max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids'] for i in examples["neg"]]

    pos_output = dict()
    neg_output = dict()

    if instruction_left:
        ## TODO
        input_ids = [model.tokenizer.bos_token_id] + instruction_ids + query_input_ids
    else:
        input_ids = [model.tokenizer.bos_token_id] + query_input_ids + instruction_ids 
        pos_ids = [model.tokenizer.bos_token_id] + pos_input_ids + instruction_ids 
    
    pos_output['input_ids'] = input_ids
    pos_output['pos_ids'] = pos_ids
    neg_output['input_ids'] = input_ids

    pos_output['target_ids'] = pos_output_ids+[model.tokenizer.eos_token_id]
    neg_output['target_ids'] = [i+[model.tokenizer.eos_token_id] for i in neg_output_ids]

    pos_output['labels'] = [-100]*model.num_compress_token+pos_output_ids + [model.tokenizer.eos_token_id]
    neg_output['labels'] = [[-100]*model.num_compress_token+ i + [model.tokenizer.eos_token_id] for i in neg_output_ids]
    
    pos_output['reference_score'] = reference_score[:1]
    neg_output['reference_score'] = reference_score[1:]

    inputs = dict()
    inputs['pos_output'] = pos_output
    inputs['neg_output'] = neg_output

        
    return inputs


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, inputs):
        pos_output = [i['pos_output'] for i in inputs]
        neg_output = [i['neg_output'] for i in inputs]

        examples = pos_output

        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        pos_ids = [torch.tensor(example["pos_ids"], dtype=torch.long) for example in examples]
        target_ids = [torch.tensor(example["target_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        
        reference_score = [i['reference_score'] for i in examples]
        reference_score = torch.cat(reference_score, dim=0)

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        pos_ids = self.dynamic_padding(pos_ids, fill_value=self.pad_token_id)
        pos_attention_mask = torch.where(pos_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        target_ids = self.dynamic_padding(target_ids, fill_value=self.pad_token_id)
        target_attention_mask = torch.where(target_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))
        

        labels = self.dynamic_padding(labels)


        pos_batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels,
            "pos_ids": pos_ids,  
            "pos_attention_mask": pos_attention_mask,
            "reference_score": reference_score}

        examples = neg_output

        input_ids = list(chain.from_iterable([torch.tensor(example["input_ids"], dtype=torch.long)] * len(example['target_ids']) for example in examples))
        target_ids = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["target_ids"]]
        labels = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["labels"]]
        reference_score = [i['reference_score'] for i in examples]
        reference_score = torch.cat(reference_score, dim=0)

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        target_ids = self.dynamic_padding(target_ids, fill_value=self.pad_token_id)
        target_attention_mask = torch.where(target_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))
        
        labels = self.dynamic_padding(labels)

        neg_batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels,
            "reference_score": reference_score}
        
        return {'pos_batch':pos_batch,
            'neg_batch': neg_batch,
            }
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences