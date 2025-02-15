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


class DPOTrainDataset(Dataset):
    def __init__(
        self, 
        path,
        instruction,
        dpo_score_path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)

        dpo_score = torch.load(dpo_score_path)

        self.args = args
        self.model = model
        self.total_len = len(self.data)
        self.dpo_score = dpo_score
        self.instruciton = instruction


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        return pretrain_tokenize_function(
            examples = self.data[idx],
            dpo_score = self.dpo_score[idx],
            instruction = self.instruciton,
            model = self.model,
            context_max_len = self.args.context_maxlen,
            instruction_max_len = self.args.instruction_maxlen,
            output_max_len = self.args.output_maxlen,
            instruction_left = self.args.instruction_left,
            num_hn = self.args.num_hn
        )





def pretrain_tokenize_function(examples, 
        dpo_score,
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
    context_ids = model.tokenizer(examples["query"]+'\n', max_length=context_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    instruction_ids = model.tokenizer(instruction+'\n', max_length=instruction_max_len, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids = model.tokenizer(examples["pos"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    pos_output_ids_self = model.tokenizer(examples["pos"]+'\n', max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    context_ids_self = model.tokenizer(examples["query"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    
    neg_output_ids = [model.tokenizer(i, max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids'] for i in examples["neg"]]

    pos_output = dict()
    neg_output = dict()

    if instruction_left:
        # text_output['input_ids'] = [model.tokenizer.bos_token_id, model.instruction_token_id] + instruction_ids + [model.context_token_id] + context_ids
        input_ids = [model.tokenizer.bos_token_id] + instruction_ids + context_ids
    else:
        # text_output['input_ids'] = [model.tokenizer.bos_token_id, model.context_token_id] + context_ids + [model.instruction_token_id] + instruction_ids 
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

    
    pos_output['dpo_score'] = dpo_score[:2]
    neg_output['dpo_score'] = dpo_score[2:]

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
        input_ids_self = [torch.tensor(example["input_ids_self"], dtype=torch.long) for example in examples]
        target_ids = [torch.tensor(example["target_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        target_ids_self = [torch.tensor(example["target_ids_self"], dtype=torch.long) for example in examples]
        labels_self = [torch.tensor(example["labels_self"], dtype=torch.long) for example in examples]
        dpo_scores = [i['dpo_score'] for i in examples]
        dpo_scores = torch.cat(dpo_scores, dim=0)

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        input_ids_self = self.dynamic_padding(input_ids_self, fill_value=self.pad_token_id)
        attention_mask_self = torch.where(input_ids_self != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        target_ids = self.dynamic_padding(target_ids, fill_value=self.pad_token_id)
        target_attention_mask = torch.where(target_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))
        
        
        target_ids_self = self.dynamic_padding(target_ids_self, fill_value=self.pad_token_id)
        target_attention_mask_self = torch.where(target_ids_self !=  self.pad_token_id, torch.tensor(1), torch.tensor(0))

        labels = self.dynamic_padding(labels)
        labels_self = self.dynamic_padding(labels_self)


        pos_batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels,
            "target_ids_self": target_ids_self,
            "target_attention_mask_self": target_attention_mask_self,
            "labels_self": labels_self,
            "input_ids_self": input_ids_self,  
            "attention_mask_self": attention_mask_self,
            "dpo_score": dpo_scores}

        examples = neg_output

        # input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        # target_ids = [torch.tensor(i, dtype=torch.long) for i in example["target_ids"] for example in examples]
        # labels = [torch.tensor(i, dtype=torch.long) for i in example["labels"] for example in examples]
        input_ids = list(chain.from_iterable([torch.tensor(example["input_ids"], dtype=torch.long)] * len(example['target_ids']) for example in examples))
        target_ids = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["target_ids"]]
        labels = [torch.tensor(i, dtype=torch.long) for example in examples for i in example["labels"]]
        dpo_scores = [i['dpo_score'] for i in examples]
        dpo_scores = torch.cat(dpo_scores, dim=0)

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
            "dpo_score": dpo_scores}
        
        return {'pos_batch':pos_batch,
            'neg_batch': neg_batch,
            }
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of #pad_to_multiple_of的整数倍
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)#先全部填充
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences