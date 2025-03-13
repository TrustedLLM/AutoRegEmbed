import json
import torch
from torch.utils.data import Dataset


def read_jsonl(input_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  
                data.append(json.loads(line))
    return data


class InformationCompressDataset(Dataset):
    def __init__(
        self, 
        path,
        args, 
        model
    ):
        if path is not None:
            self.data = read_jsonl(path)

        self.args = args
        self.model = model
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        return pretrain_tokenize_function(
            examples = self.data[idx],
            model = self.model,
            context_max_len = self.args.context_maxlen,
            instruction_max_len = self.args.instruction_maxlen,
            output_max_len = self.args.output_maxlen,
            instruction_left = self.args.instruction_left,
        )





def pretrain_tokenize_function(examples, 
        model, 
        context_max_len, 
        instruction_max_len, 
        output_max_len,
        instruction_left,
    ):
    context_ids = model.tokenizer(examples["context"]+'\n', max_length=context_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    instruction_ids = model.tokenizer(examples["instruction"]+'\n', max_length=instruction_max_len, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']
    output_ids = model.tokenizer(examples["output"], max_length=output_max_len, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)['input_ids']

    text_output = dict()

    if instruction_left:
        text_output['input_ids'] = [model.tokenizer.bos_token_id] + instruction_ids + context_ids
    else:
        text_output['input_ids'] = [model.tokenizer.bos_token_id] + context_ids + instruction_ids 
    

    text_output['target_ids'] = output_ids+[model.tokenizer.eos_token_id]
    text_output['labels'] = [-100]*model.num_compress_token+output_ids + [model.tokenizer.eos_token_id]
        
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        target_ids = [torch.tensor(example["target_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]

        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        attention_mask = torch.where(input_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))

        target_ids = self.dynamic_padding(target_ids, fill_value=self.pad_token_id)
        target_attention_mask = torch.where(target_ids != self.pad_token_id, torch.tensor(1), torch.tensor(0))
        
        labels = self.dynamic_padding(labels)

        batch = {"input_ids": input_ids,  
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,
            "labels": labels}
        return batch
        
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences) 
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of 
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences