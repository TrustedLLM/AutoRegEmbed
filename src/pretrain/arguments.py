import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    num_compress_token: int = field(
        default=1, metadata={"help": "number of compress tokens"}
    )
    bfloat16: bool = field(
        default=True
    )
    use_flash_attention_2: bool = field(
        default=True
    )



@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )

    test_data_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )

    context_maxlen: int = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization for context. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    instruction_maxlen: int = field(
        default=512,
        metadata={
            "help": "The maximum total instruction sequence length after tokenization for instructon. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    output_maxlen: int = field(
        default=4096,
        metadata={
            "help": "The maximum total output sequence length after tokenization for . Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    instruction_left: bool = field(
        default=False,
        metadata={
            "help": "将指令拼接到左边"
        },
    )


@dataclass
class CompressTrainingArguments(TrainingArguments):
    
    lora_tune: bool = field(
        default=True, metadata={"help": "Whether to use lora"}
    )
    lora_path: str = field(
        default=None, metadata={"help": "Lora path"}
    )
    lora_rank: int = field(
        default=32, metadata={"help": "Lora rank, only valid when `lora_tune=True`"}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Lora dropout, only valid when `lora_tune=True`"}
    )
    training: bool = field(
        default=True, metadata={"help": "Whether to training"}
    )
    