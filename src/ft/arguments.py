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
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    normalized: bool = field(
        default=True
    )
    sentence_pooling_method: str = field(
        default='lasttoken', metadata={"help": "The pooling method, should be cls, mean, or lasttoken"}
    )

    similarity_method: str = field(
        default='cos', metadata={"help": "Similarity method."}
    )
    attention_method: str = field(
        default='causal', metadata={"help": "attention method."}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    num_hn: int = field(default=1)


    query_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )





@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(
        default=False, metadata={"help": "share negatives across devices"}
    )
    temperature: Optional[float] = field(
        default=0.02
    )
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    use_inbatch_neg: bool = field(
        default=True, metadata={"help": "Use passages in the same batch as negatives"}
    )
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