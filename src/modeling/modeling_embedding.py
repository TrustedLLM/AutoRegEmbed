import logging
import json
from dataclasses import dataclass
from typing import Dict, Optional
import os
import torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers.file_utils import ModelOutput
from transformers import AutoConfig, AutoModel,AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM
from .dist_utils import mismatched_sizes_gather_tensor_with_grad, gather_tensor_without_grad


logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EmbeddingModel(nn.Module):
    def __init__(self,
        model_name: str = None,
        normalized: bool = False, 
        similarity_method: str = 'cos',
        sentence_pooling_method: str = 'mean', 
        negatives_cross_device: bool = False, 
        use_inbatch_neg: bool = True, 
        temperature: float = 1.0, 
        lora_tune: bool = False, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        **kwargs
    ):
        super().__init__()
        assert similarity_method in {'cos', 'dot'}
        self.model_name = model_name
        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.use_inbatch_neg = use_inbatch_neg
        self.temperature = temperature
        self.lora_tune = lora_tune
        self.save_path = save_path
        self.similarity_method = similarity_method

        if 'bf16' in kwargs.keys() and kwargs['bf16']:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation='flash_attention_2',
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 
            )
                
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.float16 
            )
        if lora_tune:
            self.config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            self.model = get_peft_model(self.model, self.config)
            
                
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)#
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.config = self.model.config

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'lasttoken': 
            d = mask.sum(axis=1) - 1 # (batch_size)
            return torch.gather(
                input=hidden_state,
                dim=1, index=d[:, None, None].repeat(1, 1, hidden_state.shape[-1])
            ).squeeze(1) # (batch_size, hidden_dim)
        elif self.sentence_pooling_method == 'eos': #TODO
            raise ValueError('`eos` is changed to `lasttoken`')

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])

        if self.normalized:
            p_reps = F.normalize(p_reps, dim=-1)

        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if self.similarity_method == 'cos':
            q_reps = F.normalize(q_reps, dim=-1)
            p_reps = F.normalize(p_reps, dim=-1)
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self, 
        query: Dict[str, Tensor] = None, 
        passage: Dict[str, Tensor] = None,
        return_loss = True
    ):

        q_reps = self.encode(query)
        p_reps = self.encode(passage)


        if self.negatives_cross_device and self.use_inbatch_neg:
            q_reps = mismatched_sizes_gather_tensor_with_grad(q_reps) # (batch_size, hidden_dim)
            p_reps = mismatched_sizes_gather_tensor_with_grad(p_reps) # (batch_size * group_size, hidden_dim)

        group_size = p_reps.size(0) // q_reps.size(0) # batch_size * group_size / batch_size
        if self.use_inbatch_neg:
            scores = self.compute_similarity(q_reps, p_reps) / self.temperature # (batch_size, batch_size * group_size)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * group_size
            loss = self.compute_loss(scores, target)
        else:
            scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

            scores = scores.view(q_reps.size(0), -1)
            target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)

        if dist.get_rank() == 0:
            
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

class AutoRegEmbed(nn.Module):
    def __init__(self,
        model_name: str = None,
        num_compress_token: int = 1,
        normalized: bool = False,
        similarity_method: str = 'cos',
        negatives_cross_device: bool = False, 
        use_inbatch_neg: bool = True, 
        temperature: float = 1.0, 
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        **kwargs
    ):
        super().__init__()

        assert similarity_method in {'cos', 'dot'}
        self.model_name = model_name
        self.normalized = normalized
        self.similarity_method = similarity_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.lora_tune = lora_tune
        self.save_path = save_path
        self.num_compress_token = num_compress_token

        if 'bf16' in kwargs.keys() and kwargs['bf16']:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation='flash_attention_2',
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.float16 
            )

        if lora_tune:
            if lora_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model, lora_path
            )
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                
                self.model = get_peft_model(self.model, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_tokens =  [f'<EMBED{i}>' for i in range(num_compress_token)]   
        

        self.embed_token_ids = [self.model.config.vocab_size - num_compress_token + i for i in range(num_compress_token)]

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.config = self.model.config


        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _compress(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, 
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)

        lengths = attention_mask.sum(dim=1)

        embedding_ids = torch.cat((input_ids, torch.full((batch_size, self.num_compress_token), self.tokenizer.pad_token_id).to(device)), dim=1)
        embedding_attention_mask = torch.cat((attention_mask, torch.zeros((batch_size, self.num_compress_token), dtype=torch.long).to(device)), dim=1)

        insert_indices = lengths.unsqueeze(1) + torch.arange(self.num_compress_token).unsqueeze(0).to(device)

        embedding_ids.scatter_(1, insert_indices, torch.tensor(self.embed_token_ids, dtype=torch.long).to(device).unsqueeze(0).repeat(batch_size,1))
        embedding_attention_mask.scatter_(1, insert_indices, 1)

        compress_outputs = self.model(embedding_ids, attention_mask = embedding_attention_mask)
        compress_embedding = torch.gather(compress_outputs.last_hidden_state, 1, insert_indices.unsqueeze(-1).expand(-1, -1, compress_outputs.last_hidden_state.size(-1)))
        
        compress_attention_mask = torch.ones(compress_embedding.size(0),compress_embedding.size(1), dtype=torch.long).to(device)

        return compress_embedding, compress_attention_mask

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)


    def encode(self, features):
        if features is None:
            return None
        
        p_reps, p_mask = self._compress(
                    features['input_ids'],
                    attention_mask =  features['attention_mask'])
        
        p_reps = torch.mean(p_reps, dim=1)

        if self.normalized:
            p_reps = F.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if self.similarity_method == 'cos':
            q_reps = F.normalize(q_reps, dim=-1)
            p_reps = F.normalize(p_reps, dim=-1)
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self, 
        query: Dict[str, Tensor] = None, 
        passage: Dict[str, Tensor] = None,
        **kwargs
    ):

        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:

            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = mismatched_sizes_gather_tensor_with_grad(q_reps) # (batch_size, hidden_dim)
                p_reps = mismatched_sizes_gather_tensor_with_grad(p_reps) # (batch_size * group_size, hidden_dim)

            group_size = p_reps.size(0) // q_reps.size(0) # batch_size * group_size / batch_size
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # (batch_size, batch_size * group_size)
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        if dist.get_rank() == 0:
            
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)


    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)