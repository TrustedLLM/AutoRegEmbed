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
from .bimistral import BiMistralModel
from .dist_utils import mismatched_sizes_gather_tensor_with_grad, gather_tensor_without_grad


logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None


class RerankerModel(nn.Module):

    def __init__(self,
        model_name: str = None,
        lora_tune: bool = False, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name

        self.lora_tune = lora_tune
        self.save_path = save_path


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if 'bf16' in kwargs.keys() and kwargs['bf16']:
            # if 'infer_bnb' in  kwargs.keys() and kwargs['infer_bnb']:
            
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.model_name,
            #     attn_implementation='flash_attention_2',
            #     use_cache=False,
            #     trust_remote_code=True,
            #     # torch_dtype=torch.float16 
            #     torch_dtype=torch.bfloat16 
            # )#

            # else:
            #     # dist.barrier()
            #     # print('1')
            #     # dist.barrier()
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config)
            #     # dist.barrier()
            #     # print('2')
            #     # dist.barrier()
                
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # attn_implementation='flash_attention_2',
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.float16 
            )
        if lora_tune:
            self.config = LoraConfig(
                r=lora_rank,
                inference_mode=False,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            # print(f'LoRA Config: \n{self.config}')
            
            
            self.model = get_peft_model(self.model, self.config)
            
                
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True, padding_side='left')#

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.config = self.model.config
        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

    

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)



    def encode(self, features):
        if features is None:
            return None
        # features['input_ids'].requires_grad_(True)
        # features['attention_mask'].requires_grad_(True)
        outputs = self.model(input_ids=features['input_ids'], #bs,num,logits
                            attention_mask=features['attention_mask'],
                            # position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                            output_hidden_states=True)
        # print("Outputs structure:", outputs)
        # outputs.logits.requires_grad = True
        # print(outputs.logits.requires_grad)
        
        # 
        scores = outputs.logits[:, -1, self.yes_loc]
        # print(scores)
    
        return scores.contiguous()


    def forward(
        self, 
        pair: Dict[str, Tensor] = None, 
        return_loss = True
    ):

        # print()
        ranker_logits = self.encode(pair) # (batch_size * num, dim)
        # print(ranker_logits.shape)

        grouped_logits = ranker_logits.view(ranker_logits.size(0)//51, -1)
        # print(grouped_logits.shape)
        target = torch.zeros(ranker_logits.size(0)//51, device=grouped_logits.device, dtype=torch.long)
        # print(grouped_logits.requires_grad)
        # print(target.requires_grad)
        loss = self.compute_loss(grouped_logits, target)

        

        if dist.get_rank() == 0:
            
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return RerankerOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)


    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)




@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EmbeddingModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
        model_name: str = None,
        normalized: bool = False, # 是否对embedding归一化
        similarity_method: str = 'cos',
        sentence_pooling_method: str = 'mean', # cls, mean, eos
        negatives_cross_device: bool = False, # 是否全局in-batch负样本
        use_inbatch_neg: bool = True, # 是否device上in-batch负样本
        temperature: float = 1.0, # 温度参数
        lora_tune: bool = False, # 是否使用lora
        lora_rank: int = 32, # lora的rank
        lora_dropout: float = 0.1, # lora的dropout,
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

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        if 'bf16' in kwargs.keys() and kwargs['bf16']:
            # if 'infer_bnb' in  kwargs.keys() and kwargs['infer_bnb']:
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                attn_implementation='flash_attention_2',
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 
            )#

            # else:
            #     # dist.barrier()
            #     # print('1')
            #     # dist.barrier()
            # self.model = AutoModel.from_pretrained(self.model_name, quantization_config=bnb_config)
            #     # dist.barrier()
            #     # print('2')
            #     # dist.barrier()
                
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                # attn_implementation='flash_attention_2',
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
            # print(f'LoRA Config: \n{self.config}')
            
            
            self.model = get_peft_model(self.model, self.config)
            
                
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)#
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.instruction_len = len(self.tokenizer('For a given math problem, along with its relevant background information, the correct answer, and a commonly misunderstood wrong answer, identify the precise cause of the misunderstanding.',return_tensors="pt")['input_ids'][0])
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
        elif self.sentence_pooling_method == 'eos': # 遗留问题：eos
            raise ValueError('`eos` is changed to `lasttoken`')

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])

        if self.normalized:
            p_reps = F.normalize(p_reps, dim=-1)

        # print(features.keys())
        # if instruction_prefix:
        #     pool_mask = features['attention_mask'][:,:self.instruction_len]

        #     p_reps = self.model(input_ids=features['input_ids'],attention_mask=features['attention_mask'],pool_mask=pool_mask ,return_dict=True)['sentence_embeddings']
        # else:
        #     p_reps = self.model(input_ids=features['input_ids'],attention_mask=features['attention_mask'],pool_mask=features['attention_mask'] ,return_dict=True)['sentence_embeddings']
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

        # if self.training:

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

        # else:
        #     scores = self.compute_similarity(q_reps, p_reps)
        #     loss = None

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


class BiEmbeddingModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
        model_name: str = None,
        normalized: bool = False, # 是否对embedding归一化
        similarity_method: str = 'cos',
        sentence_pooling_method: str = 'mean', # cls, mean, eos
        negatives_cross_device: bool = False, # 是否全局in-batch负样本
        use_inbatch_neg: bool = True, # 是否device上in-batch负样本
        temperature: float = 1.0, # 温度参数
        lora_tune: bool = False, # 是否使用lora
        lora_rank: int = 32, # lora的rank
        lora_dropout: float = 0.1, # lora的dropout,
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
            self.model = BiMistralModel.from_pretrained(
                self.model_name,
                attn_implementation='flash_attention_2',
                use_cache=False,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 
            )
        else:
            self.model = BiMistralModel.from_pretrained(
                self.model_name,
                # attn_implementation='flash_attention_2',
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
            # print(f'LoRA Config: \n{self.config}')
            self.model = get_peft_model(self.model, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
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
        elif self.sentence_pooling_method == 'eos': # 遗留问题：eos
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
        passages: Dict[str, Tensor] = None
    ):

        q_reps = self.encode(query)
        p_reps = self.encode(passages)

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



class CompressEmbeddingModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

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
                # attn_implementation='flash_attention_2',
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
                # print(f'LoRA Config: \n{self.config}')
                self.model = get_peft_model(self.model, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        
        self.embed_token = '<EMBED>'
        self.instruction_token = '<INSTRUCTION>'
        self.context_token = '<CONTEXT>'

        self.embed_token_id = self.tokenizer.convert_tokens_to_ids(self.embed_token)
        self.instruction_token_id = self.tokenizer.convert_tokens_to_ids(self.instruction_token)
        self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        
        # self.vocab_size = self.model.config.vocab_size + 2 + num_compress_token

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
        attention_mask: Optional[torch.Tensor] = None, #right_padding
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)

        
        lengths = attention_mask.sum(dim=1)

        
        embedding_ids = torch.cat((input_ids, torch.full((batch_size, self.num_compress_token), self.tokenizer.pad_token_id).to(device)), dim=1)
        embedding_attention_mask = torch.cat((attention_mask, torch.zeros((batch_size, self.num_compress_token), dtype=torch.long).to(device)), dim=1)

        
        insert_indices = lengths.unsqueeze(1) + torch.arange(self.num_compress_token).unsqueeze(0).to(device)

        embedding_ids.scatter_(1, insert_indices, self.embed_token_id)
        embedding_attention_mask.scatter_(1, insert_indices, 1)

        #bs:seq:h_size
        # print(embedding_ids.shape)
        # print(embedding_attention_mask.shape)
        # print(embedding_ids)
        # print(embedding_attention_mask)
        # raise ValueError('Distributed training has not been initialized for representation all gather.')
        # if self.model.device != device:
        #     self.model = self.model.to(device)
        compress_outputs = self.model(embedding_ids, attention_mask = embedding_attention_mask)
        # raise ValueError('Distributed training has not been initialized for representation all gather.')
        compress_embedding = torch.gather(compress_outputs.last_hidden_state, 1, insert_indices.unsqueeze(-1).expand(-1, -1, compress_outputs.last_hidden_state.size(-1)))
        # compress_embedding = compress_outputs.last_hidden_state[:,-1*self.num_compress_token:] #bs:seq:h_size
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


class CompressWithDiffTokenEmbeddingModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

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
                # attn_implementation='flash_attention_2',
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
                # print(f'LoRA Config: \n{self.config}')
                self.model = get_peft_model(self.model, self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_tokens =  [f'<EMBED{i}>' for i in range(num_compress_token)]   
        # self.instruction_token = '<INSTRUCTION>'
        # self.context_token = '<CONTEXT>'

        self.embed_token_ids = [self.model.config.vocab_size - num_compress_token + i for i in range(num_compress_token)]
        # self.instruction_token_id = self.tokenizer.convert_tokens_to_ids(self.instruction_token)
        # self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        
        # self.vocab_size = self.model.config.vocab_size + 2 + num_compress_token

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
        attention_mask: Optional[torch.Tensor] = None, #right_padding
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 获取每个样本的实际长度
        lengths = attention_mask.sum(dim=1)

        #+1保证每个example均有padding_id*num_compress_token，以方便填充
        embedding_ids = torch.cat((input_ids, torch.full((batch_size, self.num_compress_token), self.tokenizer.pad_token_id).to(device)), dim=1)
        embedding_attention_mask = torch.cat((attention_mask, torch.zeros((batch_size, self.num_compress_token), dtype=torch.long).to(device)), dim=1)

        # 插入位置
        insert_indices = lengths.unsqueeze(1) + torch.arange(self.num_compress_token).unsqueeze(0).to(device)
        # print(self.embed_token_ids)

        embedding_ids.scatter_(1, insert_indices, torch.tensor(self.embed_token_ids, dtype=torch.long).to(device).unsqueeze(0).repeat(batch_size,1))
        embedding_attention_mask.scatter_(1, insert_indices, 1)
        # raise ValueError('Distributed training has not been initialized for representation all gather.')

        #bs:seq:h_size
        compress_outputs = self.model(embedding_ids, attention_mask = embedding_attention_mask)
        compress_embedding = torch.gather(compress_outputs.last_hidden_state, 1, insert_indices.unsqueeze(-1).expand(-1, -1, compress_outputs.last_hidden_state.size(-1)))
        # compress_embedding = compress_outputs.last_hidden_state[:,-1*self.num_compress_token:] #bs:seq:h_size
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