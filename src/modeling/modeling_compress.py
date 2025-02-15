import logging
import json
from dataclasses import dataclass
from typing import Dict, Optional
import os
import sys
sys.path.append('../src/modeling')
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers.file_utils import ModelOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from .utils import _get_batch_logps,filter_logits_by_labels

logger = logging.getLogger(__name__)

@dataclass
class CompressOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
@dataclass
class DPOReferneceOutput(ModelOutput):
    score: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    g_logits: Optional = None
@dataclass
class DPOOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None



class CompressModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self,
        model_name_or_path: str = None,  
        num_compress_token: int = 1, 
        bfloat16: bool = True, 
        use_flash_attention_2: bool = True,
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1, 
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.num_compress_token = num_compress_token
        self.model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_token = '<EMBED>'
        self.instruction_token = '<INSTRUCTION>'
        self.context_token = '<CONTEXT>'
        if self.training:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            special_tokens_dict = {'additional_special_tokens': [self.embed_token, self.instruction_token, self.context_token]}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            # self.model.config.vocab_size = self.model.config.vocab_size + 3

        self.embed_token_id = self.tokenizer.convert_tokens_to_ids(self.embed_token)
        self.instruction_token_id = self.tokenizer.convert_tokens_to_ids(self.instruction_token)
        self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        
        # self.vocab_size = self.model.config.vocab_size + 2 + num_compress_token
        # self.model.config.vocab_size = self.model.config.vocab_size + 3
        
        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model, lora_path
            )
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules='all-linear',#["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                # print(f'LoRA Config: \n{self.config}')
                self.model = get_peft_model(self.model, self.config)

        if self.training:    # indepedent model for gradient checkpointing

            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )
            self.init_decoder()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = self.model.config

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        if dist.get_rank() == 0:
            print("Freezing the decoder...")
            self.print_trainable_parameters(self)
            print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)


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
        compress_outputs = self.model(embedding_ids, attention_mask = embedding_attention_mask)
        compress_embedding = torch.gather(compress_outputs.last_hidden_state, 1, insert_indices.unsqueeze(-1).expand(-1, -1, compress_outputs.last_hidden_state.size(-1)))
        # compress_embedding = compress_outputs.last_hidden_state[:,-1*self.num_compress_token:] #bs:seq:h_size
        compress_attention_mask = torch.ones(compress_embedding.size(0),compress_embedding.size(1), dtype=torch.long).to(device)

        return compress_embedding, compress_attention_mask


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: torch.LongTensor = None,
        target_attention_mask: Optional[torch.Tensor] = None, #right padding
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding, compress_attention_mask = self._compress(input_ids, attention_mask)

        #拼接
        target_embeddings = self.decoder.model.embed_tokens(target_ids) #bs:seq

        decoder_embeddings = torch.cat([compress_embedding,target_embeddings], dim = 1)
        decoder_attention_mask = torch.cat([compress_attention_mask, target_attention_mask], dim = 1)

        
        decoder_outputs = self.decoder(inputs_embeds=decoder_embeddings, 
            attention_mask = decoder_attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # print(shift_logits.shape)
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            # print(shift_logits.shape)
            # print(shift_labels.shape)
            loss = self.loss_fct(shift_logits, shift_labels)

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

class CompressWithDiffTokenModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self,
        model_name_or_path: str = None,  
        num_compress_token: int = 1, 
        bfloat16: bool = True, 
        use_flash_attention_2: bool = True, 
        lora_tune: bool = False, 
        lora_path: str = None, 
        lora_rank: int = 32, 
        lora_dropout: float = 0.1,
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.num_compress_token = num_compress_token
        self.model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_tokens =  [f'<EMBED{i}>' for i in range(num_compress_token)]      
        # self.instruction_token = '<INSTRUCTION>'
        # self.context_token = '<CONTEXT>'
        if self.training:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            special_tokens_dict = {'additional_special_tokens': self.embed_tokens} #[self.instruction_token, self.context_token]+
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            # self.model.config.vocab_size = self.model.config.vocab_size + 3

        # self.embed_token_id = self.tokenizer.convert_tokens_to_ids(self.embed_token)
        # print(self.model.config.vocab_size)
        self.embed_token_ids = [self.model.config.vocab_size - num_compress_token + i for i in range(num_compress_token)]
        # self.instruction_token_id = self.tokenizer.convert_tokens_to_ids(self.instruction_token)
        # self.context_token_id = self.tokenizer.convert_tokens_to_ids(self.context_token)
        
        # self.vocab_size = self.model.config.vocab_size + 2 + num_compress_token
        # self.model.config.vocab_size = self.model.config.vocab_size + 3
        
        self.lora_tune = lora_tune
        self.save_path = save_path

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

        if self.training:    # indepedent model for gradient checkpointing

            self.decoder = AutoModelForCausalLM.from_pretrained(  self.model_name, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )
            self.init_decoder()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = self.model.config
        # self.decoder_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        # self.gate_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size*4, bias=False)
        # self.up_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size*4, bias=False)
        # self.down_proj = nn.Linear(self.config.hidden_size*4, self.config.hidden_size, bias=False)
        # self.act_fn = nn.SiLU()

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        if dist.get_rank() == 0:
            print("Freezing the decoder...")
            self.print_trainable_parameters(self)
            print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)


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
        # L-2 norm
        # compress_embedding = F.normalize(compress_embedding, p=2, dim=-1)
        # compress_embedding = compress_outputs.last_hidden_state[:,-1*self.num_compress_token:] #bs:seq:h_size
        compress_attention_mask = torch.ones(compress_embedding.size(0),compress_embedding.size(1), dtype=torch.long).to(device)

        return compress_embedding, compress_attention_mask


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: torch.LongTensor = None,
        target_attention_mask: Optional[torch.Tensor] = None, #right padding
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding, compress_attention_mask = self._compress(input_ids, attention_mask)

        # compress_embedding = self.decoder_layer(compress_embedding)
        # compress_embedding = self.down_proj(self.act_fn(self.gate_proj(compress_embedding)) * self.up_proj(compress_embedding))

        #拼接
        target_embeddings = self.decoder.model.embed_tokens(target_ids) #bs:seq

        decoder_embeddings = torch.cat([compress_embedding,target_embeddings], dim = 1)
        decoder_attention_mask = torch.cat([compress_attention_mask, target_attention_mask], dim = 1)

        
        decoder_outputs = self.decoder(inputs_embeds=decoder_embeddings, 
            attention_mask = decoder_attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # print(shift_logits.shape)
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            # print(shift_logits.shape)
            # print(shift_labels.shape)
            loss = self.loss_fct(shift_logits, shift_labels)

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'loss': loss.item()
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)






class CompressWithDiffTokenvMFModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self,
        model_name_or_path: str = None,  #base_model路径
        num_compress_token: int = 1, # 压缩后的token数量
        bfloat16: bool = True, #是否启用bf16精度执行训练
        use_flash_attention_2: bool = True, # 是否启用flash_attention_2执行训练
        lora_tune: bool = False, # 是否使用lora
        lora_path: str = None, # lora adapter地址
        lora_rank: int = 32, # lora的rank
        lora_dropout: float = 0.1, # lora的dropout,
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.num_compress_token = num_compress_token
        self.model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float32
        )
        # print(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_tokens =  [f'<EMBED{i}>' for i in range(num_compress_token)]      
        
        if self.training:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            special_tokens_dict = {'additional_special_tokens': self.embed_tokens} #[self.instruction_token, self.context_token]+
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            

        self.embed_token_ids = [self.model.config.vocab_size - num_compress_token + i for i in range(num_compress_token)]
        
        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model, lora_path
            )
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","embed_tokens"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                # print(f'LoRA Config: \n{self.config}')
                self.model = get_peft_model(self.model, self.config)

        if self.training:    # indepedent model for gradient checkpointing

            self.decoder = AutoModelForCausalLM.from_pretrained(  self.model_name, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float32,
            use_cache=False,
            trust_remote_code=True
        )
            self.init_decoder()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = self.model.config

        # Linear layers for mean and concentration parameter (log of kappa)
        self.mean_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.kappa_layer = nn.Linear(self.model.config.hidden_size, 1)
        self.decoder_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        if dist.get_rank() == 0:
            print("Freezing the decoder...")
            self.print_trainable_parameters(self)
            print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)


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
        # print(self.embed_token_ids)

        embedding_ids.scatter_(1, insert_indices, torch.tensor(self.embed_token_ids, dtype=torch.long).to(device).unsqueeze(0).repeat(batch_size,1))
        embedding_attention_mask.scatter_(1, insert_indices, 1)
        # raise ValueError('Distributed training has not been initialized for representation all gather.')

        #bs:seq:h_size
        compress_outputs = self.model(embedding_ids, attention_mask = embedding_attention_mask)
        compress_embedding = torch.gather(compress_outputs.last_hidden_state, 1, insert_indices.unsqueeze(-1).expand(-1, -1, compress_outputs.last_hidden_state.size(-1)))
        # compress_embedding = compress_outputs.last_hidden_state[:,-1*self.num_compress_token:] #bs:seq:h_size
        compress_attention_mask = torch.ones(compress_embedding.size(0),compress_embedding.size(1), dtype=torch.long).to(device)


        # Compute mean and kappa for vMF
        mean = self.mean_layer(compress_embedding)  # (batch_size, num_compress_token, hidden_size)
        mean = mean / mean.norm(dim=-1, keepdim=True)  # Normalize to unit vector

        log_kappa = self.kappa_layer(compress_embedding).squeeze(-1)  # (batch_size, num_compress_token)
        kappa = torch.exp(log_kappa)  # Concentration parameter

        return compress_embedding, compress_attention_mask, mean,kappa


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: torch.LongTensor = None,
        target_attention_mask: Optional[torch.Tensor] = None, #right padding
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding, compress_attention_mask,z_mean,z_kappa = self._compress(input_ids, attention_mask)

        # print(z_mean.shape)
        # print(z_kappa.shape)

        #
        z_mean_shape = z_mean.shape
        z_mean = z_mean.view(z_mean_shape[0]*z_mean_shape[1],z_mean_shape[2])
        z_kappa_shape = z_kappa.shape
        # if len(z_kappa_shape == 3):
        #     z_kappa = z_kappa.view(z_kappa_shape[0]*z_kappa_shape[1],z_kappa_shape[2])
        # else:
        z_kappa = z_kappa.view(z_kappa_shape[0]*z_kappa_shape[1])

        # z_kappa = z_kappa.unsqueeze(-1)
        # print(z_kappa)
        # print(z_mean)


        #重参数化post_z
        # temp_compress_embedding = []
        # for i in range(z_mean.shape[0]):
        # post_z_dist = VonMisesFisher(z_mean, z_kappa)
        post_z_dist = PowerSpherical(z_mean, z_kappa)
        # print(z_mean.shape)
        # print(z_kappa.shape)
        compress_embedding=post_z_dist.rsample()
        # print(temp_compress_embedding[0].shape)
        # print(compress_embedding.shape)
        compress_embedding = compress_embedding.view(z_mean_shape[0],z_mean_shape[1],z_mean_shape[2])
        compress_embedding = self.decoder_layer(compress_embedding)

        #获取prior_z_dist
        prior_z_dist = HypersphericalUniform(self.model.config.hidden_size - 1, device = compress_embedding.device)

        # print(post_z_dist.entropy())
        # print(prior_z_dist.entropy())

        #拼接
        target_embeddings = self.decoder.model.embed_tokens(target_ids) #bs:seq

        decoder_embeddings = torch.cat([compress_embedding,target_embeddings], dim = 1)
        decoder_attention_mask = torch.cat([compress_attention_mask, target_attention_mask], dim = 1)

        
        decoder_outputs = self.decoder(inputs_embeds=decoder_embeddings, 
            attention_mask = decoder_attention_mask
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # print(shift_logits.shape)
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.decoder.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            # print(shift_logits.shape)
            # print(shift_labels.shape)
            loss = self.loss_fct(shift_logits, shift_labels)

        loss_KL = torch.distributions.kl.kl_divergence(post_z_dist, prior_z_dist).mean()
        

        if dist.get_rank() == 0:
            if self.save_path is not None:
                with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                    line = {
                        'ce_loss': loss.item(),
                        'kl_loss': loss_KL.item(),
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            
        # loss = loss + 0.1*loss_KL
        return CompressOutput(
            loss=loss,
            logits=logits,
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

        torch.save(self.mean_layer.state_dict(), os.path,join(output_dir,'mean_layer.pth'))

        torch.save(self.kappa_layer.state_dict(), os.path,join(output_dir,'kappa_layer.pth'))
        torch.save(self.decoder_layer.state_dict(), os.path,join(output_dir,'decoder_layer.pth'))








class CompressWithDiffTokenReferenceDPOModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self,
        model_name_or_path: str = None,  
        decoder_name_or_path: str = None, 
        num_compress_token: int = 1, 
        bfloat16: bool = True, 
        use_flash_attention_2: bool = False, 
        lora_tune: bool = False, 
        lora_path: str = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: str = None,
        training: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.decoder_name = decoder_name_or_path
        self.num_compress_token = num_compress_token
        self.model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, trust_remote_code=True)
        self.embed_tokens =  [f'<EMBED{i}>' for i in range(num_compress_token)]      
        
        self.embed_token_ids = [self.model.config.vocab_size - num_compress_token + i for i in range(num_compress_token)]
        
        
        self.save_path = save_path

        self.lora_tune = lora_tune
        self.save_path = save_path

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

        if self.training:    # indepedent model for gradient checkpointing

            self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_name, 
            attn_implementation='flash_attention_2' if use_flash_attention_2 else None,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )
            self.init_decoder()


        
        self.config = self.model.config
    
    def freeze_model(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self, model):
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()
        print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


    def init_decoder(self):
        
        self.freeze_model(self.decoder)
        self.decoder.eval()
        # if dist.get_rank() == 0:
        #     print("Freezing the decoder...")
        #     self.print_trainable_parameters(self)
        #     print("Enabling gradient checkpointing...")
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)


    def _compress(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, #right_padding
    ):

        # print(type(input_ids))
        device = input_ids.device
        batch_size = input_ids.size(0)

       
        lengths = attention_mask.sum(dim=1)

      
        embedding_ids = torch.cat((input_ids, torch.full((batch_size, self.num_compress_token), self.tokenizer.pad_token_id).to(device)), dim=1)
        embedding_attention_mask = torch.cat((attention_mask, torch.zeros((batch_size, self.num_compress_token), dtype=torch.long).to(device)), dim=1)

      
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


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: torch.LongTensor = None,
        target_attention_mask: Optional[torch.Tensor] = None, #right padding
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        compress_embedding, compress_attention_mask = self._compress(input_ids, attention_mask)

        # NEFT
        # dims = torch.tensor(compress_embedding.size(1) * compress_embedding.size(2))
        # # mag_norm = 5 / torch.sqrt(dims)
        # # compress_embedding = compress_embedding + torch.zeros_like(compress_embedding).uniform_(-mag_norm, mag_norm)
        # mean = torch.zeros_like(compress_embedding).to(compress_embedding.device)
        # # std_dev = 50 / torch.sqrt(dims).to(compress_embedding.device)
        # noise = torch.normal(mean=mean, std=1).to(compress_embedding.device)
        # compress_embedding = compress_embedding + noise


        #拼接
        target_embeddings = self.decoder.model.embed_tokens(target_ids) #bs:seq

        decoder_embeddings = torch.cat([compress_embedding,target_embeddings], dim = 1)
        decoder_attention_mask = torch.cat([compress_attention_mask, target_attention_mask], dim = 1)

        
        decoder_outputs = self.decoder(inputs_embeds=decoder_embeddings, 
            attention_mask = decoder_attention_mask
        )

        logits = decoder_outputs.logits

        score = _get_batch_logps(logits, labels)

        
        g_logits = filter_logits_by_labels(logits, labels)

        return DPOReferneceOutput(
            score=score,
            logits=logits,
            g_logits=g_logits
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


