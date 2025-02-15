from sentence_transformers import SentenceTransformer, models
# from transformers.trainer import Trainer, seed_worker
import sys
sys.path.append('/etc/ssd1/dengjingcheng/compress2retriever')

from transformers.trainer import *
from transformers import AutoModel
from torch.utils.data import DataLoader
from peft import AutoPeftModel, PeftModel
# import shutil
from typing import *
import os
import torch.distributed as dist

def save_ckpt_for_sentence_transformers(
    ckpt_dir,
    tokenizer,
    pooling_mode: str = 'mean', 
    normalized: bool=True
):
    word_embedding_model = models.Transformer(
        os.path.join(ckpt_dir, 'hf'),
        model_args={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True
        },
        config_args={
            "trust_remote_code": True
        }
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), 
        pooling_mode=pooling_mode
    )
    modules = [word_embedding_model, pooling_model]
    if normalized:
        normalize_layer = models.Normalize()
        modules.append(normalize_layer)
    model = SentenceTransformer(modules=modules, device='cpu')
    model.save(os.path.join(ckpt_dir, 'sentence_transformer'))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, 'sentence_transformer'))

class BiTrainer(Trainer):

    def __init__(self, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
        # print('=========== ARGS OF TRAINER ===========')
        # print('global_rank=', dist.get_rank())
        # print(f'global_batch_size={self.args.world_size}*{self.args.per_device_train_batch_size}*{self.args.gradient_accumulation_steps}={self.args.world_size*self.args.per_device_train_batch_size*self.args.gradient_accumulation_steps}')

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            if self.model.lora_tune:
                self.model.save(os.path.join(output_dir, 'lora_adapter'))
                if self.is_world_process_zero():
                    model = AutoModel.from_pretrained(
                        self.model.model_name,
                        use_cache=False,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16 
                    )
                    model = PeftModel.from_pretrained(
                        model, 
                        os.path.join(output_dir, 'lora_adapter')
                    )
                    model = model.merge_and_unload()
                    # shutil.rmtree(os.path.join(output_dir, 'lora_adapter')) # 删除lora_adapter
                    model.save_pretrained(os.path.join(output_dir, 'hf'))
                    del model
            else:
                self.model.save(os.path.join(output_dir, 'hf'))

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


        # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(
        #         output_dir,
        #         self.tokenizer,
        #         pooling_mode=self.model.sentence_pooling_method,
        #         normalized=self.model.normalized
        #     )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
