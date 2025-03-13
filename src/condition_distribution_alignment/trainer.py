from transformers.trainer import Trainer
from transformers import AutoModel
from peft import AutoPeftModel, PeftModel
import torch
from typing import *
import os
import torch.distributed as dist
from transformers.trainer import *
from peft import AutoPeftModel, PeftModel
import torch.nn.functional as F



class ConditionDistributionAligmentTrainer(Trainer):

    def __init__(self, beta, temperature, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
        self.beta = beta
        self.temperature = temperature
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info("Saving model checkpoint to %s", output_dir)
        print("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            if self.model.lora_tune:
                
                if self.is_world_process_zero():
                    self.model.save(os.path.join(output_dir, 'lora_adapter'))
                #     model = AutoModel.from_pretrained(
                #         self.model.model_name,
                #         torch_dtype=torch.bfloat16,
                #         trust_remote_code=True,
                #         use_cache=False
                #     )
                #     model = PeftModel.from_pretrained(
                #         model, 
                #         os.path.join(output_dir, 'lora_adapter')
                #     )
                #     model = model.merge_and_unload()
                    # shutil.rmtree(os.path.join(output_dir, 'lora_adapter')) # 删除lora_adapter
                    # self.model.model.save_pretrained(os.path.join(output_dir, 'hf'))
                    # del model
            else:
                self.model.save(os.path.join(output_dir, 'hf'))

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        query_output = model(**inputs['pos_batch'])
        query_chosen_logps = query_output.score.unsqueeze(1)

        inputs['pos_batch']['input_ids'] = inputs['pos_batch']['pos_ids']
        inputs['pos_batch']['attention_mask'] = inputs['pos_batch']['pos_attention_mask']

        pos_output = model(**inputs['pos_batch'])
        pos_chosen_logps = pos_output.score.unsqueeze(1)

        query_rejected_logps = model(**inputs['neg_batch']).score
        query_rejected_logps = query_rejected_logps.view(len(query_chosen_logps),-1)

        logits = (query_chosen_logps - query_rejected_logps) + (inputs['pos_batch']['reference_score'].view(len(query_chosen_logps),-1) - inputs['neg_batch']['reference_score'].view(len(query_rejected_logps),-1))
        s2 = F.sigmoid(self.beta * logits)


        s1 = F.sigmoid(self.beta*torch.abs(query_chosen_logps-pos_chosen_logps))

        loss = -torch.log(torch.exp(-s1/self.temperature) / (torch.exp(-s1/self.temperature)+torch.exp(-s2/self.temperature)))
        
        return (loss, outputs) if return_outputs else loss.mean()
