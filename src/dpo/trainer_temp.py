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

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        # losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        losses = F.sigmoid(beta *logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

    # policy_chosen_logps 
    # policy_chosen = torch.cat([policy_chosen_logps,policy_rejected_logps],dim=-1)
    # policy_chosen = F.softmax(policy_chosen, dim=-1)
    # target = torch.full((policy_chosen.size(0),), 0, dtype=torch.long, device=policy_chosen.device)
    # loss = F.cross_entropy(policy_chosen, target, reduction='mean')
    # return loss, policy_chosen_logps, policy_rejected_logps






class DPOTrainer(Trainer):

    def __init__(self, dpo_beta, dpo_label_smoothing, *sargs, **kwargs):
        super().__init__(*sargs, **kwargs)
        self.dpo_beta = dpo_beta
        self.dpo_label_smoothing = dpo_label_smoothing
    

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
                    self.model.model.save_pretrained(os.path.join(output_dir, 'hf'))
                    # del model
            else:
                self.model.save(os.path.join(output_dir, 'hf'))

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(os.path.join(output_dir, 'hf'))


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        #多样本dpo loss

    
        # policy_chosen_logps = model(**inputs['pos_batch']).score.unsqueeze(1) #bs:1
        # inputs['pos_batch']['target_ids'] = inputs['pos_batch']['target_ids_self']
        # inputs['pos_batch']['target_attention_mask'] = inputs['pos_batch']['target_attention_mask_self']
        # inputs['pos_batch']['labels'] = inputs['pos_batch']['labels_self']
        # inputs['pos_batch']['input_ids'] = inputs['pos_batch']['input_ids_self']
        # inputs['pos_batch']['attention_mask'] = inputs['pos_batch']['attention_mask_self']
        

        # policy_chosen_logps_self = model(**inputs['pos_batch']).score.unsqueeze(1)#bs:1

        # policy_rejected_logps = model(**inputs['neg_batch']).score#bs*num_hn:1
        # policy_rejected_logps = policy_rejected_logps.view(len(policy_chosen_logps),-1)

        pos_output = model(**inputs['pos_batch'])
        policy_chosen_logps = pos_output.score.unsqueeze(1)
        query_logits = pos_output.g_logits

        inputs['pos_batch']['input_ids'] = inputs['pos_batch']['input_ids_self']
        inputs['pos_batch']['attention_mask'] = inputs['pos_batch']['attention_mask_self']

        pos_self_output = model(**inputs['pos_batch'])
        policy_chosen_logps_self = pos_self_output.score.unsqueeze(1)
        pos_logits = pos_self_output.g_logits

        policy_rejected_logps = model(**inputs['neg_batch']).score#bs*num_hn:1
        policy_rejected_logps = policy_rejected_logps.view(len(policy_chosen_logps),-1)



        bs = len(query_logits)
        log_query_probs = []
        log_pos_probs = []
        for i in query_logits:
            log_query_probs.append(F.log_softmax(i, dim=-1))
        for i in pos_logits:
            log_pos_probs.append(F.log_softmax(i, dim=-1))

        query_probs = []
        pos_probs = []
        

        for i in log_query_probs:
            query_probs.append(i.exp())
        for i in log_pos_probs:
            pos_probs.append(i.exp())
        
        # for i in neg_logits:
        #     log_neg_probs.append(F.log_softmax(i, dim=-1))

        # loss,_,_ = preference_loss(
        #     policy_chosen_logps, 
        #     policy_rejected_logps, 
        #     inputs['pos_batch']['dpo_score'],
        #     inputs['neg_batch']['dpo_score'],
        #     self.dpo_beta,
        #     self.dpo_label_smoothing
        #     )

        bs_js_dist_qp = []
        for i in range(bs):

            qp_probs = 0.5 * (query_probs[i] + pos_probs[i])

            kl_mqp = F.kl_div(query_probs[i].log(),qp_probs,  reduction='batchmean')
            kl_mpp = F.kl_div(pos_probs[i].log(),qp_probs,  reduction='batchmean')
            bs_js_dist_qp.append(0.5 * kl_mqp + 0.5 * kl_mpp )
        bs_js_dist_qp = [tensor.unsqueeze(0) for tensor in bs_js_dist_qp]
        js_dist_qp = torch.cat(bs_js_dist_qp)
        
        loss,_,_ = preference_loss( #bs
            policy_chosen_logps, 
            policy_rejected_logps, 
            inputs['pos_batch']['dpo_score'].view(len(policy_chosen_logps),-1)[:,:1],
            inputs['neg_batch']['dpo_score'].view(len(policy_chosen_logps),-1),
            self.dpo_beta,
            self.dpo_label_smoothing
            )
        #F.sigmoid(
        # if 
        # loss2 = -F.logsigmoid(-self.dpo_beta*torch.abs(policy_chosen_logps-policy_chosen_logps_self))#+torch.log(torch.tensor(2.0))
        # loss2 = F.sigmoid(self.dpo_beta*torch.abs(policy_chosen_logps-policy_chosen_logps_self))

        loss = -torch.log(torch.exp(-js_dist_qp/0.1) / (torch.exp(-js_dist_qp/0.1)+torch.exp(-loss/0.1)))#torch.exp(loss2/0.1) / (torch.exp(loss2/0.1)+torch.exp(loss/0.1))
        # loss2,_,_ = preference_loss(
        #     policy_chosen_logps_self, 
        #     policy_rejected_logps, 
        #     inputs['pos_batch']['dpo_score'].view(len(policy_chosen_logps),-1)[:,1:2],
        #     inputs['neg_batch']['dpo_score'].view(len(policy_chosen_logps),-1),
        #     self.dpo_beta,
        #     self.dpo_label_smoothing
        #     )
        
        # loss2,_,_ = preference_loss(
        #     policy_chosen_logps2, 
        #     policy_rejected_logps2, 
        #     inputs['pos_batch']['dpo_score'],
        #     inputs['neg_batch']['dpo_score'],
        #     self.dpo_beta,
        #     self.dpo_label_smoothing
        #     )

        # loss3,_,_ = preference_loss(
        #     policy_chosen_logps3, 
        #     policy_rejected_logps3, 
        #     inputs['pos_batch']['dpo_score'],
        #     inputs['neg_batch']['dpo_score'],
        #     self.dpo_beta,
        #     self.dpo_label_smoothing
        #     )

        # loss = outputs.loss
        # loss = (loss+loss2) / 2.0
        # loss = loss + loss2
        return (loss, outputs) if return_outputs else loss.mean()
