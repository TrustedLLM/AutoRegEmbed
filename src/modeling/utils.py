import torch
import torch.nn.functional as F




def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def filter_logits_by_labels(logits, labels):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    batch_size, seq_num, vocab_size = logits.shape
    selected_logits_list = []

    for i in range(batch_size):
        # 对每个样本创建布尔掩码
        mask = (labels[i] != -100)
        
        # 使用掩码选择该样本对应的 logits 元素
        sample_logits = logits[i][mask]  # 保持 batch 维度
        
        # 将结果添加到列表中
        selected_logits_list.append(sample_logits)

    return selected_logits_list