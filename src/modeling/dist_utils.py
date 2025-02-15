import torch
import torch.distributed as dist
from typing import *

def gather_tensor_without_grad(t: Optional[torch.Tensor]):
    world_size = dist.get_world_size()
    process_rank = dist.get_rank()
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(all_tensors, t)

    # if verbose and dist.get_rank() == 1:
    #     print('==================')
    #     for i, tensor in enumerate(all_tensors):
    #         print(i, tensor.device, tensor.requires_grad)

    all_tensors[process_rank] = t

    # if verbose and dist.get_rank() == 1:
    #     print('==================')
    #     for i, tensor in enumerate(all_tensors):
    #         print(i, tensor.device, tensor.requires_grad)
            
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        dist.all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True)
            for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None

all_gather_with_grad = AllGather.apply

def mismatched_sizes_gather_tensor_with_grad(
    tensor: torch.Tensor, group=None, async_op=False, mismatched_axis=0
):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert dist.is_initialized(), "dist not initialized"
    world_size = dist.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor(
        [tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda"
    )
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    dist.all_gather(
        sizes, mismatched_sizes, group=group, async_op=async_op
    )
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros(
        (
            *tensor.shape[:mismatched_axis],
            max_size,
            *tensor.shape[mismatched_axis + 1 :],
        ),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    all_tensors = [
        torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype)
        for _ in range(world_size)
    ]
    all_gather_with_grad(all_tensors, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert (
            not all_tensors[rank]
            .narrow(
                mismatched_axis,
                sizes[rank],
                padded.shape[mismatched_axis] - sizes[rank],
            )
            .count_nonzero()
            .is_nonzero()
        ), "This would remove non-padding information"
        all_tensors[rank] = all_tensors[rank].narrow(mismatched_axis, 0, sizes[rank])

    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors