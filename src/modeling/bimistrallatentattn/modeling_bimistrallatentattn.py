import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.utils.generic import ModelOutput


from ..bimistral import BiMistralModel
from .configuration_bimistrallatentattn import BiMistralLatentAttnConfig

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class BiMistralLatentAttnModel(BiMistralModel):
    config_class = BiMistralLatentAttnConfig

    def __init__(self, config):
        super().__init__(config)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(
                config.hidden_size, 
                Attention(
                    config.hidden_size, 
                    config.hidden_size, 
                    heads = config.num_cross_heads, 
                    dim_head = config.hidden_size
                ),
                context_dim = config.hidden_size
            ),
            PreNorm(
                config.hidden_size, 
                FeedForward(config.hidden_size)
            ),
        ])
        self.register_parameter("latents", nn.Parameter(torch.randn(config.num_latents_value, config.hidden_size)))
        

    def forward(self, *args, **kwargs):
        base_outputs = super().forward(*args, **kwargs)
        cross_attn, cross_ff = self.cross_attend_blocks
        last_hidden_state = base_outputs.last_hidden_state
        b, *_, device = *last_hidden_state.shape, last_hidden_state.device
        x = repeat(self.latents, 'n d -> b n d', b = b)
        last_hidden_state = cross_attn(last_hidden_state, context = x, mask = None) + last_hidden_state
        last_hidden_state = cross_ff(last_hidden_state) + last_hidden_state

        return ModelOutput(
            last_hidden_state=last_hidden_state
        )