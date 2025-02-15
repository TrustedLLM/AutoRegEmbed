from ..bimistral import BiMistralConfig

from transformers import MistralConfig

class BiMistralLatentAttnConfig(BiMistralConfig):
    model_type = "bimistrallatentattn"

    def __init__(
        self,
        num_latents_value=512,
        num_cross_heads=8,
        **kwargs
    ):
        self.num_latents_value = num_latents_value
        self.num_cross_heads = num_cross_heads
        super().__init__(**kwargs)