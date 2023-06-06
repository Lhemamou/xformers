import torch
from xformers.factory.model_factory import xFormer, xFormerConfig
from xformers.components.attention.attention_patterns import (
    global_token_pattern,
        local_1d_pattern,
)

from xformers.components.attention import (
    maybe_sparsify,
)
EMB = 768  # Roberta base has 768 hidden dimensions
SEQ = 8192  # Maximum sequence length 
BATCH = 1
VOCAB = 50265  # Roberta uses a byte-level BPE vocabulary of size 50265


# Building Longformer Matrix

global_mask=torch.zeros(SEQ)
global_mask[0:25]=1 # position of global token
global_mask=global_mask.bool().unsqueeze(-1)
attention_mask = global_token_pattern(global_mask[:, 0])
window_size=511 # Window size of the local attention

local_mask = local_1d_pattern(global_mask.shape[0], window_size)
combined_mask = local_mask + attention_mask
combined_mask = maybe_sparsify(combined_mask)  # Sparsify the mask
combined_mask=combined_mask.to(torch.device(0))


my_config_lf = [
    {
        "reversible": False,  
        "block_type": "encoder",
        "num_layers": 12,  # Roberta base has 12 layers
        "dim_model": EMB,
        "residual_norm_style": "pre",  
        "position_encoding_config": {
            "name": "vocab",  # Roberta uses learned position embeddings
            "dim": EMB,
            "seq_len":SEQ,
            "vocab_size":VOCAB
        },
        "multi_head_config": {
            "num_heads": 12,  # Roberta base has 12 attention heads
            "residual_dropout": 0.1,
            "attention": {
                "name": "sparsemask",  # Roberta uses scaled dot product attention
                "dropout": 0.1,
                "causal": False,  # Roberta is not causal
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": 0.1,
            "activation": "gelu",  # Roberta uses GELU activation
            "dim_feedforward": 3072,  # Roberta base has 3072 feedforward dimensions
            "hidden_layer_multiplier":1
        },
    }
]
config = xFormerConfig(my_config_lf)
model = xFormer.from_config(config)

# Test with dummy inputs
# Note: Roberta uses byte-level BPE tokenization, which this code doesn't handle
x = (torch.rand((BATCH, SEQ)) * VOCAB).abs().to(torch.int).to('cuda:0')
model=model.to('cuda:0')
y = model(src=x, tgt=x,encoder_att_mask=combined_mask) # Here we provide the mask.
loss=y.mean()
loss.backward()
