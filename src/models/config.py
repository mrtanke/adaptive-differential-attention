from dataclasses import dataclass
@dataclass
class ModelConfig:
    vocab_size:int=50257
    d_model:int=256
    n_layers:int=6
    n_heads:int=8
    ffn_dim:int=704
    attention_type:str="standard"
