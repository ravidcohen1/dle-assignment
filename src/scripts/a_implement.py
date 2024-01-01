import sys

from model.transformer import get_model_from_config
from utils.config_utils import load_config

if __name__ == "__main__":
    config = load_config()
    transformer = get_model_from_config(config)
    print(transformer)
    number_of_params = sum(p.numel() for p in transformer.parameters())
    print("num params:", number_of_params)
    parameters_dtype = transformer.transformer_layers[0].mlp.fc1.weight.dtype
    print("dtype:", parameters_dtype)
    model_ram_size = number_of_params * 4 / 1024 / 1024
    print(f"model RAM size (MB): {model_ram_size:.2f}")
