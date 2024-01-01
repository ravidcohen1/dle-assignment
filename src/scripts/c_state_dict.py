import json
from pathlib import Path

import torch

from src.model.transformer import get_model_from_config
from utils.config_utils import load_config

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"


def load_model_weights(model, weights_path, key_mapping_path):
    """
    Load a model's weights from a file.
    :param model: the model to load the weights into
    :param weights_path: the path to the weights file
    :param key_mapping_path: the path to the key mapping file
    """
    weights = torch.load(weights_path, map_location=torch.device("cpu"))
    new_weights = {}
    with open(key_mapping_path, "r") as f:
        key_mapping = json.load(f)
    for k, v in key_mapping.items():
        if k.endswith("mlp.down_proj.weight") or k.endswith("mlp.up_proj.weight"):
            new_weights[v] = weights.pop(k).T
        elif isinstance(v, list):
            qkv_weight = weights.pop(k)
            split_size = qkv_weight.size(-1) // 3
            query, key, value = torch.split(qkv_weight, split_size, dim=-1)
            new_weights[v[0]] = query
            new_weights[v[1]] = key
            new_weights[v[2]] = value
        else:
            new_weights[v] = weights.pop(k)

    model_keys = set(model.state_dict().keys())
    mapped_keys = set(new_weights.keys())
    missing_keys = model_keys - mapped_keys
    unhandled_keys = weights.keys()

    if len(missing_keys) > 0:
        print("missing keys:")
        for k in sorted(list(missing_keys)):
            print(k)
    if len(unhandled_keys) > 0:
        print("unhandled keys:")
        print(unhandled_keys)
    model.load_state_dict(new_weights)
    return model


def verify_model_on_examples(model, examples_path):
    """
    Verify that a model can generate the examples in the given file.
    :param model: the model to verify
    :param examples_path: the path to the examples file (.pt)
    """
    examples = torch.load(examples_path)
    for example in examples:
        input_ids, ouput_logits = example["input_ids"], example["logits"]
        pred_logits = model(input_ids.unsqueeze(0))
        if not torch.allclose(pred_logits, ouput_logits, atol=1e-3):
            print("failed to generate example")
        else:
            print("example generated successfully")


if __name__ == "__main__":
    config = load_config()
    config["num_heads"] = 1
    config["attention_hidden_dim"] = 256

    model = get_model_from_config(config)
    load_model_weights(
        model, WEIGHTS_DIR / "dummy_gpt2_model.pt", WEIGHTS_DIR / "keys_mapping.json"
    )
    verify_model_on_examples(model, WEIGHTS_DIR / "example_outputs.pt")
