import json
from typing import Any

import numpy as np
import torch

from model.transformer import get_model_from_config
from utils.config_utils import load_config
from utils.latency_utils import measure_latency_with_cache


def inference_example(model, examples: list[dict]):
    """
    Infer the logits of the examples using the model defined in the config.
    """
    output_logits = []
    for example in examples:
        input_ids, ouput_logits = example["input_ids"], example["logits"]
        pred_logits = model(input_ids.unsqueeze(0))
        output_logits.append(pred_logits.detach().cpu().numpy())
    return output_logits


def compute_diff(before_logits, after_logits):
    before_logits = np.concatenate(before_logits)
    after_logits = np.concatenate(after_logits)

    # Ensure the arrays have the same shape
    if before_logits.shape != after_logits.shape:
        raise ValueError("The two logits arrays must have the same shape")

    # Calculate differences
    differences = before_logits - after_logits

    # Metrics
    mean_absolute_diff = np.mean(np.abs(differences))
    std_dev_of_abs_diff = np.std(np.abs(differences))
    percent_first_greater = np.mean(before_logits > after_logits) * 100
    rmse = np.sqrt(np.mean(differences**2))

    # Printing the results
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"Mean Absolute Difference: {mean_absolute_diff:.4f}")
    print(f"Standard Deviation of Absolute Differences: {std_dev_of_abs_diff:.4f}")
    print(f"Percentage First Greater: {percent_first_greater:.2f}%")


def measure_a_single_change(
    config: dict, variable_name: str, variable_value: Any, examples: list[dict]
):
    """
    Measure the effect of a variable on the latency of the model and logits.
    :param config: a dictionary containing the model's config
    :param variable_name: the name of the variable to measure
    :param variable_value: the value to test
    """
    before_latency = measure_latency_with_cache(config)
    model1 = get_model_from_config(config)

    before_logits = inference_example(model1, examples)
    original_value = config[variable_name]
    config[variable_name] = variable_value
    model2 = get_model_from_config(config)
    model2.load_state_dict(model1.state_dict())

    after_latency = measure_latency_with_cache(config)
    after_logits = inference_example(model2, examples)

    print(f"{variable_name}: {original_value}, latency: {before_latency:.1f} ms")
    print(f"{variable_name}: {variable_value}, latency: {after_latency:.1f} ms")
    print(f"Latency diff: {after_latency - before_latency:.1f} ms")

    compute_diff(before_logits, after_logits)

    config[variable_name] = original_value


if __name__ == "__main__":
    config = load_config()
    examples = torch.load("weights/example_outputs.pt")

    # measure_a_single_change(config, "epsilon", "1e-4", examples)
    # print()
    # measure_a_single_change(config, "scale_attention", False, examples)
    # print()
    # measure_a_single_change(config, "gelu_approximation", 'tanh', examples)
    print()
    measure_a_single_change(config, "transformer_forward_alternative", "1", examples)
    print()
    measure_a_single_change(config, "transformer_forward_alternative", "2", examples)
    print()
    measure_a_single_change(config, "transformer_forward_alternative", "3", examples)
