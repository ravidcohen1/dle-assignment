import json
import time

import numpy as np
import torch
from joblib import Memory
from matplotlib import pyplot as plt

from src.model.transformer import get_model_from_config

memory = Memory("cache", verbose=0)


def measure_latency_with_cache(config: dict) -> float:
    """
    Measure latency of a model with a given config.
    :param config: a dictionary containing the model's config
    :return: latency in milliseconds
    """

    @memory.cache
    def get_model_and_measure_latency(config):
        config = json.loads(config)
        transformer = get_model_from_config(config)
        x = torch.randint(0, config["vocab_size"], (1, config["input_len"]))
        t = measure_latency(transformer, x)
        return t

    config = json.dumps(config, sort_keys=True)
    return get_model_and_measure_latency(config)


def measure_latency(model, x, warmup_steps=10, n_steps=100):
    for _ in range(warmup_steps):
        model(x)
    start_time = time.time()
    for _ in range(n_steps):
        model(x)
    end_time = time.time()
    return 1000 * (end_time - start_time) / n_steps


def plot_latencies(latencies, x_label, x_values, title, dest=None):
    plt.plot(x_values, latencies)
    plt.xlabel(x_label)
    plt.ylabel("Latency (ms)")
    plt.ylim(bottom=0)
    plt.title(title)
    # linear fit and print slope
    m, b = np.polyfit(x_values, latencies, 1)
    plt.text(x_values[0], 10, f"slope of linear fit: {m:.2f}")
    plt.savefig(dest)
    plt.show()
