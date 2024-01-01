from pathlib import Path
from typing import Any

from src.utils.config_utils import load_config
from src.utils.latency_utils import measure_latency_with_cache, plot_latencies

PLOTS_DIR = Path(__file__).parent.parent.parent / "plots"


def measure_config_effect(config: dict, variable_name: str, variable_values: list[Any]):
    """
    Measure the effect of a variable on the latency of the model.
    :param config: a dictionary containing the model's config
    :param variable_name: the name of the variable to measure
    :param variable_values: the values to test
    """
    latencies = []
    original_value = config[variable_name]

    for value in variable_values:
        config[variable_name] = value
        t = measure_latency_with_cache(config)
        print(f"{variable_name}: {value}, latency: {t:.1f} ms")
        latencies.append(t)

    config[variable_name] = original_value
    plot_dest = PLOTS_DIR / f"{variable_name}_latency.png"
    plot_latencies(
        latencies,
        variable_name,
        variable_values,
        title=f"{variable_name} effect on latency",
        dest=plot_dest,
    )


def measure_default_config(config):
    t = measure_latency_with_cache(config)
    print(f"latency with default configs: {t:.1f} ms")


if __name__ == "__main__":
    config = load_config()
    measure_default_config(config)
    measure_config_effect(config, "input_len", [256, 512, 1024, 2048, 4096])
    measure_config_effect(config, "vocab_size", [256, 512, 1024, 2048, 4096])

    measure_config_effect(config, "embedding_dim", [32, 64, 128, 256, 512, 1024])
    measure_config_effect(config, "attention_hidden_dim", [32, 64, 128, 256, 512, 1024])
    measure_config_effect(config, "num_heads", [1, 2, 4, 8])
    measure_config_effect(config, "num_transformer_layers", [1, 2, 3, 4, 5, 6])
    measure_config_effect(config, "mlp_hidden_dim", [32, 64, 128, 256, 512, 1024])
