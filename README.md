# dle-assignment

To run this project, you need to install the requirements in `requirements.txt` (use python 3.10) and download the weights and put them in the `weights` folder.

## Implementation

Running `python scripts/a_implement.py` prints the number of the parameters and the size of the model in MB.

Each parameter is a 32-bit float, so the size of the model in bytes is the number of parameters times 4.

## Latency

Running `python scripts/b_measure_latency.py` measure the affect of all configurations on the latency of the model. The plots are saved in the `plots` folder.

The length of the input increases the latency linearly, while the vocabulary size has no effect on the latency.

In terms of architecture configurations, all of them increases the latency linearly (in the measured ranges), 
but the most influential parameters are the number of attention layers and the number of attention heads. 
![num_transformer_layers_latency.png](plots%2Fnum_transformer_layers_latency.png)

## Supplied State Dict

Running `python scripts/c_supplied_state_dict.py` loads the supplied state dict, load the model with it 
(using [keys_mapping.json](weights%2Fkeys_mapping.json)), and verify
that the model is getting the same outputs for the given examples. 

I struggled to load the state dict to the model with the default configs, so I had to change the configs to what seemed
right - a single attention head and hidden size of 128. Unsurprisingly, the results did not match the given ones. 


## Architecture Changes

1. Changing the epsilon of the layer norm from 1e-5 to 1e-4 should definitely not affect the latency, and it did not.
Also, the logits seems to not get affected by this change, probably because the epsilon is insignificant compared to the layers variance.
```angular2html
epsilon: 1e-5, latency: 78.6 ms
epsilon: 1e-4, latency: 77.9 ms
Latency diff: -0.7 ms
RMSE (Root Mean Squared Error): 0.0000
Mean Absolute Difference: 0.0000
Standard Deviation of Absolute Differences: 0.0000
Percentage First Greater: 0.00%
```
2. Removing the scaling of the attention product might decrease the latency insignificantly. 
The output logits are different but I did not observe a change towards one direction (i.e. all logits increase or decrease). I do know that we need this scale to prevent attention logit growth.
```angular2html
scale_attention: True, latency: 78.6 ms
scale_attention: False, latency: 76.1 ms
Latency diff: -2.5 ms
RMSE (Root Mean Squared Error): 3.5280
Mean Absolute Difference: 2.7729
Standard Deviation of Absolute Differences: 2.1811
Percentage First Greater: 49.98%
```
3. Approximating GELU supposed to decrease the latency, but it did not. Regarding the logits, I did not observe any change.
```angular2html
gelu_approximation: none, latency: 78.6 ms
gelu_approximation: tanh, latency: 88.6 ms
Latency diff: 10.0 ms
RMSE (Root Mean Squared Error): 0.0008
Mean Absolute Difference: 0.0006
Standard Deviation of Absolute Differences: 0.0005
Percentage First Greater: 49.36%
```

4. I don't see why changing the residual connection from the attention output to the input should affect much the forward pass as 
the second line is still dependent on the first. However, we observe almost 4% speedup. The logits are different because it's a different architecture but 
the question is how it will affect the training, I don't have an intuition about it.
```angular2html
transformer_forward_alternative: none, latency: 79.1 ms
transformer_forward_alternative: 1, latency: 76.3 ms
Latency diff: -2.9 ms
RMSE (Root Mean Squared Error): 1.2523
Mean Absolute Difference: 0.9369
Standard Deviation of Absolute Differences: 0.8310
Percentage First Greater: 49.59%
```

5. Dropping the residual connection might make the model faster (and we do observe ~3% speedup) but we know these connections are important for training stability.
```angular2html
transformer_forward_alternative: none, latency: 79.1 ms
transformer_forward_alternative: 2, latency: 76.6 ms
Latency diff: -2.6 ms
RMSE (Root Mean Squared Error): 23.5129
Mean Absolute Difference: 18.0500
Standard Deviation of Absolute Differences: 15.0684
Percentage First Greater: 50.02%
```

6. In the last change, the MLP doesn't depend on the attention output so it can be parallelized. In my implementation, on my hardware, it did not affect the latency.
```angular2html 
transformer_forward_alternative: none, latency: 79.1 ms
transformer_forward_alternative: 3, latency: 79.4 ms
Latency diff: 0.3 ms
RMSE (Root Mean Squared Error): 0.1682
Mean Absolute Difference: 0.1258
Standard Deviation of Absolute Differences: 0.1116
Percentage First Greater: 52.06%
```

## Matrix Multiplication

[parallel_mat_mul.py](src%2Fmat_mul%2Fparallel_mat_mul.py) is my implementation for parallel matrix multiplication.
This script runs correctness validation and compare the speedup to the naive implementation. I tried to make it recursive 
but I got this error `AssertionError: daemonic processes are not allowed to have children`.
