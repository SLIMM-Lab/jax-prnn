# Physically Recurrent Neural Networks using JAX

This is a JAX-based implementation of PRNNs. This version accelerates training and inference time by >10x. 
If you are unfamiliar with PRNNs, start with the [main repository](https://github.com/SLIMM-Lab/pyprnn).
This repository follows a very similar structure, and the demonstration uses the same datasets.

## What is different compared to the torch-based implementation?
### Just-In-Time compilation
The main difference in speed is achieved by using [Just-In-Time compilation (JIT)](https://docs.jax.dev/en/latest/jit-compilation.html). 
To enable this, the material model has been heavily modified. Instead of being a class, it consists of pure functions.
This is more difficult to implement, can be challenging to debug, but it results in significantly faster computation times.

During the first pass of a function, it is compiled (which can be significantly slower than normal), making any subsequent calls to that function significantly faster.

### Parameter handling
The network and material parameters are explicitly passed around, following JAX's intended use. 

### Scan replaces for loop
The for loop computing a sequence has been replaced by a [scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) function. 
The main advantage is that this only needs to be compiled once, rather than for all timesteps in the initial sequence, significantly reducing compilation time.

### Minor differences

- The encoder and decoder run only once on the full sequence for the current batch, instead of once per time step. 

- Some network layers are implemented using [einsum notation](https://rockt.ai/2018/04/30/einsum) for clarity (and minor performance).

- Several decoder layers are implemented, which can further reduce the training data requirement. Many more material points can be used, with fewer training samples, without causing overfitting.

- No dataloader is implemented. In the demo, the data is manually split.

- The trainer function is moved from utils to a separate script.

## Demonstration
The prnn-demo.ipynb notebook shows an example of how to train a PRNN.

## Fun-fact
JAX-PRNN is also known as PRRRNN, because it makes your computer go brrr.
