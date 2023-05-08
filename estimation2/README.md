# Optimization Phase #2

### Requirements

For the installation of JAX, please follow the instructions on the
[JAX GitHub page](https://github.com/google/jax#installation). The code has been tested with JAX release v0.4.1 with NVIDIA GPU support (using CUDA and CUDNN). For the other requirements, jut run:

```bash 
pip install -r requirements.txt
```

### Estimate C-PPTBF parameters from a set of input structure maps

The datasets used to produce the results of our paper is available
following this [[link](https://seafile.unistra.fr/f/980f592b7fdc4757953d/?dl=1)]. See the README.txt file contained in the archive for detailed content. Download
the archive and extract its content. If required, change the values of the variables `dir_in` and `dir_out` at the beginning of the file `jax_pptbf_launcher.py`. Then, run:

```bash 
python jax_pptbf_launcher.py
```

Notes:

- JAX compiles code at execution time, and then caches this
code for efficient execution on the GPU (see
[Just In Time Compilation with JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)), this is why the first execution takes a longer
time thatn the subsequent executions.

- Sometimes, the best estimation result is not given by the tiling type
with the highest score in the output of Optimization Phase #1. This is
why we perform an estimation of the parameters for all the tiling
types with a probability that is at least 10%.
