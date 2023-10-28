# Optimization Phase #2

### Requirements

For the installation of JAX, please follow the instructions on the
[JAX GitHub page](https://github.com/google/jax#installation). The code has been tested with JAX release v0.4.1 with NVIDIA GPU support (using CUDA and CUDNN). For the other requirements, jut run:

```bash 
pip install -r requirements.txt
```

#### Update (October 28, 2023)

Command lines to install JAX 0.4.19 with CUDA 11 and CUDNN 8.6 in an Anaconda env with pip:

```bash 
conda create --name jax
conda activate jax
conda install python=3.10
python -m pip install jax==0.4.19 https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.19+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl
python -m pip install h5py requests tdqm optax==0.1.7 pyopencl==2023.1.4 flax==0.7.5 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cudnn-cu11==8.6.0.163 nvidia-cufft-cu11==10.9.0.58 nvidia-cusolver-cu11==11.4.1.48 nvidia-cuda-cupti-cu11==11.8.87 nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-nvcc-cu11==11.8.89 nvidia-cusparse-cu11==11.7.5.86 nvidia-nccl-cu11==2.16.2 optax pyopencl 
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
[Just In Time Compilation with JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)). The compiled code is not cached on-disk (which is
actually not supported by JAX, as far as we know) and is re-compiled
at each execution of the code.

- Sometimes, the best estimation result is not given by the tiling type
with the highest score in the output of Optimization Phase #1. This is
why we perform an estimation of the parameters for all the tiling
types with a probability that is at least 10%.
