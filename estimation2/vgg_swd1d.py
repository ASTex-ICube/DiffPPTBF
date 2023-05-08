import os

# JAX imports
import jax.numpy as jnp
from jax import random

# VGG-19 import
from models.vgg import VGG19

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

# Number of random directions for SWD1D projections
num_dir = 100

# Compute random directions
def random_directions(rng, channels):
	directions = [random.normal(rng[i], shape=(num_dir, c)) for (i, c) in enumerate(channels)]
	norm = [jnp.reshape(jnp.sqrt(jnp.sum(jnp.square(d), axis=-1)), (num_dir, 1)) for (i, d) in enumerate(directions)]
	directions = [jnp.divide(d, n) for i, (d, n) in enumerate(zip(directions, norm))]
	return directions

# SWD1D computation of projected feature vectors
def Slicing(F, directions):
	# Project each pixel feature onto directions
	proj = jnp.dot(F, directions.T)
	# Flatten pixel indices
	N, H, W, NDir = proj.shape
	proj_flatten = jnp.reshape(proj, (H*W, NDir))
	# Sort projections for each direction
	return jnp.sort(proj_flatten, axis=0)

# VGG-19 pre-trained on ImageNet (i.e. images with 3 color channels)
def define_model(layers):
	vgg19 = VGG19(output='activations', pretrained='imagenet', include_head=False, num_layers=layers)
	init_rngs = {'params': random.PRNGKey(1), 'dropout': random.PRNGKey(2)}
	return vgg19, init_rngs

# Computation of SWD1D projected feature vectors for the
# selected VGG-19 layers
def VGG_SWD1D(model, init_rngs, x0, x1, x2, directions):
	x0_centered = x0 - jnp.mean(x0)
	x1_centered = x1 - jnp.mean(x1)
	x2_centered = x2 - jnp.mean(x2)
	x = jnp.concatenate((x0_centered, x1_centered,
	x2_centered), axis=-1)
	params = model.init(init_rngs, x)
	out = model.apply(params, x, train=False)
	activations = [out[output] for (i, output) in enumerate(out)]
	s = [Slicing(output, directions[int(i)]) for (i, output) in enumerate(activations)]
	return s


