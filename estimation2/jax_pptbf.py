'''
Guillaume Baldi, Rémi Allègre, Jean-Michel Dischler.
Differentiable Point Process Texture Basis Functions for inverse
procedural modeling of cellular stochastic structures,
Computers & Graphics, Volume 112, 2023, Pages 116-131,
ISSN 0097-8493, https://doi.org/10.1016/j.cag.2023.04.004.
LGPL-2.1 license
'''

import math
import os
import time
import sys
import json
import argparse
from functools import partial
import numpy as np
from PIL import Image

# JAX and Optax imports
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, lax, random, custom_jvp
import optax

# PPTBF OpenCL module imports
from pptbf_opencl_module import cl_pptbf_0, cl_agauss

# VGG-19 and SWD1D module imports
from vgg_swd1d import define_model, random_directions, VGG, VGGConcat

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

# Memory allocation, see the following page for details:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'

# ----- JAX addons -----

# Safe jnp.sqrt function, see this page for details:
# https://github.com/google/jax/issues/5798
@custom_jvp
def safe_sqrt(x):
    return jnp.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(xpack, vpack):
	x, = xpack
	v, = vpack
	f = safe_sqrt(x)
	df = v * jnp.where( x <= 0.0,
						0.0,
						0.5/jnp.sqrt(x) )
	return f, df

# Make hashable array type, after:
# https://github.com/google/jax/issues/4572#issuecomment-709677518
def some_hash_function(x):
  return int(jnp.sum(x))

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

def gnool_jit(fun, static_array_argnums=()):
  @partial(jit, static_argnums=static_array_argnums)
  def callee(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = args[i].val
    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = HashableArrayWrapper(args[i])
    return callee(*args)

  return caller

# ----- PRNG -----

# PRNG, see the following page for details:
# https://jax.readthedocs.io/en/latest/jax.random.html
rng = random.PRNGKey(4321)

# ----- Image size -----

# Image size along 1 dimension
im_size = 256

# ----- Parsing of arguments -----

parser = argparse.ArgumentParser(description="arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir_in", required=True, help="input directory")
parser.add_argument("--dir_out", required=True, help="output directory")
parser.add_argument("--input", required=True, help="input structure")
parser.add_argument("--params", help="json file initial params")
parser.add_argument("--write_images", help="set to 1 to write images")
parser.add_argument("--tt", help="tiling type in 0-13")
parser.add_argument("--ttnfp", help="numbers of feature points")

args = vars(parser.parse_args())

# Directory containing input structure maps and fake (reconstructed)
# C-PPTBF images as output of the pip2pix stage
dir_in = args['dir_in']
# Directory for output data (images and C-PPTBF estimated
# parameters)
dir_out = args['dir_out']
input_structure_name = args['input']

# Fake (reconstructed) C-PPTBF image filename
input_image_name = (dir_in + '/' + input_structure_name +
'_seg_scrop_pptbf_fake_B.png')
# Input structure map filename
input_image_t_name = (dir_in + '/' + input_structure_name +
'_seg_scrop_pptbf_real_A.png')

# Binary value indicating if the images are saved
# at each optimization step (1) or not (0)
write_images = int(args['write_images'])

# Open "tiling type number of feature points"
# data file required to correctly match an estimated
# number of feature points in a (possibly fake) C-PPTBF
# image with a scaling factor
with open(args['ttnfp'], 'r') as f:
  ttnfp = json.load(f)

# Read file with initial parameters estimated
# in Optimization stage #1
with open(args['params'], 'r') as f:
  data = json.load(f)

# The file with initial parameters may contain
# predictions for multiple inputs: The parameters
# that correspond to the loaded input structure are
# extracted 
data_structure_name = []
for i in range(0, len(data['predictions'])):
	data_structure_name.append(data['predictions'][i][0])

# Index of the parameters that correspond to the
# loaded input structure
ix = data_structure_name.index(input_structure_name)

# ----- Tiling type and zoom factor -----

# If the tiling type is not provided as an argument of the
# command line, it is read in the loaded parameter file
if args['tt'] is None:
	tt = data['predictions'][ix][1][0]
else:
	tt = int(args['tt'])

# Estimated angle of rotation
alpha = data['predictions'][ix][1][2]

# Estimated number of feature points
nfp = data['predictions'][ix][1][3]
# Maximum 600-700 detected feature points is supported
# for an input structure map because the region covered by the
# generated point set is larger by a factor or 2.0 to account
# for rotation and zoom corrections, which may yield 1000
# feature points to process during Phase #2
nfp = np.minimum(nfp, 600)

# Computation of the zoom value that corresponds to the
# estimated number of feature points, that depends on the
# tiling type
nfp_tab = np.array(ttnfp[tt]['nfp'])
zoom_tab = np.array(ttnfp[tt]['zoom'])
zoom_input = np.interp(nfp, nfp_tab, zoom_tab)
# Maximum zoom (limitation to 1000 feature points)
zoom_max = np.interp(1000.0, nfp_tab, zoom_tab)
# Zoom factor during Phase #2 allowed for corrections
zoom_factor = zoom_max / zoom_input
if zoom_factor > 2.0:
	zoom_factor = 2.0
zoom = zoom_factor * zoom_input
zoom_zeta = zoom_factor / 1.5

# Image size used for the generatiion of the set of
# feature points from the OpenCL generator code
im_size_pptbf_0 = int(math.floor(im_size * zoom_factor))
# im_size_pptbf_0 must be even for the im_size*im_size
# window computation
if im_size_pptbf_0 % 2:
	im_size_pptbf_0 += 1

print("tt", tt, "nfp", nfp, "/ zoom", zoom_input, "max",
zoom_max, "factor", zoom_factor, "zeta", zoom_zeta, "size",
im_size_pptbf_0)

# ----- Initial continuous parameter values -----

# Normalized (set between 0 and 1) initial values for continuous
# parameters only considered in Phase #2
# - Deformation amplitude
amp_i = 0.0
# - Angle for rotation correction (secondary rotation)
theta_i = 0.5
# - Scaling along X and Y for scaling correction
# (secondary zoom)
zetax_i = 0.5
zetay_i = 0.5

# Other continuous parameters read from the input parameter file
jitter_i = data['predictions'][ix][1][1]
jitter_i = (jitter_i - 0.01) / (0.5 - 0.01)
normblend_i = data['predictions'][ix][1][4]
wsmooth_i = data['predictions'][ix][1][5]
winfeatcorrel_i = data['predictions'][ix][1][6]
feataniso_i = data['predictions'][ix][1][7] / 5.0
sigcos_i = data['predictions'][ix][1][8] / 10.0
deltaorient_i = data['predictions'][ix][1][9] / (0.5 * jnp.pi)
larp_i = data['predictions'][ix][1][10]	

# ----- Read input images -----

# Read input structure map
print("Read input structure map", input_image_t_name)
input_image_t = Image.open(input_image_t_name)
if input_image_t.size != (im_size, im_size):
	input_image_t = input_image_t.resize((im_size, im_size),
	resample=Image.NEAREST)
	print("Image resized to "+str(im_size)+"*"+str(im_size))
input_image_t = np.asarray(input_image_t).astype(float) / 255.0
if len(input_image_t.shape) == 3:
	input_image_t = input_image_t[:,:,0]
# Binarize image (if not a binary image)
input_image_t_rebin = 1 - (input_image_t < 0.8)
num_zero = np.count_nonzero(input_image_t_rebin == 0.0)
num_zero_input = np.count_nonzero(input_image_t == 0.0)
if num_zero != num_zero_input:
	print("Warning: input image was not binary")
	input_image_t = input_image_t_rebin
	input_image_t_rebin = input_image_t_rebin * 255.0
	input_image_t_rebin = Image.fromarray(input_image_t_rebin.astype(np.ubyte))
	input_image_t_rebin = input_image_t_rebin.convert('L')
	input_image_t_rebin.save(
		dir_in + '/' + input_structure_name + '_seg_scrop_pptbf_real_A.png'
	)

print("Number of zero pixels:", num_zero)
T = float(num_zero) / (im_size*im_size)
print("Corresponding threshold value:", T)
input_image_t = np.reshape(input_image_t, im_size*im_size)
input_image_t = jnp.array(input_image_t)

print("Read fake C-PPTBF image", input_image_name)
input_image = Image.open(input_image_name)
if input_image.size != (im_size, im_size):
	input_image = input_image.resize((im_size, im_size),
	resample=Image.NEAREST)
	print("Image resized to "+str(im_size)+"*"+str(im_size))
input_image = np.asarray(input_image).astype(float) / 255.0
if len(input_image.shape) == 3:
	input_image = input_image[:,:,0]
input_image = np.reshape(input_image, im_size*im_size)
min = np.min(input_image)
max = np.max(input_image)
input_image = input_image - min
input_image = input_image / (max - min)
input_image = jnp.array(input_image)

# Build a 3-channel image from the structure map and the
# fake C-PPTBF image (as described in Section 4.3 of the paper)
input_concat = jnp.stack((input_image, 1.0 - input_image,
input_image_t), axis=-1)

# ----- Some fixed C-PPTBF parameters -----

# Offset in PPTBF space
tx = 3.0
ty = 5.0
# Footprint of Gaussian for cellular window
footprint = 1.0
if tt >= 8:
	footprint = 0.4

# ----- PPTBF pass to compute feature points -----

print("Compute feature points for image size", im_size_pptbf_0)
start = time.time()
# Run OpenCL generator code
# Output:
# - p_np: feature points
# - npp_np: number of closest neighbors
# - fbm_np: deformation field
p_np, npp_np, fbm_np = cl_pptbf_0(im_size_pptbf_0, tx,
ty, zoom, alpha, tt)
# - g_np: data for feature function
g_np = cl_agauss(p_np)
end = time.time()
print('PPTBF pass: {:.4f} s'.format(end-start))

# Conversion of np arrays to jnp arrays
p = jnp.array(p_np)
npp = jnp.array(npp_np)
# Maximum number of nearest neighbors for feature points (18 is in
# general sufficient, as prescribed in the original PPTBF code)
#nppm = jnp.max(npp)
nppm = 18
g = jnp.array(g_np)
fbm = jnp.array(fbm_np)

# Center of rendered image in PPTBF coordinates
cpptbf_p = jnp.array([
	100.0 + ((0.5 + tx) * jnp.cos(-alpha) + (0.5 + ty) * jnp.sin(-alpha)) * zoom,
	100.0 + (-(0.5 + tx) * jnp.sin(-alpha) + (0.5 + ty) * jnp.cos(-alpha)) * zoom
])

cpptbf_t = jnp.array([
	100.0 + ((0.5 + tx) * jnp.cos(-alpha) + (0.5 + ty) * jnp.sin(-alpha)) * zoom,
	100.0 + (-(0.5 + tx) * jnp.sin(-alpha) + (0.5 + ty) * jnp.cos(-alpha)) * zoom,
	0.0, 0.0
])

# Point indices for aniotropic Gaussian (feature function) values
P = jnp.arange(0, p.shape[0])

# Pixel indices for computing distances to Voroinoi cell borders
diff_size = (im_size_pptbf_0 - im_size) // 2
Xd_x = jnp.arange(diff_size, im_size_pptbf_0 - diff_size)
Xd_y = jnp.arange(diff_size, im_size_pptbf_0 - diff_size)
Xd_x, Xd_y = jnp.meshgrid(Xd_x, Xd_y)
Xd_x = jnp.reshape(Xd_x, im_size * im_size)
Xd_y = jnp.reshape(X_y, im_size * im_size)
Xd = Xd_y * im_size_pptbf_0 + Xd_x

# Pixel indices for PPTBF computation
X = jnp.arange(0, im_size * im_size)

# Pixel indices for computation of Voronoi neighbors
Xn_x = jnp.arange(1, im_size + 1)
Xn_y = jnp.arange(1, im_size + 1)
Xn_x, Xn_y = jnp.meshgrid(Xn_x, Xn_y)
Xn_x = jnp.reshape(Xn_x, im_size * im_size)
Xn_y = jnp.reshape(Xn_y, im_size * im_size)
Xn = Xn_y * (im_size + 2) + Xn_x

# ----- Differentiable C-PPTBF functions -----

# Compute values defined per feature point and based on the
# values of winfeatcorrel, deltaorient and jitter
@partial(gnool_jit, static_array_argnums=(0,1,2,3,4))
def computePointValues(p, g, zoom, tx, ty, winfeatcorrel_s,
	deltaorient_s, jitter_s, i):
	
	# Actural parameter values computed from the normalized values
	winfeatcorrel = winfeatcorrel_s
	deltaorient = deltaorient_s * (-0.1 * jnp.pi) + (1.0 - deltaorient_s) * (0.1 * jnp.pi)
	jitter = (1.0 - jitter_s) * 0.01 + jitter_s * 0.5

	# Apply jittering
	xj = p[i, 0] * jitter + p[i, 2] * (1.0 - jitter)
	yj = p[i, 1] * jitter + p[i, 3] * (1.0 - jitter)

	lx = p[i, 2] + g[i, 0] * 0.99 * p[i, 4]
	ly = p[i, 3] + g[i, 1] * 0.99 * p[i, 5]
	lx = winfeatcorrel * xj + (1.0 - winfeatcorrel) * lx
	ly = winfeatcorrel * yj + (1.0 - winfeatcorrel) * ly
	angle = deltaorient * g[i, 2]

	return jnp.array([
		lx, ly, angle, xj, yj
	])

# Smooth maximum function
def smoothmax(x1, x2):
	return 0.5*((x1+x2)+safe_sqrt((x1-x2)*(x1-x2)+1e-3))

# Compute distances to feature points with larp
def distanceLarp(x, y, p_in, t_in, larp_s):
	# Compute distances to feature points
	t = jnp.full(shape=(p_in.shape[0], 2), fill_value=jnp.array([x, y]))
	d = safe_sqrt(jnp.sum(jnp.square(t - p_in), axis=-1))
	
	# For a point given by (x, y), compute the distances to
	# feature points using the tiling distance
	ddx = t[:,0] - p_in[:,0]
	ddy = t[:,1] - p_in[:,1]
	# if ddx < 0 then ex1 > 0 and ex2 < 0: the max is ex1
	# if ddx > 0 then ex1 < 0 and ex2 > 0: the max is ex2
	# Note: the cells are not degenerated as a property of
	# the tiling (so, the denominator is strictly positive)
	ex1 = (-ddx) / (p_in[:,0] - t_in[:,0] + t_in[:,2])
	ex2 = ddx / (t_in[:,0] + t_in[:,2] - p_in[:,0])
	ex = smoothmax(ex1, ex2)
	ey1 = (-ddy) / (p_in[:,1] - t_in[:,1] + t_in[:,3])
	ey2 = ddy / (t_in[:,1] + t_in[:,3] - p_in[:,1])
	ey = smoothmax(ey1, ey2)
	e = smoothmax(ex, ey)

	# Interpolate distances
	larp = larp_s
	d_larp = (1.0 - larp) * d + larp * e
	return d_larp

# Compute per-pixel distances to feature points
@partial(gnool_jit, static_array_argnums=(1,2,3,4,5,7))
def voronoiLarpPix(i, im_size_pptbf_0, alpha, zoom, tx, ty, p_in,
	t_in, fbm, larp_s, theta_s, zetax_s, zetay_s, amp_s):
	xx0 = (i % im_size_pptbf_0) / im_size_pptbf_0
	yy0 = (i // im_size_pptbf_0) / im_size_pptbf_0	
	
	# Secondary rotation
	theta = (1.0-theta_s)* (-0.2 * jnp.pi)+ theta_s*(0.2*jnp.pi)
	
	# Secondary zoom
	zetax = (1.0 - zetax_s) * (2.0 - zoom_zeta) + zetax_s * zoom_zeta
	zetay = (1.0 - zetay_s) * (2.0 - zoom_zeta) + zetay_s * zoom_zeta

	xx0 -= 0.5
	yy0 -= 0.5
	xx = (xx0 * jnp.cos(theta) - yy0 * jnp.sin(theta))
	yy = (xx0 * jnp.sin(theta) + yy0 * jnp.cos(theta))
	xx += 0.5
	yy += 0.5

	ppx = xx + tx
	ppy = yy + ty

	# Deformation
	amp = amp_s * 0.1
	ppx = ppx + amp * fbm[2 * i    ]
	ppy = ppy + amp * fbm[2 * i + 1]

	x = 100.0 + (ppx * jnp.cos(-alpha) + ppy * jnp.sin(-alpha)) * zoom
	y = 100.0 + (-ppx * jnp.sin(-alpha) + ppy * jnp.cos(-alpha)) * zoom
	
	# Can't use zoom * zetax/y because the point set is precomputed
	p_in_s = p_in - cpptbf_p
	p_in_s = p_in_s * jnp.array([zetax, zetay])
	t_in_s = t_in - cpptbf_t
	t_in_s = t_in_s * jnp.array([zetax, zetay, zetax, zetay])
	p_in_s += cpptbf_p
	t_in_s += cpptbf_t

	# Compute distances to feature points with larp

	d_larp = distanceLarp(x, y, p_in_s, t_in_s, larp_s)
	# Labels (index) of closest feature points
	l = jnp.argsort(d_larp)
	ddx = x - p_in_s[l[0],0]
	ddy = y - p_in_s[l[0],1]
	a = jnp.arctan2(-ddy, -ddx) + jnp.pi
	norm = safe_sqrt(ddx * ddx + ddy * ddy)
	ddx /= norm
	ddy /= norm

	#return l, d_larp[l[0]], a, x, y, p_in_s[l[0],0], p_in_s[l[0],1], ddx, ddy, norm
	# Returning l[:36] instead of l reduces the memory usage when
	# in PREALLOCATE is false
	return l[:36], d_larp[l[0]], a, x, y, p_in_s[l[0],0], p_in_s[l[0],1], ddx, ddy, norm

# Compute Voronoi diagram
def computeVoronoiLarp(X, im_size_pptbf_0, alpha, zoom, tx, ty,
	p_in, t_in, fbm, larp, theta_s, zetax_s, zetay_s, amp_s):
	voronoi = vmap(voronoiLarpPix, in_axes=(0, None, None, None,
	None, None, None, None, None, None, None, None, None, None))
	v = voronoi(X, im_size_pptbf_0, alpha, zoom, tx, ty, p_in,
	t_in, fbm, larp, theta_s, zetax_s, zetay_s, amp_s)
	return v

neigh = jnp.array([0, -(im_size+2)-1, -(im_size+2), -(im_size+2)+1,
-1, 1, (im_size+2)-1, (im_size+2), (im_size+2)+1])

def voronoiNeighPix(v, i, n):
	return v[i+n]

# Compute Voronoi neighbors
def computeVoronoiNeigh(v):
	voronoiNeigh = vmap(voronoiNeighPix, in_axes=(None, 0, None))
	voronoiNeighN = vmap(voronoiNeigh, in_axes=(None, None, 0))
	vn = jit(voronoiNeighN)(v, Xn, neigh)
	return vn

# For cellular window smoothing
# Arity == 3.0
dalpha = 2.0 * jnp.pi / pow(2.0, 3.0)
rotalpha = dalpha * 0.5

# Utiliy function for cell smoothing
def interTriangle(origx, origy, ddx, ddy, startx, starty, endx, endy):
	dirx = ( endx - startx )
	diry = ( endy - starty )
	dirno = safe_sqrt( dirx * dirx + diry * diry )
	dirx /= dirno
	diry /= dirno
	val = ddx * diry - ddy * dirx
	segx = -( startx - origx )
	segy = -( starty - origy )
	lb = ( dirx * segy - diry * segx ) / val
	return lb

# Utility function for cell smoothing
def bezier2(ts, p0x, p0y, p1x, p1y, p2x, p2y):
	p01x = ts * p1x + ( 1.0 - ts ) * p0x
	p01y = ts * p1y + ( 1.0 - ts ) * p0y
	p11x = ts * p2x + ( 1.0 - ts ) * p1x
	p11y = ts * p2y + ( 1.0 - ts ) * p1y
	splinex = ts * p11x + ( 1.0 - ts ) * p01x
	spliney = ts * p11y + ( 1.0 - ts ) * p01y
	return [splinex, spliney]

# Computation of the normalized radial distance to Voronoi cell borders
def computeCellDistLarp(p_in, t_in, fbm, alpha, zoom, tx, ty, larp,
	theta, zetax, zetay, amp, wsmooth_s):
	# Compute nppm nearest neighbors (note: im_size_pptbf_0 is used
	# because the image is enlarged in prevision of rotation and scaling)
	# The output image has size im_size*im_size
	v = computeVoronoiLarp(Xd, im_size_pptbf_0, alpha, zoom, tx, ty,
	p_in, t_in, fbm, larp, theta, zetax, zetay, amp)
	# Shorter computation time with padded image!
	vl = jnp.reshape(v[0][:,0], (im_size, im_size))
	vl = jnp.pad(vl, pad_width=((1,1),(1,1)), mode='edge')
	vl = jnp.reshape(vl, (im_size + 2) * (im_size + 2))
	# Input voronoi map + 8 maps with 1 pixel offset
	vln = computeVoronoiNeigh(vl)
	# Compute cell borders
	cb = jnp.any(vln != vln[0], axis=0)
	# Sum angle with label for cell border pixels
	# Note: labels are multiplied by 7 to get non-overlapping
	# angle values between the different labels (7 is the
	# first integer value above 2*pi)
	va = jnp.where(cb, v[2]+(7.0*v[0][:,0]+1.0), 0.0)
	# Sort angles for cell border pixels
	va_arg = jnp.argsort(va)
	# Sum angle with label for all pixels
	va_all = v[2]+(7.0*v[0][:,0]+1.0)
	# Euclidean distance to cell borders
	vd = jnp.interp(va_all, va[va_arg], v[9][va_arg])
	
	# Smoothing based on Bézier interpolation (as done
	# in the EGSR 2020 PPTBF paper)

	palpha = v[2] - rotalpha + 2.0 * jnp.pi
	ka = jnp.floor(palpha/dalpha)
	
	# Note the usage of mod to get angles in the range [0, 2.0*pi]
	start_alpha = jnp.mod(ka * dalpha + rotalpha, 2.0*jnp.pi)
	end_alpha = jnp.mod(ka * dalpha + dalpha + rotalpha, 2.0*jnp.pi)
	nend_alpha = jnp.mod(ka * dalpha + 2.0 * dalpha + rotalpha, 2.0*jnp.pi)
	nstart_alpha = jnp.mod(ka * dalpha - dalpha + rotalpha, 2.0*jnp.pi)

	# Get corresponding distances to cell borders
	start_alpha_d = jnp.interp(start_alpha+(7.0*v[0][:,0]+1.0), va[va_arg], v[9][va_arg])
	end_alpha_d = jnp.interp(end_alpha+(7.0*v[0][:,0]+1.0), va[va_arg], v[9][va_arg])
	nend_alpha_d = jnp.interp(nend_alpha+(7.0*v[0][:,0]+1.0), va[va_arg], v[9][va_arg])
	nstart_alpha_d = jnp.interp(nstart_alpha+(7.0*v[0][:,0]+1.0), va[va_arg], v[9][va_arg])
	# start point
	start_p_x = v[5] + start_alpha_d * jnp.cos(start_alpha)
	start_p_y = v[6] + start_alpha_d * jnp.sin(start_alpha)
	# end point
	end_p_x = v[5] + end_alpha_d * jnp.cos(end_alpha)
	end_p_y = v[6] + end_alpha_d * jnp.sin(end_alpha)
	# midpoint
	mid_p_x = 0.5*(start_p_x + end_p_x)
	mid_p_y = 0.5*(start_p_y + end_p_y)
	# Signed area of the triangle ((v[3],v[4]), (v[5],v[6]),
	# (mid_p_x, mid_p_y)): the sign determines the side of
	# (v[3],v[4]) w.r.t. the median of ((startx_, starty), (endx, endy))
	sa = ((v[5] - v[3]) * (mid_p_y - v[6]) - (mid_p_x - v[5]) * (v[6] - v[4]))

	nend_p_x = v[5] + nend_alpha_d * jnp.cos(nend_alpha)
	nend_p_y = v[6] + nend_alpha_d * jnp.sin(nend_alpha)
	
	nstart_p_x = v[5] + nstart_alpha_d * jnp.cos(nstart_alpha)
	nstart_p_y = v[6] + nstart_alpha_d * jnp.sin(nstart_alpha)
	
	lda = interTriangle(v[5], v[6], v[7], v[8], jnp.where(sa > 0.0, mid_p_x, nstart_p_x),
	jnp.where(sa > 0.0, mid_p_y, nstart_p_y), jnp.where(sa > 0.0, nend_p_x, mid_p_x),
	jnp.where(sa > 0.0, nend_p_y, mid_p_y))
	
	smoothx = v[5] + v[7] * lda
	smoothy = v[6] + v[8] * lda
	
	spline = bezier2(safe_sqrt(jnp.square(smoothx - jnp.where(sa > 0.0, mid_p_x, nstart_p_x)) + jnp.square(smoothy - jnp.where(sa > 0.0, mid_p_y, nstart_p_y))) /
			 safe_sqrt(jnp.square(jnp.where(sa > 0.0, nend_p_x, mid_p_x) - jnp.where(sa > 0.0, mid_p_x, nstart_p_x)) + jnp.square(jnp.where(sa > 0.0, nend_p_y, mid_p_y) - jnp.where(sa > 0.0, mid_p_y, nstart_p_y))),
			 jnp.where(sa > 0.0, mid_p_x, nstart_p_x),
			 jnp.where(sa > 0.0, mid_p_y, nstart_p_y),
			 jnp.where(sa > 0.0, end_p_x, start_p_x),
			 jnp.where(sa > 0.0, end_p_y, start_p_y),
			 jnp.where(sa > 0.0, nend_p_x, mid_p_x),
			 jnp.where(sa > 0.0, nend_p_y, mid_p_y))
	
	splinedist = safe_sqrt(jnp.square(spline[0] - v[5]) + jnp.square(spline[1] - v[6]))
	wsmooth = wsmooth_s
	distances = wsmooth * (1.0 - (v[9] / splinedist)) + \
	(1.0 - wsmooth) * (1.0 - (v[9] / vd))

	# distances: distances to cell borders
	# v[9]: sdd
	# v[7]: ddx (normalized)
	# v[8]: ddy (normalized)
	# v[5]: closest feature point x coordinate
	# v[6]: closest feature point y coordinate
	# v[3]: transformed x coordinate
	# v[4]: transformed y coordinate
	# v[0]: closest neighbors
	# v[1]: larp distance
	# v[2]: angles (not returned)
	#return distances, v[3], v[4], v[0], v[1]
	return distances, v[3], v[4], v[0]

# Differentiable PPTBF generator
@partial(gnool_jit, static_array_argnums=(1,2,3,4,5,6,7))
def procedural_pptbf(iii,
					 zoom, alpha, footprint, tx, ty,
					 p, nppm, pv,
					 normblend_s, feataniso_s, 
					 sigcos_s, larp_s, celldist):
	
	x = celldist[1][iii]
	y = celldist[2][iii]
	
	normblend = normblend_s
	feataniso = 5.0 * feataniso_s
	sigcos = 10.0 * sigcos_s

	##############
	## PPTBF = PP x ( W F )
	##############

	# Get the n closest neighbors
	ri = celldist[3][iii]

	pptbf = 0.0

	# Distance to closest feature point
	ddx = (x - pv[ri[0],3])
	ddy = (y - pv[ri[0],4])
	sdd = safe_sqrt(ddx*ddx+ddy*ddy)
	
	gauss = (jnp.exp(-2.0 * sdd) - jnp.exp(-2.0 * footprint)) / (1.0 - jnp.exp(-2.0 * footprint))
	gauss = jnp.clip(gauss, 0.0, 1.0)

	# Arity == 3.0
	dalpha = 2.0 * jnp.pi / pow(2.0, 3.0)
	rotalpha = dalpha * 0.5

	cv = 0.0
	
	# Distance to cell border
	cv = celldist[0][iii]

	# clip
	cv = jnp.clip(cv, 0.0, 1.0)
    
	coeff1 = normblend * (jnp.exp(cv - 1.0) - jnp.exp(-1.0)) / (1.0 - jnp.exp(-1.0))
	coeff2 = (1.0 - normblend) * gauss
    
	winsum = coeff1+coeff2

	##############
	## Anisotropic Gaussian Feature Function
	##############
	
	lx = pv[ri[0],0]
	ly = pv[ri[0],1]
	angle = pv[ri[0],2]
	deltalx = (x - lx) / p[ri[0],4]
	deltaly = (y - ly) / p[ri[0],5]
	ddxloc = (deltalx * jnp.cos(-angle) + deltaly * jnp.sin(-angle))
	iddy = (- deltalx * jnp.sin(-angle) + deltaly * jnp.cos(-angle))
	ddyloc = iddy / pow(2.0, feataniso)
	dd2 = safe_sqrt(ddxloc*ddxloc+ddyloc*ddyloc)
	ddist = (sigcos * dd2 ) / footprint
	feat = 0.5 * jnp.exp(-ddist)

	pptbf += winsum * feat

	# For all k in {1, ..., nppm - 1}:
	# k is the index of k-th nearest feature point
	def winfeatFunc(k, val):
		##############
		## Window Function: W
		##############

		ddx = (x - pv[ri[k],3])
		ddy = (y - pv[ri[k],4])
		
		# Distance to current feature point
		sdd = safe_sqrt(ddx*ddx+ddy*ddy)
		gauss = (jnp.exp(-2.0 * sdd) - jnp.exp(-2.0 * footprint)) / (1.0 - jnp.exp(-2.0 * footprint))
		gauss = jnp.clip(gauss, 0.0, 1.0)
		
		##############
		## Cellular Window
		##############

		winsum = (1.0 - normblend) * gauss

		##############
		## Feature Function
		############## 

		lx = pv[ri[k],0]
		ly = pv[ri[k],1]
		angle = pv[ri[k],2]
		deltalx = (x - lx) / p[ri[k],4]
		deltaly = (y - ly) / p[ri[k],5]
		ddxloc = (deltalx * jnp.cos(-angle) + deltaly * jnp.sin(-angle))
		iddy = (- deltalx * jnp.sin(-angle) + deltaly * jnp.cos(-angle))
		ddyloc = iddy / pow(2.0, feataniso)
		dd2 = safe_sqrt(ddxloc*ddxloc+ddyloc*ddyloc)
		ddist = (sigcos * dd2) / footprint
		feat = 0.5 * jnp.exp(-ddist)

		return (val + winsum * feat)
	
	pptbf = lax.fori_loop(1, nppm, winfeatFunc, pptbf)
    
	return pptbf

# Full PPTBF pass call
def computePPTBF(P, zoom, alpha, footprint, tx, ty, p, nppm, fbm,
jitter, theta, zetax, zetay, amp, larp, normblend, winfeatcorrel,
feataniso, sigcos, deltaorient, wsmooth):
	
	# Compute feature points values
	cpv = vmap(computePointValues, in_axes=(None, None, None,
	None, None, None, None, None, 0))
	pv = cpv(p, g, zoom, tx, ty, winfeatcorrel, deltaorient,
	jitter, P)

	# Compute distances to cell borders
	celldist = computeCellDistLarp(pv[:,3:5], p[:,2:6], fbm,
	alpha, zoom, tx, ty, larp, theta, zetax, zetay, amp, wsmooth)

	# Compute PPTBF values (per pixel)
	pptbf = vmap(procedural_pptbf, in_axes=(0, None, None, None,
	None, None, None, None, None, None, None, None, None, None))
	result = pptbf(X, zoom, alpha, footprint, tx, ty,
	p, nppm, pv, normblend, feataniso, sigcos, larp, celldist)
	
	# Normalize
	min = jnp.min(result)
	max = jnp.max(result)
	result = result - min
	result = result / (max - min)

	# Compute threshold value
	result_argsorted = jnp.argsort(result)
	v = result[result_argsorted[num_zero]]

	# The probability that two pixels have the same
	# value is very low for genetated PPTBF images,
	# which makes possible to get the exact same
	# number of pixels as in the input binary image
	# Compute binary image
	result_bin = jnp.where(
		result <= v, 0.0, 1.0
	)

	return result, result_bin

# ----- Optimization Phase #2 -----

# Format the target for optimization

# Fake (reconstructed) C-PPTBF image
target = input_image
targetg = jnp.expand_dims(target, axis=0)
targetg = jnp.reshape(targetg, (1, im_size, im_size, 1))

# Input structure map
target_t = input_image_t
target_tg = jnp.expand_dims(target_t, axis=0)
target_tg = jnp.reshape(target_tg, (1, im_size, im_size, 1))

# Pre-trained VGG-19 model fo compute feature vectors
# For the layers of the three first VGG-19 blocks
num_layers = 8
subkeys = random.split(rng, num_layers)
# Layers of the three first VGG-19 blocks
channels = [64, 64, 128, 128, 256, 256, 256, 256]
directions = random_directions(subkeys, channels)
model, init_rngs = define_model(num_layers)

# Compute projected VGG-19 feature vectors for SWD1D
targetg_sw = VGG_SWD1D(model, init_rngs, targetg, 1.0 - targetg,
target_tg, directions)

# Loss function for a single set of parameters
def loss_fn(params):
	pred, pred_t = computePPTBF(P, zoom, alpha, footprint, tx, ty,
	p, nppm, fbm, params[0], params[1], params[2], params[3], params[4],
	params[5], params[6], params[7], params[8], params[9], params[10],
	params[11])

	pred = jnp.expand_dims(pred, axis=0)
	pred = jnp.reshape(pred, (1, im_size, im_size, 1))

	pred_t = jnp.expand_dims(pred_t, axis=0)
	pred_t = jnp.reshape(pred_t, (1, im_size, im_size, 1))

	pred_sw = VGG_SWD1D(model, init_rngs,
	pred, 1.0 - pred, pred_t, directions)

	err_sw = [jnp.square(p - targetg_sw[i]) for i, p in enumerate(pred_sw)]
	err_sw = [jnp.mean(e) for (i, e) in enumerate(err_sw)]
	err_sw = jnp.sum(jnp.array(err_sw))

	return err_sw

# Loss function for an array of parameter sets
def loss_fn_tab(i, params_tab, P, zoom, alpha,
footprint, tx, ty, p, nppm, fbm):

	pred, pred_t = computePPTBF(P, zoom, alpha, footprint, tx, ty,
	p, nppm, fbm, params_tab[12*i], params_tab[12*i+1], params_tab[12*i+2], params_tab[12*i+3], params_tab[12*i+4],
	params_tab[12*i+5], params_tab[12*i+6], params_tab[12*i+7], params_tab[12*i+8], params_tab[12*i+9],
	params_tab[12*i+10], params_tab[12*i+11])
	
	pred = jnp.expand_dims(pred, axis=0)
	pred = jnp.reshape(pred, (1, im_size, im_size, 1))

	pred_t = jnp.expand_dims(pred_t, axis=0)
	pred_t = jnp.reshape(pred_t, (1, im_size, im_size, 1))

	pred_sw = VGGConcat(model, init_rngs,
	pred, 1.0 - pred, pred_t, directions)

	err_sw = [jnp.square(p - targetg_sw[j]) for j, p in enumerate(pred_sw)]
	err_sw = [jnp.mean(e) for (j, e) in enumerate(err_sw)]
	err_sw = jnp.sum(jnp.array(err_sw))

	return err_sw

params = jnp.array([jitter_i, theta_i, zetax_i, zetay_i,
amp_i, larp_i, normblend_i, winfeatcorrel_i, feataniso_i,
sigcos_i, deltaorient_i, wsmooth_i])

def getError(params_dict):
	return params_dict.get('error')

# Function to write parameters in an output JSON file
def writeParams(params, error, global_time, params_json):
	global params_global_dict, params_global_dict_out

	"""
	"jitter": float(params[0]),
	"theta": float(params[1]),
	"zetax": float(params[2]),
	"zetay": float(params[3]),
	"amp": float(params[4]),
	"larp": float(params[5]),
	"normblend": float(params[6]),
	"winfeatcorrel": float(params[7]),
	"feataniso": float(params[8]),
	"sigcos": float(params[9]),
	"deltaorient": float(params[10]),
	"wsmooth": float(params[11])
	"""

	# params_dict: with zetax/y in [0,1]
	
	params_dict = {
		"structure": input_structure_name,
		"error": float(error),
		"global_time": float(global_time),
		"params": params.tolist(),
		"tt": int(tt),
		"zoom": float(zoom_input),
		"alpha": float(alpha),
		"numzero": int(num_zero)
	}

	params_global_dict.append(params_dict)
	params_global_dict.sort(key=getError)

	# Only keep the 10 best values
	last = len(params_global_dict)
	if last > 10:
		last = 10

	params_global_dict = params_global_dict[0:last]

	# params_dict_out: with zetax/y corrected
	
	# zetax/y real values
	params_list_corrected = params.tolist()
	zetax_s = params_list_corrected[2]
	zetay_s = params_list_corrected[3]
	params_list_corrected[2] = ((1.0 - zetax_s) * 
	(2.0 - zoom_zeta) + zetax_s * zoom_zeta)
	params_list_corrected[3] = ((1.0 - zetay_s) *
	(2.0 - zoom_zeta) + zetay_s * zoom_zeta)

	params_dict_out = {
		"structure": input_structure_name,
		"error": float(error),
		"global_time": float(global_time),
		"params": params_list_corrected,
		"tt": int(tt),
		"zoom": float(zoom_input),
		"alpha": float(alpha),
		"numzero": int(num_zero)
	}
	
	params_global_dict_out.append(params_dict_out)
	params_global_dict_out.sort(key=getError)

	# Only keep the 10 best values
	last = len(params_global_dict_out)
	if last > 10:
		last = 10

	params_global_dict_out = params_global_dict_out[0:last]
	
	print("Best local minima:",
	[pd['error'] for pd in params_global_dict_out])

	with open(params_json, "w") as outfile:
		json.dump(params_global_dict_out, outfile, indent=2)

# Function to write generated images all along
# the optimization, if desired
def writeImages(params, pred_name, pred_t_name,
pred_concat_name=None, write_im=0):

	if not write_im:
		return
	
	jitter_p = params[0]
	theta_p = params[1]
	zetax_p = params[2]
	zetay_p = params[3]
	amp_p = params[4]
	larp_p = params[5]
	normblend_p = params[6]
	winfeatcorrel_p = params[7]
	feataniso_p = params[8]
	sigcos_p = params[9]
	deltaorient_p = params[10]
	wsmooth_p = params[11]

	pred_image, pred_image_t = computePPTBF(P, zoom, alpha,
	footprint, tx, ty, p, nppm, fbm, jitter_p, theta_p, zetax_p, zetay_p,
	amp_p, larp_p, normblend_p, winfeatcorrel_p, feataniso_p, sigcos_p,
	deltaorient_p, wsmooth_p)

	pred_image_np = np.array(pred_image)
	pred_image_np = np.reshape(pred_image_np, (im_size, im_size))
	image_output = pred_image_np * 255
	image_output = Image.fromarray(image_output.astype(np.ubyte))
	image_output = image_output.convert('L')
	image_output.save(pred_name)

	pred_image_t_np = np.array(pred_image_t)
	pred_image_t_np = np.reshape(pred_image_t_np, (im_size, im_size))
	image_output = pred_image_t_np * 255
	image_output = Image.fromarray(image_output.astype(np.ubyte))
	image_output = image_output.convert('L')
	image_output.save(pred_t_name)

	if pred_concat_name != None:
		pred_concat_np = np.stack((pred_image_np, 1.0 - pred_image_np, pred_image_t_np), axis=-1)
		pred_concat_np = np.array(pred_concat_np)
		pred_concat_np = np.reshape(pred_concat_np, (im_size, im_size, 3))
		image_output = pred_concat_np * 255
		image_output  = Image.fromarray(image_output.astype(np.ubyte))
		image_output.save(pred_concat_name)

# Set Optax Adam optimizer (this algorithm computes the
# exponential moving average of gradients and square gradients)
@optax.inject_hyperparams
def optimizer(learning_rate, eps=1e-8):
  return optax.chain(
  optax.adam(learning_rate=0.05),
  optax.zero_nans()
)

opt = optimizer(learning_rate)
opt_state = opt.init(params)

# Optimization (gradient descent) step function
def step(params, opt_state):
	value, grads = value_and_grad(loss_fn)(params)
	updates, opt_state = opt.update(grads, opt_state, params)
	params = optax.apply_updates(params, updates)
	norm_grad = l2_norm(grads)
	return value, norm_grad, params, opt_state

paramsNoDeform = jnp.array([
	1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
	1.0, 1.0, 1.0, 1.0, 1.0, 1.0
])

# Basin-hopping perturbation function (all parameters are
# perturbed, except the amplitude of the deformation)
def perturb(params):
	global rng
	key, subkey = random.split(rng)
	val = 0.8
	paramsNew = params + random.uniform(subkey, shape=params.shape,
	minval=-val, maxval=val)
	paramsNew = paramsNew * paramsNoDeform
	paramsNew = jnp.clip(paramsNew, 0.001, 0.999)
	rng = key
	return paramsNew

# Basin-hopping acceptance test
def acceptTest(valueOld, valueNew, T):
	global rng
	if valueNew < valueOld:
		return True
	else:
		key, subkey = random.split(rng)
		rng = key
		prob = math.exp(-(valueNew - valueOld) / T)
		if random.uniform(subkey) <= prob:
			return True
		else:
			return False

def computeLoss_fn(params_tab_indices, params_tab, 
P, zoom, alpha, footprint, tx, ty, p, nppm, fbm):
	loss = vmap(loss_fn_tab,
		in_axes=(0, None, None, None,
		None, None, None, None, None, None, None)
	)
	errors = loss(params_tab_indices, params_tab,
	P, zoom, alpha, footprint, tx, ty, p, nppm, fbm)
	return errors

# Function to find a good starting point for the optimization
def initOptim(P, zoom, alpha, footprint, tx, ty, p, nppm, fbm):
	total_points = 0
	# For 256x256 images (tilings 0, 2, 3)
	params_tab_size = 20
	# For 256x256 images (other tilings)
	#params_tab_size = 10
	params_tab_indices = jnp.arange(0, params_tab_size)
	global_time = 0
	min_error = 1000.0
	seed = int(time.time())
	rng_starting = random.PRNGKey(seed)
	for k in range(0, 10):
		rng_starting, subkey = random.split(rng_starting)
		params_tab = random.uniform(subkey,
		shape=(params_tab_size*12,))
		start = time.time()
		errors = computeLoss_fn(params_tab_indices, params_tab,
		P, zoom, alpha, footprint, tx, ty, p, nppm, fbm)
		end = time.time()
		global_time += end - start
		total_points += params_tab_size
		print(total_points, '/ {:.4f} s'.format(end-start),
		'/ {:.4f} s'.format((end - start)/params_tab_size),
		'/ {:.4f} s'.format(global_time))
		error_index = jnp.argmin(errors)
		if errors[error_index] < min_error:
			min_error = errors[error_index]
			params = params_tab[error_index*12:(error_index+1)*12]
	print("Starting error", min_error)
	print("Global time", global_time)
	return params

# Array to store best local minima
params_global_dict = []
params_global_dict_out = []

# Number of basin-hopping steps
bh_steps = 10
# Number of gradient descent steps for each
# basin-hopping step
gd_steps = 200
# Temperature used for the acceptance test (The higher the
# temperature, the more arbitrary jumps are accepted)
T = 5.0

paramsBest_absolute = jnp.copy(params)
min_error_absolute = 1000.0
global_time = 0

# Parameters of the exponential moving average
# computed on the error
# - Window size close to 20 steps
#a = 0.25
#mean_window_size = 20
# - Window size close to 30 steps
a = 0.1
mean_window_size = 30

# ---- First minimization step ----

paramsBest = jnp.copy(params)
min_error = loss_fn(params)

ema_value = 1000.0
prev_ema_value = 1000.0

print("First minimization")
for i in range(0, gd_steps):
	# Gradient descent step
	start = time.time()
	value, norm_grad, params, opt_state = step(params, opt_state)
	end = time.time()
	global_time += end - start
	params = jnp.clip(params, 0.001, 0.999)

	writeImages(params, 'pred.png',
	'pred_t.png', 'pred_concat.png',
	write_images)
	
	# Update the min error and best parameters
	# for this minimization step
	if value < min_error:
		paramsBest = jnp.copy(params)
		min_error = value

		writeImages(paramsBest, 'pred_min.png',
		'pred_t_min.png', 'pred_min_concat.png',
		write_images)
	
	print(i, value, '/ Min:', min_error, '/ Grad:', norm_grad, '/ LR:', opt_state.hyperparams['learning_rate'],
	'/ {:.4f} s'.format(end-start), '/ {:.4f} s'.format(global_time))
	
	# Exponential moving average update
	prev_ema_value = ema_value
	if i == 0:
		ema_value = value
	else:
		ema_value = a * value + (1-a) * ema_value
	diff_ema_value = ema_value - prev_ema_value
	if i > mean_window_size and diff_ema_value > 0.0:
		print("Break loop")
		break

# Write the parameters
writeParams(paramsBest, min_error, global_time,
dir_out + '/' + input_structure_name + '_pred_params_{:0>2}.json'.format(tt))

# Update the absolute min error and absolute best parameters
if min_error < min_error_absolute:
	print("Update absolute error", min_error)
	
	min_error_absolute = min_error
	paramsBest_absolute = jnp.copy(paramsBest)

	writeImages(paramsBest, 'pred_min.png',
	'pred_t_min.png', 'pred_min_concat.png',
	write_images)

# ---- Basin-hopping steps ----

# Perturb current best parameters before starting
# the next minimization step
params = perturb(paramsBest)

for k in range(bh_steps):
	print("Basin-hopping step", k)
	
	opt_state.hyperparams['learning_rate'] = learning_rate
	paramsBestNew = jnp.copy(params)
	min_error_New = loss_fn(params)

	ema_value = 1000.0
	prev_ema_value = 1000.0
	
	# Minimization step
	for i in range(0, gd_steps):
		# Gradient descent step
		start = time.time()
		value, norm_grad, params, opt_state = step(params, opt_state)
		end = time.time()
		global_time += end - start
		params = jnp.clip(params, 0.001, 0.999)

		writeImages(params, 'pred.png',
		'pred_t.png', 'pred_concat.png',
		write_images)
		
		# Update the min error and best parameters
		# for this minimization step
		if value < min_error_New:
			paramsBestNew = jnp.copy(params)
			min_error_New = value
		
		print(k, i, value, '/ Min:', min_error_New, '/ Grad:', norm_grad, '/ Min abs:',
		min_error_absolute, '/ LR:', opt_state.hyperparams['learning_rate'],
		'/ {:.4f} s'.format(end-start), '/ {:.4f} s'.format(global_time))

		# Exponential moving average update
		prev_ema_value = ema_value
		if i == 0:
			ema_value = value
		else:
			ema_value = a * value + (1-a) * ema_value
		diff_ema_value = ema_value - prev_ema_value
		if i > mean_window_size and diff_ema_value > 0.0:
			print("Break loop")
			break
	
	# Store the param values
	writeParams(paramsBestNew, min_error_New, global_time,
	dir_out + '/' + input_structure_name + '_pred_params_{:0>2}.json'.format(tt))
	
	# Update the absolute min error and absolute best parameters
	if min_error_New < min_error_absolute:
		print("Update absolute error", min_error_New)
		
		min_error_absolute = min_error_New
		paramsBest_absolute = jnp.copy(paramsBestNew)

		writeImages(paramsBestNew, 'pred_min.png',
		'pred_t_min.png', 'pred_min_concat.png',
		write_images)
		
	# Perform acceptance test and perturb parameters
	# before starting the next minimization step
	if acceptTest(min_error, min_error_New, T):
		print(k, "Acceptance test: Accepted")
		params = perturb(paramsBestNew)
	else:
		print(k, "Acceptance test: Refused")
		params = perturb(paramsBest)

for i in range(0, len(params_global_dict)):
	
	writeImages(params_global_dict[i]['params'],
	dir_out + '/' + params_global_dict[i]['structure'] + '_pred_{:0>2}'.format(tt) + '_{:0>2}.png'.format(i),
	dir_out + '/' + params_global_dict[i]['structure'] + '_pred_t_{:0>2}'.format(tt) + '_{:0>2}.png'.format(i),
	None, 1)
