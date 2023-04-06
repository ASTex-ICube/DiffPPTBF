import time
import numpy as np
import pyopencl as cl
from PIL import Image

im_size = 200

# load the kernel source code
kernelFile = open("opencl/pptbf_opencl_smoothing.cl", "r")
kernelSrc = kernelFile.read()

# Uncomment to choose device at run time
#ctx = cl.create_some_context()

# Select the first platform [0]
platform = cl.get_platforms()[0]
print(platform)
# Select the first device on this platform [0]
device = platform.get_devices()[0]
print(device)
# Create a context with your device
ctx = cl.Context([device])

queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, kernelSrc).build()

#------------------------------
# /!\ Only once for all the generated images
# Create random image with uniform distribution
rng = np.random.default_rng(0)
#uniform = rng.random(size=(im_size*im_size), dtype=np.float32)
# Number of intensity values
K = 256
# To ensure that random values include 0 and K-1
uniform = rng.integers(low=0, high=255, size=(im_size*im_size),
dtype=np.int32, endpoint=True).astype(np.float32) / 255.0
uniform_argsorted = np.argsort(uniform).astype(np.int32)

knl_equalize = prg.equalize

# Image thresholding
# - Input image: 2D with float type
# - Output image: 2D with integer type and values in {0,255}
def cl_thresh(im_size, image, t):
	
	image_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
	image_t = np.zeros((im_size, im_size), dtype=np.int32)
	image_t_g = cl.Buffer(ctx, mf.WRITE_ONLY, image_t.nbytes)

	knl_thresh = prg.thresh
	knl_thresh(queue, (im_size, im_size), None, np.int32(im_size),
	image_g, np.float32(t), image_t_g)

	cl.enqueue_copy(queue, image_t, image_t_g)

	return image_t

def cl_equalize(im_size, image_np):
	uniform_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uniform)
	uniform_argsorted_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uniform_argsorted)
	image_flat = image_np.ravel()
	image_flat_argsorted = np.argsort(image_flat)
	image_flat_argsorted = np.argsort(image_flat_argsorted).astype(np.int32)
	image_flat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat)
	image_flat_argsorted_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat_argsorted)
	image_flat_matched = np.zeros(im_size*im_size, dtype=np.float32)
	image_flat_matched_g = cl.Buffer(ctx, mf.WRITE_ONLY, image_flat_matched.nbytes)

	knl_equalize(queue, image_flat.shape, None, uniform_g, uniform_argsorted_g,
	image_flat_g, image_flat_argsorted_g, image_flat_matched_g)

	cl.enqueue_copy(queue, image_flat_matched, image_flat_matched_g)

	image_matched = image_flat_matched.reshape((im_size, im_size))

	return image_matched

def cl_pptbf(tx, ty, rescalex, rescaley, zoom, alpha, tt, jitter, arity, ismooth, wsmooth, normblend, normsig, larp, normfeat, winfeatcorrel, feataniso, sigcos, deltaorient, amp, rx ,ry):
	knl = prg.pptbf
	image_np = np.zeros((im_size, im_size), dtype=np.float32)
	image_g = cl.Buffer(ctx, mf.WRITE_ONLY, image_np.nbytes)
	knl(queue, image_np.shape, None, np.int32(im_size), np.float32(tx), np.float32(ty),
	np.float32(zoom), np.float32(alpha), np.int32(tt), np.float32(jitter), np.float32(arity),
	np.float32(ismooth), np.float32(wsmooth), np.float32(normblend), np.float32(normsig),
	np.float32(larp), np.float32(normfeat), np.float32(winfeatcorrel), np.float32(feataniso),
	np.float32(sigcos), np.float32(deltaorient), np.float32(amp), np.float32(rx), np.float32(ry),
	image_g)
	
	cl.enqueue_copy(queue, image_np, image_g)
	
	"""min = np.min(image_np)
	max = np.max(image_np)
	image_np = image_np - min
	image_np /= max - min

	image_np *= 255
	image_np = Image.fromarray(image_np)
	image_np = image_np.convert('RGB')
	image_np = np.asarray(image_np)"""
	
	return image_np
