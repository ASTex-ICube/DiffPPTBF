import time
import numpy as np
import pyopencl as cl
from PIL import Image

# load the kernel source code
kernelFile = open("opencl/pptbf_opencl0.cl", "r")
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

# PPTBF pass to compute point locations (without jittering)
def cl_pptbf_0(im_size, tx, ty, zoom, alpha, tt):
	
	p = np.zeros((im_size, im_size, 36, 6), dtype=np.float32)
	npp = np.zeros((im_size, im_size), dtype=np.int32)
	fbm = np.zeros((im_size, im_size, 2), dtype=np.float32)

	p_g = cl.Buffer(ctx, mf.WRITE_ONLY, p.nbytes)
	npp_g = cl.Buffer(ctx, mf.WRITE_ONLY, npp.nbytes)
	fbm_g = cl.Buffer(ctx, mf.WRITE_ONLY, fbm.nbytes)

	knl_pptbf_0 = prg.pptbf_0
	knl_pptbf_0(queue, (im_size, im_size), None, np.int32(im_size), np.float32(tx),
	np.float32(ty), np.float32(zoom), np.float32(alpha), np.int32(tt),
	p_g, npp_g, fbm_g)

	cl.enqueue_copy(queue, p, p_g)
	cl.enqueue_copy(queue, npp, npp_g)
	cl.enqueue_copy(queue, fbm, fbm_g)

	# Map to store unique feature points
	map = {}
	count = 0
	for i in range(im_size):
		for j in range(im_size):
			for k in range(npp[i,j]):
				d = p[i,j,k,0:2]
				d = d.tobytes()
				if d not in map:
					# Feature point not in map: create index
					map[d] = {'ix': count, 'data':p[i,j,k]}
					count += 1

	#print("Feature points map length:", count)

	# Array of unique feature points
	p = np.zeros((len(map),6), dtype=np.float32)
	for key in map:
		d = map[key]
		p[d['ix'],0:6] = (d['data']).astype(np.float32)

	npp = np.reshape(npp, (im_size * im_size))
	fbm = np.reshape(fbm, (im_size * im_size * 2))

	return p, npp, fbm

def cl_gabor(p):

	p_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p)
	
	gabor_values = np.zeros((p.shape[0], 3), dtype=np.float32)
	gabor_values_g = cl.Buffer(ctx, mf.WRITE_ONLY, gabor_values.nbytes)

	knl_gabor = prg.gabor
	knl_gabor(queue, (p.shape[0],), None, p_g, gabor_values_g)

	cl.enqueue_copy(queue, gabor_values, gabor_values_g)	

	return gabor_values

# params:
#jitter, theta, zetax, zetay, amp, larp, normblend,
#winfeatcorrel, feataniso, sigcos, deltaorient, wsmooth
def cl_pptbf(im_size, height_size, tx, ty, zoom, alpha, tt,
	params, num_zero, image_name, image_t_name):

	image_np = np.zeros((im_size, im_size), dtype=np.float32)
	image_g = cl.Buffer(ctx, mf.WRITE_ONLY, image_np.nbytes)
	knl_pptbf = prg.pptbf
	
	knl_pptbf(queue, image_np.shape, None, np.int32(im_size), np.float32(tx), np.float32(ty),
	np.float32(zoom), np.float32(alpha), np.int32(tt),
	np.float32(params[0]), np.float32(params[1]), np.float32(params[2]),
	np.float32(params[3]), np.float32(params[4]), np.float32(params[5]),
	np.float32(params[6]), np.float32(params[7]), np.float32(params[8]),
	np.float32(params[9]), np.float32(params[10]), np.float32(params[11]),
	image_g)

	cl.enqueue_copy(queue, image_np, image_g)

	min = np.min(image_np)
	max = np.max(image_np)
	image_np = image_np - min
	image_np /= max - min

	# Compute threshold value
	image_np_flat = np.reshape(image_np, im_size*im_size)
	image_np_argsorted = np.argsort(image_np_flat)
	v = image_np_flat[image_np_argsorted[num_zero]]
	image_np_bin = np.where(
		image_np <= v, 0.0, 1.0
	)

	# Crop image to the specified height size
	image_np_bin = image_np_bin[:height_size,:]

	"""
	output_image = image_np * 255
	output_image = Image.fromarray(output_image.astype(np.ubyte))
	output_image = output_image.convert('L')
	output_image.save(image_name)
	"""

	output_image = image_np_bin * 255
	output_image = Image.fromarray(output_image.astype(np.ubyte))
	output_image = output_image.convert('L')
	output_image.save(image_t_name)
