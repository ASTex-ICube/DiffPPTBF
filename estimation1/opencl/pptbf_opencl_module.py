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
