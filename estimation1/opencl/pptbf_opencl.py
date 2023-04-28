'''
Guillaume Baldi, Rémi Allègre, Jean-Michel Dischler.
Differentiable Point Process Texture Basis Functions for inverse
procedural modeling of cellular stochastic structures,
Computers & Graphics, Volume 112, 2023, Pages 116-131,
ISSN 0097-8493, https://doi.org/10.1016/j.cag.2023.04.004.
LGPL-2.1 license
'''

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

	# Array of unique feature points
	p = np.zeros((len(map),6), dtype=np.float32)
	for key in map:
		d = map[key]
		p[d['ix'],0:6] = (d['data']).astype(np.float32)

	npp = np.reshape(npp, (im_size * im_size))
	fbm = np.reshape(fbm, (im_size * im_size * 2))

	return p, npp, fbm

# Complete PPTBF pass to generate PPTBF images
def cl_pptbf(im_size, tx, ty, rescalex, rescaley, zoom, alpha, tt,
jitter, arity, ismooth, wsmooth, normblend, normsig, larp, normfeat,
winfeatcorrel, feataniso, sigcos, deltaorient, amp, rx ,ry):
	
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
	
	return image_np
