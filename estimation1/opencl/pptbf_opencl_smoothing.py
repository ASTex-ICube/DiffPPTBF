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
	
	return image_np
