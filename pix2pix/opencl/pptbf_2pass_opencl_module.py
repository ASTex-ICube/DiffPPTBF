import numpy as np
import pyopencl as cl

im_size = 256

# load the kernel source code
kernelFile = open("opencl/pptbf_opencl.cl", "r")
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


def cl_mean_curvature_flow(im_size, image, n):

	image_s_g_1 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image)
	image_s_g_2 = cl.Buffer(ctx, mf.READ_WRITE, image.nbytes)
	image_s_np = np.zeros((im_size, im_size), dtype=np.float32)

	knl_mcf	= prg.mean_curvature_flow

	for k in range(n//2):
		knl_mcf(queue, image_s_np.shape, None, np.int32(im_size), image_s_g_1, image_s_g_2)
		knl_mcf(queue, image_s_np.shape, None, np.int32(im_size), image_s_g_2, image_s_g_1)

	cl.enqueue_copy(queue, image_s_np, image_s_g_1)

	return image_s_np
