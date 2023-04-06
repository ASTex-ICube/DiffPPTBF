import numpy as np
from opencl.pptbf_2pass_opencl_module import cl_thresh
import torchvision.transforms as transforms

size = 256
transform = transforms.Grayscale()

def convertGrayscale(img):
	img = (img + 1) * 127.5
	img = img.transpose(1, 2, 0).reshape(size, size, 3)
	img = np.dot(img[:,:,:3], [0.2989, 0.5870, 0.1140])
	img /= 255
	return img

def l2error(binary, generatedBinary):
	return np.square(np.subtract(binary, generatedBinary)).mean()

def thresh(result, num_zero):
	result_flat = np.reshape(result, (256*256))
	result_argsorted = np.argsort(result_flat)
	v = result_flat[result_argsorted[int(num_zero*256*256)]]
	return v

def getError(real, input_image, real_thresh):
	real = real.cpu().detach().numpy()
	real_thresh = float(real_thresh[0])
	real = (real[0,0,:,:] + 1) / 2

	input_image = input_image.cpu().detach().numpy()

	error = 0.0

	for i in range(input_image.shape[0]):

		bin_image = convertGrayscale(input_image[i]).astype(np.float32)
		bin_image = (bin_image - np.min(bin_image)) / (np.max(bin_image) - np.min(bin_image))
		
		binary = cl_thresh(256, bin_image, thresh(bin_image, real_thresh))

		binary = binary / 255

		error += l2error(real, binary)

	error = error / input_image.shape[0]

	return error
