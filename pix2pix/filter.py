from opencl.pptbf_2pass_opencl_module import cl_mean_curvature_flow
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import os

import argparse
parser = argparse.ArgumentParser(description="arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input", required=True, help="input image")
parser.add_argument("--iterations", default=40, help="number of filter iterations")
args = vars(parser.parse_args())
input = args["input"]
iterations = int(args["iterations"])

if '/' in input:
	name = input[input.rindex('/'):input.rindex('.')]
else:
	name = input[:input.rindex('.')]

image = Image.open(input).convert("L").resize((256, 256))
target_image = np.array(image, dtype=np.float32)
target_image /= 255.0
input_image = target_image

input_image_rebin = 1 - (input_image < 0.8)
input_image_rebin = input_image_rebin.astype(np.float32)
num_zero = np.count_nonzero(input_image_rebin == 0.0)
num_zero_input = np.count_nonzero(input_image == 0.0)
if num_zero != num_zero_input:
	print("Warning: input image was not binary")
	input_image = input_image_rebin

output_image = cl_mean_curvature_flow(
	input_image.shape[0], input_image, iterations)

thresh = threshold_otsu(output_image)
output_image = output_image > thresh

output_image = output_image.astype(float)
output_image *= 255.0
output_image = Image.fromarray(output_image.astype(np.ubyte))
output_image = output_image.convert('L')
output_image.save(name + "_filter" + str(iterations) + ".png")



