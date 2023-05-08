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
import numpy as np

# File with the number of feature points for
# each tiling type w.r.t. the zoom
ttnfp_file = "data/ttnfp.json"

# Dataset of structure maps extracted from natural textures
dir_in = "../datasets/natural"

# Dataset of structure maps extracted from synthetic textures
#dir_in = "../datasets/synthetic"

# Output directory
dir_out = "../datasets/output"

params_global_file = dir_in + "/init_params.json"

# Read JSON global params file
with open(params_global_file, 'r') as f:
  data = json.load(f)

print("Number of input structure maps:", len(data['predictions']))

# Get tiling types for a minimal given probability in
# the distribution computed in Optimization Phase #1

# Min 10% for a tiling type
min_tt_prob = 0.1

tt_global = []
for i in range(0, len(data['predictions'])):
	tt = np.array(data['predictions'][i][1][11][0])
	tt_is = np.argsort(tt)[::-1]
	tt = np.extract(tt[tt_is] >= min_tt_prob, tt_is)
	tt_global.append(tt.tolist())

tt_global_flat = [item for sublist in tt_global for item in sublist]
print("Number of tilings to consider:", len(tt_global_flat))

# Python command
python_command = 'python'

# Loop on input structure maps
for i in range(0, len(data['predictions'])):
	input_structure_name = data['predictions'][i][0]
	# Loop on tiling type
	for k in range(0, len(tt_global[i])):
		print("-----", input_structure_name, "-----")
		print("Tiling type", tt_global[i][k],
		("("+str(k+1)+" on "+str(len(tt_global[i]))+")"))

		# Continue if the input has already been processed
		path_to_file = (dir_out + '/' + input_structure_name + 
		'_pred_t_{:0>2}'.format(tt_global[i][k]) +
		'_09.png')
		if os.path.exists(path_to_file):
			print("Already processed")
			continue
		
		command = (python_command + " jax_pptbf.py"
		" --input " + data['predictions'][i][0] +
		" --params " + params_global_file +
		" --dir_in " + dir_in + " --dir_out " + dir_out +
		" --write_images 0" +
		" --tt " + str(tt_global[i][k]) +
		" --ttnfp " + ttnfp_file)
		print(command)
		os.system(command)
