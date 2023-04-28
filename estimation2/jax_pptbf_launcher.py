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

# Natural results with this file
params_global_file = "predictions_filtered_points_filters40.json"
dir_in = "natural/pix2pix"
dir_out = "natural/opti"

# Read JSON global params file
with open(params_global_file, 'r') as f:
  data = json.load(f)

print("Number of inputs:", len(data['predictions']))

# Get tiling types

# Min 4% for a tiling
#min_tt_prob = 0.04

# Min 8% for a tiling
#min_tt_prob = 0.08

# Min 10% for a tiling
min_tt_prob = 0.1

# Min 20% for a tiling
#min_tt_prob = 0.2

# Min 25% for a tiling
#min_tt_prob = 0.25

tt_global = []
for i in range(0, len(data['predictions'])):
	tt = np.array(data['predictions'][i][1][11][0])
	tt_is = np.argsort(tt)[::-1]
	tt = np.extract(tt[tt_is] >= min_tt_prob, tt_is)
	tt_global.append(tt.tolist())

tt_global_flat = [item for sublist in tt_global for item in sublist]
print("Number of tilings to test:", len(tt_global_flat))

# Loop on input structures
#for i in range(0, 1):
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
		
		command = ("python jax_pptbf.py" +
		" --input " + data['predictions'][i][0] +
		" --params " + params_global_file +
		" --dir_in " + dir_in + " --dir_out " + dir_out +
		" --write_images 0" +
		" --tt " + str(tt_global[i][k]) +
		" --ttnfp " + ttnfp_file)
		print(command)
		os.system(command)
