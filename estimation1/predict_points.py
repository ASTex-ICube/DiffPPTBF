import numpy as np
from PIL import Image
import tensorflow as tf
import json

import argparse
parser = argparse.ArgumentParser(description="arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input", required=True, help="input image")
parser.add_argument("--minPoints", default=24, help="minimum number of feature points in model, for the normalization")
parser.add_argument("--maxPoints", default=1459, help="maximum number of feature points in model, for the normalization")
args = vars(parser.parse_args())
input = args["input"]
minPoints = int(args["minPoints"])
maxPoints = int(args["maxPoints"])

model = tf.keras.models.load_model("complete_withPoints.h5")
model2 = tf.keras.models.load_model("alpha.h5")

zoom_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10, 12.5, 15.0, 17.5, 20.0]

def renormalize(value, min, max):
	newValue = (np.clip(value, 0.0, 1.0) * (max - min)) + min
	return newValue

def predict(image):
	image = np.array(image, dtype=float)
	image /= 127.5
	image -= 1
	image = image[np.newaxis, :]

	predictions = model.predict(image)
	tiling = int(np.argmax(predictions[0]))
	jittering = float(np.round(renormalize(predictions[1][0][0], 0.01, 0.5), 6))
	points = int(np.round(renormalize(predictions[2][0][0], minPoints, maxPoints), 6))
	normBlend = float(np.round(np.clip(predictions[3][0][0], 0.0, 1.0), 6))
	wsmooth = float(np.round(np.clip(predictions[4][0][0], 0.0, 1.0), 6))
	winfeat = float(np.round(np.clip(predictions[5][0][0], 0.0, 1.0), 6))
	aniso = float(np.round(renormalize(predictions[6][0][0], 0.0, 5.0), 6))
	sigcos = float(np.round(renormalize(predictions[7][0][0], 0.0, 10.0), 6))
	delta = float(np.round(renormalize(predictions[8][0][0], -0.1*np.pi, 0.1*np.pi), 6))
	larp = float(np.round(np.clip(predictions[9][0][0], 0.0, 1.0), 6))

	return tiling, jittering, points, normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp, predictions[0].tolist()

def predictAlpha(image):
	image = np.array(image, dtype=float)
	image /= 127.5
	image -= 1
	image = image[np.newaxis, :]

	predictions = model2.predict(image)
	
	alpha = float(np.round(renormalize(predictions[0][0], 0.0, 2.0), 6))
	
	return alpha

image = Image.open(input)
image = image.resize((200, 200))

tiling, jittering, points, normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp, tilings = predict(image)
alpha = predictAlpha(image)

predictions = []
predictions.append([input[:input.rindex('.')], [tiling, jittering, points, alpha,  normBlend, wsmooth, winfeat, aniso, sigcos, delta, larp, tilings]])
with open('params.json', 'w') as f:
	json.dump({'predictions': predictions}, f, indent=4)
