'''
Guillaume Baldi, Rémi Allègre, Jean-Michel Dischler.
Differentiable Point Process Texture Basis Functions for inverse
procedural modeling of cellular stochastic structures,
Computers & Graphics, Volume 112, 2023, Pages 116-131,
ISSN 0097-8493, https://doi.org/10.1016/j.cag.2023.04.004.
LGPL-2.1 license
'''

import time
import random
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from opencl.pptbf_opencl import cl_pptbf

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

size = 200
num = 50000
trainNum = int(num * 0.7)
testNum = int(num * 0.3)
batch_size = 32 * 4

images = []
alphas = []

zoom_values = [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10, 12.5, 15.0, 17.5, 20.0]
a = np.random.normal(1.0, 0.19, num)
b = np.random.normal(1.0, 0.19, num)


def generate_new_image(count):
    alpha = random.uniform(0.0, 2.0)
    tiling = random.randint(0, 13)
    zoom = zoom_values[random.randint(0, 11)]
    if tiling >= 8:
        if zoom <= 4.0:
            zoom = 1
        elif zoom > 4.0 and zoom <= 5.0:
            zoom = 1.5
        elif zoom == 7.5:
            zoom = 2.0
        elif zoom == 10:
            zoom = 2.5
        elif zoom == 12.5:
            zoom = 3.0
        elif zoom == 15.0:
            zoom = 3.5
        elif zoom == 17.5:
            zoom = 4.5
        elif zoom == 20.0:
            zoom = 5.0
        
    jitter = random.uniform(0.01, 0.5)
    arity = 10
    ismooth = 0
    normblend = random.uniform(0.0, 1.0)
    wsmooth = random.uniform(0.0, 1.0)
    normsig = 1.0
    normfeat = 2.0
    winfeatcorrel = random.uniform(0.0, 1.0)
    larp = random.uniform(0.0, 1.0)
    feataniso = random.uniform(0.0, 5.0)
    sigcos = random.uniform(0.0, 10.0)
    deltaorient = random.uniform(-0.1*np.pi, 0.1*np.pi)
    amp = random.uniform(0.0, 0.15)
    rx = float(random.randint(0, 10))
    ry = float(random.randint(0, 10))
    tx = random.uniform(-zoom, zoom)
    ty = random.uniform(-zoom, zoom)
    rescalex = a[count]
    rescaley = b[count]

    target_image = cl_pptbf(tx, ty, rescalex, rescaley, zoom, alpha,
                        tiling, jitter, arity, ismooth, wsmooth, normblend, normsig,
                        larp, normfeat, winfeatcorrel, feataniso, sigcos, deltaorient, amp, rx, ry)
    
    min = np.min(target_image)
    max = np.max(target_image)
    target_image = target_image - min
    target_image /= max - min
    target_image *= 255
    
    image_np = Image.fromarray(target_image)
    image_np = image_np.convert('RGB')
    image = np.asarray(image_np)

    if image.mean() < 30.0:
        return image

    alphas.append(alpha)

    return image

start = time.time()

i = 0
while i < num:
    image = generate_new_image(i)
    if image.mean() < 30.0:
        continue
    image = np.array(image, dtype = float)
    image /= 127.5
    image -= 1
    images.append(image)
    i += 1

end = time.time()
print('{:.4f} s'.format(end-start))

images = np.asarray(images, dtype=np.float16)

alphas = np.asarray(alphas, dtype=np.float16)
alphas = normalizeData(alphas)

trainX = images[:trainNum]
testX = images[-testNum:]

trainAlphaY = alphas[:trainNum]
testAlphaY = alphas[-testNum:]

del alphas

dataset = tf.data.Dataset.from_tensor_slices((trainX, {"alpha_output": trainAlphaY}))
dataset = dataset.shuffle(buffer_size=512).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((testX, {"alpha_output": testAlphaY}))
val_dataset = val_dataset.batch(batch_size)


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",
                        input_shape=(size,
                                    size,
                                    3)),
    ]
    )

    base_model = keras.applications.ResNet152V2(
        weights='../resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5', 
        input_shape=(size, size, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(size, size, 3))
    #x = data_augmentation(inputs)

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    bottleneck = keras.layers.Dropout(0.8)(x)

    outputsAlpha = keras.layers.Dense(1, activation='linear', name='alpha_output')(bottleneck)

    model = keras.Model(inputs, outputs=[outputsAlpha])
    model.summary()

    model.compile(
        optimizer=Adam(),
        loss={'alpha_output' : 'mean_squared_error'},
        metrics={}
    )

# Fit data to model
model.fit(dataset, validation_data=val_dataset, epochs=50, callbacks=[], verbose=2, batch_size=batch_size)

base_model.trainable = True

checkpoint = ModelCheckpoint('alpha-{val_loss:.3f}.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

with mirrored_strategy.scope():
    model.compile(
        optimizer=Adam(1e-5),
        loss={'alpha_output' : 'mean_squared_error'},
        metrics={}
    )

# Fit data to model
model.fit(dataset, validation_data=val_dataset, epochs=75, callbacks=[checkpoint], verbose=2, batch_size=batch_size)




