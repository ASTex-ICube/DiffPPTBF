import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import time
from PIL import Image

import time

from opencl.pptbf_opencl_smoothing import cl_pptbf

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto')

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

size = 200
num = 50000
trainNum = int(num * 0.7)
testNum = int(num * 0.3)
batch_size = 32 * 4

images = []
zooms = []
tilings = []
jitters = []
larps = []
normblends = []
wsmooths = []
winfeats = []
anisos = []
sigcoss = []
deltas = []

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

    zooms.append(zoom)
    tilings.append(tiling)
    jitters.append(jitter)
    normblends.append(normblend)
    wsmooths.append(wsmooth)
    winfeats.append(winfeatcorrel)
    anisos.append(feataniso)
    sigcoss.append(sigcos)
    deltas.append(deltaorient)
    larps.append(larp)

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

tilings = np.asarray(tilings, dtype=int).reshape(-1, 1)
tilings = encoder.fit_transform(tilings).toarray()
zooms = np.asarray(zooms, dtype=np.float16).reshape(-1, 1)
zooms = encoder.fit_transform(zooms).toarray()

jitters = np.asarray(jitters, dtype=np.float16)
jitters = normalizeData(jitters)
normblends = np.asarray(normblends, dtype=np.float16)
normblends = normalizeData(normblends)
wsmooths = np.asarray(wsmooths, dtype=np.float16)
wsmooths = normalizeData(wsmooths)
winfeats = np.asarray(winfeats, dtype=np.float16)
winfeats = normalizeData(winfeats)
anisos = np.asarray(anisos, dtype=np.float16)
anisos = normalizeData(anisos)
sigcoss = np.asarray(sigcoss, dtype=np.float16)
sigcoss = normalizeData(sigcoss)
deltas = np.asarray(deltas, dtype=np.float16)
deltas = normalizeData(deltas)
larps = np.asarray(larps, dtype=np.float16)
larps = normalizeData(larps)

trainX = images[:trainNum]
testX = images[-testNum:]

trainTilingY = tilings[:trainNum]
testTilingY = tilings[-testNum:]
trainZoomY = zooms[:trainNum]
testZoomY = zooms[-testNum:]
trainJitterY = jitters[:trainNum]
testJitterY = jitters[-testNum:]

trainNormY = normblends[:trainNum]
testNormY = normblends[-testNum:]
trainWsmoothY = wsmooths[:trainNum]
testWsmoothY = wsmooths[-testNum:]
trainWinfeatY = winfeats[:trainNum]
testWinfeatY = winfeats[-testNum:]
trainAnisoY = anisos[:trainNum]
testAnisoY = anisos[-testNum:]
trainSigcosY = sigcoss[:trainNum]
testSigcosY = sigcoss[-testNum:]
trainDeltaY = deltas[:trainNum]
testDeltaY = deltas[-testNum:]
trainLarpY = larps[:trainNum]
testLarpY = larps[-testNum:]

del images, tilings, zooms, jitters, normblends, wsmooths, winfeats, anisos, sigcoss, deltas, larps

dataset = tf.data.Dataset.from_tensor_slices((trainX, {"tiling_output": trainTilingY, "jittering_output": trainJitterY, 
                                                        "zoom_output": trainZoomY,
                                                        "normblend_output": trainNormY, "wsmooth_output": trainWsmoothY, "winfeat_output": trainWinfeatY, "aniso_output": trainAnisoY, "sigcos_output": trainSigcosY, "delta_output": trainDeltaY, "larp_output": trainLarpY}))
dataset = dataset.shuffle(buffer_size=512).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((testX, {"tiling_output": testTilingY, "jittering_output": testJitterY,
                                                        "zoom_output": testZoomY,
                                                        "normblend_output": testNormY, "wsmooth_output": testWsmoothY, "winfeat_output": testWinfeatY, "aniso_output": testAnisoY, "sigcos_output": testSigcosY, "delta_output": testDeltaY, "larp_output": testLarpY}))
val_dataset = val_dataset.batch(batch_size)

n_tilings = 14
n_zoom = 15

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
    x = data_augmentation(inputs)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    bottleneck = keras.layers.Dropout(0.8)(x)

    outputsTiling = keras.layers.Dense(n_tilings, activation='softmax', name='tiling_output')(bottleneck)
    outputsJittering = keras.layers.Dense(1, activation='linear', name='jittering_output')(bottleneck)
    outputsZoom = keras.layers.Dense(n_zoom, activation='softmax', name='zoom_output')(bottleneck)
    outputsNorm = keras.layers.Dense(1, activation='linear', name='normblend_output')(bottleneck)
    outputsWsmooth = keras.layers.Dense(1, activation='linear', name='wsmooth_output')(bottleneck)
    outputsWinfeat = keras.layers.Dense(1, activation='linear', name='winfeat_output')(bottleneck)
    outputsAniso = keras.layers.Dense(1, activation='linear', name='aniso_output')(bottleneck)
    outputsSigcos = keras.layers.Dense(1, activation='linear', name='sigcos_output')(bottleneck)
    outputsDelta = keras.layers.Dense(1, activation='linear', name='delta_output')(bottleneck)
    outputsLarp = keras.layers.Dense(1, activation='linear', name='larp_output')(bottleneck)

    model = keras.Model(inputs, outputs=[outputsTiling, outputsJittering, outputsZoom, outputsNorm, outputsWsmooth, outputsWinfeat, outputsAniso, outputsSigcos, outputsDelta, outputsLarp])
    model.summary()

    model.compile(
        optimizer=Adam(),
        loss={'tiling_output': categorical_crossentropy, 'jittering_output' : 'mean_squared_error',
        'zoom_output' : categorical_crossentropy, 
        'normblend_output' : 'mean_squared_error', 'wsmooth_output' : 'mean_squared_error', 'winfeat_output' : 'mean_squared_error', 'aniso_output' : 'mean_squared_error', 'sigcos_output' : 'mean_squared_error', 'delta_output' : 'mean_squared_error', 'larp_output' : 'mean_squared_error'},
        metrics={'tiling_output': 'accuracy', 'zoom_output': 'accuracy'}
    )

# Fit data to model
model.fit(dataset, validation_data=val_dataset, epochs=50, callbacks=[], verbose=2, batch_size=batch_size)

base_model.trainable = True

checkpoint = ModelCheckpoint('complete_withoutAlpha-{val_loss:.3f}.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

with mirrored_strategy.scope():
    model.compile(
        optimizer=Adam(1e-5),
        loss={'tiling_output': categorical_crossentropy, 'jittering_output' : 'mean_squared_error', 
        'zoom_output' : categorical_crossentropy,
        'normblend_output' : 'mean_squared_error', 'wsmooth_output' : 'mean_squared_error', 'winfeat_output' : 'mean_squared_error', 'aniso_output' : 'mean_squared_error', 'sigcos_output' : 'mean_squared_error', 'delta_output' : 'mean_squared_error', 'larp_output' : 'mean_squared_error'},
        metrics={'tiling_output': 'accuracy', 'zoom_output': 'accuracy'}
    )

# Fit data to model
model.fit(dataset, validation_data=val_dataset, epochs=75, callbacks=[checkpoint], verbose=2, batch_size=batch_size)




