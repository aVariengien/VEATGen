import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from VAE import Sampling, VAE

images = np.load("honey/dataset_honey.npy")

latent_dim = 20

##example of a training with the honey 50x50 dataset and model

encoder_inputs = keras.Input(shape=(50, 50, 3))

x = layers.Conv2D(64, (3,3), activation="relu", strides=2, padding="valid")(encoder_inputs)
x = layers.Conv2D(128, (3,3), activation="relu", strides=2, padding="valid")(x)
x = layers.MaxPooling2D( (2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(30, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2 * 2 * 64, activation="relu")(latent_inputs)

x = layers.Dense(10 * 10 * 64, activation="relu")(x)

x = layers.Reshape((10, 10, 64))(x)
x = layers.Conv2DTranspose(64, 8, activation="relu", strides=5, padding="same")(x)

decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

vae.fit(images, epochs=1000, batch_size=1, verbose=2)


