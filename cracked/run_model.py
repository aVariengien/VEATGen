import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import sys
sys.path.append('..')

from VAE import Sampling, VAE
from VEATGen_GUI import run_gui

latent_dim = 20 #dimension of the laten space


#construction of the model

#encoder
encoder_inputs = keras.Input(shape=(50, 50, 3))
x = layers.Conv2D(64, (3,3), activation="relu", strides=2, padding="valid")(encoder_inputs)
x = layers.Conv2D(128, (3,3), activation="relu", strides=2, padding="valid")(x)
x = layers.MaxPooling2D( (2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(200, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


#decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2 * 2 * 64, activation="relu")(latent_inputs)
x = layers.Dropout(0.1)(x)
x = layers.Dense(10 * 10 * 64, activation="relu")(x)

x = layers.Reshape((10, 10, 64))(x)
x = layers.Conv2DTranspose(64, 8, activation="relu", strides=5, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

#loading the pretrained weights
encoder.load_weights("encod_cracked.h5")
decoder.load_weights("decod_cracked.h5")

#building the whole VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())


images = np.load("dataset_cracked.npy")

run_gui(6, vae, images, 50)
#6 is the number of components to run the PCA, it's also the number of cursor available
#to explore the image space












