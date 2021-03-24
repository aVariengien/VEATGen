import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from sklearn.decomposition import PCA




def run_gui(nb_comp, vae, images, im_size):
    #begin the gui of VEATGen to explore the image space
    #nb_comp is the number of dimension to include in the search
    # it's also the number of components to be used in the pca decomposition


    def generate_image(coo):
        vect = np.zeros(latent_dim)
        for i in range(len(coo)):
            vect += coo[i]*pca.components_[i]
        return vae.decoder.predict(np.array([vect]))[0]



    def generate_grid_around(coo, N=7, zoom = 2):
        grid = np.zeros((im_size*N, im_size*N, 3))

        samples = []
        for i in range(N):
            for j in range(N):
                pos = np.dot(coo, pca.components_)
                pos += zoom*(i/N-0.5)*pca.components_[0] + zoom*(j/N-0.5)*pca.components_[1]
                samples.append(pos)

        x_decoded = vae.decoder.predict(np.array(samples))

        for i in range(N):
            for j in range(N):
                grid[i*im_size: (i+1)*im_size, j*im_size: (j+1)*im_size] = x_decoded[i*N +j]
        return cv2.cvtColor(grid.astype('float32'), cv2.COLOR_BGR2RGB)

    def update(val):
        coo = []
        for i in range(nb_comp):
            coo.append(comp_sliders[i].val)
        coo = np.array(coo)
        l.set_array(generate_grid_around(coo, int(sNimg.val),sZoom.val))
        fig.canvas.draw_idle()


    def get_random_id():
        return str(np.random.random())[2::7]


    def save_img(val):
        cv2.imwrite('VEATGen_Image_Saved_'+get_random_id()+'.png', cv2.cvtColor(l.get_array()*255, cv2.COLOR_BGR2RGB))



    y_pred = vae.encoder.predict(images)
    means = y_pred[0]

    pca = PCA(n_components=nb_comp)
    pca.fit(means)

    red_means = pca.transform(means)


    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.1, top = 1, right = 1)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    i = generate_grid_around(np.zeros(nb_comp))
    l = plt.imshow(i)
    ax.margins(x=0)
    ax.axis("off")

    comp_sliders = []
    sliders_axes = []

    ax_txt = plt.axes([0.05, 0.75, 0.2, 0.1])
    ax_txt.text(0.3, 0, "VEATGen")
    ax_txt.axis("off")

    for i in range(nb_comp):
        ax1 = plt.axes([0.05, 0.7 - i*0.05, 0.2, 0.03])
        c = Slider(ax1, 'C'+str(i+1), -2, 2, valinit=0)
        c.on_changed(update)
        sliders_axes.append(ax1)
        comp_sliders.append(c)

    ax_N = plt.axes([0.1, 0.7 - nb_comp*0.05, 0.2, 0.03])
    sNimg = Slider(ax_N, 'Nb img', 1, 30, valinit=7, valstep=1)

    sNimg.on_changed(update)

    ax_Z = plt.axes([0.1, 0.7 - (nb_comp+1)*0.05, 0.2, 0.03])
    sZoom= Slider(ax_Z, 'Zoom', 0, 2, valinit=1)
    sZoom.on_changed(update)

    saveax = plt.axes([0.05, 0.7 - (nb_comp+2)*0.05, 0.2, 0.03])
    button = Button(saveax, 'Save Image', hovercolor='0.975')
    button.on_clicked(save_img)

    plt.show()
