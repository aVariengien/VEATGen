
# VEATGen
### Variational Autoencoder for Texture Generation
###### By Alexandre Variengien

![VEATGen image space](https://github.com/aVariengien/VEATGen/blob/main/results_images/honey_grid1.png "VEATGen image space")

## Description

I trained VAEs on 3 datasets extracted from the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
 * Two 50x50 images datasets: `cracked` (43 images from the `cracked` category of the DTD) and `honey` (30 images from the `honeycombed` category)
 * One 100x100 images dataset: `honey-large` (30 images, same as the `honey`)

Explore the latent space of these pretrained VAE using an intuitive GUI ! 

Read the `report.pdf` file to learn more about this project.

## Environment

This project was developed in Python 3. The GUI uses matplotlib interactive features. 
    
For the VAE, I used the keras (v. 2.3 or higher, running on tensorflow).
The original VAE implementation used can be found [here](https://keras.io/examples/generative/vae/). 

## Context

This project was developed as part of the *Computer Graphics and Digital Images* course from the computer science first year of Master at ENS de Lyon.