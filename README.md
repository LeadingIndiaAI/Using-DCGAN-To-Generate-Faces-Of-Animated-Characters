# DCGAN-Anime-Faces

Using the DCGAN architecture to generate anime-images for internship project

# Dataset

The dataset used were:

1) https://www.kaggle.com/soumikrakshit/anime-faces

It has 21551 anime faces scraped from www.getchu.com, which are then cropped using the anime face detection algorithm in ttps://github.com/nagadomi/lbpcascade_animeface. All images are resized to 64 * 64 for the sake of convenience.


2) https://www.kaggle.com/aadilmalik94/animecharacterfaces

It has 9754 anime images, which are of around 40KB.

# Network

This implementation of DCGAN uses Conv2Dtranspose and Conv2D layers. LeakyReLU activation function was used instead of the most widely used ReLU activation function, so that the problem of dead ReLU does not occur, BatchNorm was also used.

# Training

Training is done on both the dataset, first the 10k images dataset was used, then we used this already trained model with the 20k images dataset. Images are not loaded entirely into memory instead, each time a batch is sampled, only the sampled images are loaded. An overview of what happens each step is:  
-Sample images from dataset (real data)  
-Generate images using generator (gaussian noise as input) (fake data)  
-Add noise to labels of real and fake data  
-Train discriminator on real data 
-Train discriminator on fake data  
-Train GAN on fake images and real data labels  
Training is done for a total of 20,000 steps.


## Faces generated :

# Faces generated at the starting 10 epochs or so :

![0020_image.png](https://github.com/Harshil2001/DCGAN-Anime-Faces/blob/master/images/0020_image.png)
![0070_image.png](https://github.com/Harshil2001/DCGAN-Anime-Faces/blob/master/images/0070_image.png)

# Faces generated around 20,000 epochs:

![17920_image.png](https://github.com/Harshil2001/DCGAN-Anime-Faces/blob/master/images/17920_image.png)
![19880_image.png](https://github.com/Harshil2001/DCGAN-Anime-Faces/blob/master/images/19880_image.png)

The faces look pretty good, but it might look more like an actual face with more training, more data and probably with a better network.
