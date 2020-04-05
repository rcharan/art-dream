# ArtDream
## Exploring Deep Representation Learning

I used a convolutional neural network (CNN) to create a classifier for artwork
(classifying into one of 53 artists based on a datset of about 14,700 artworks) in order to explore the implicit representations learned by the classifier. To visualize and understand the representations, I
1. created a t-stochastic neighborhood embedding (t-SNE); and
2. generated "Deep Dreams" (see below) with the goal of generating dreams that are different from those of an ImageNet classifier (to visual inspection) as well as generating different dreams based on the artist in question.

I also deployed this using a simple client/server architecture and a streamlit frontend in order to create a live demonstration.

This was my final project for the Flatiron School Data Science Immersive program in New York City.

For results and more details, you can see the presentation [here](https://docs.google.com/presentation/d/1je4H8SJdYYj8dAzFf_Jfc3kqyNcvyp4vR5nLnIBfMqo/edit?usp=sharing) or the full results of the demonstration on my [website](http://www.ravicharan.com/artdream). Here is a sample:

![Study in Marco](./lit_app/sample.png)

# Technologies
## Models
I trained a VGG-19 architecture as a classifier and explored representations (dreams) from both VGG-19 and Inception v3. All models were implemented in Tensorflow. The Tensorflow 2.0 paradigm (Keras) was used for the main model, with Tensorflow 1.0 style computations for losses some custom layers (e.g. the Gram Matrix for style transfers). I also implemented style transfers with VGG-19. Training and inference were done on a Google Cloud Compute Virtual Machine.

## Dreams
A "Deep Dream" was introduced in a [Google blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html). Roughly speaking, given a neural classifier for images, we feed an image and ask the classifier to morph the image to make it see "more of whatever it is looking for". In the original implementation, if the classifier was asked to classify "dog or cat" then we would morph the image to increase both the amount of "cat" and "dog" that the classifier "sees". Intuitively, this gives us some insight into what the otherwise black-box classifier is doing. Technically speaking, we perform gradient *ascent* on the image in order to increase the activations at various layers of the network. I also experimented with masked dreams, where we only perform gradient ascent on one output node (along the lines of "Give me more Picasso").

## Style Transfers
Style Transfers were introduced in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

## Demonstration
The demonstration was implemented with a streamlit app. A client side script periodically checks a directory for new HEIC files sent by airdrop from any iPhone to the client, converts them to pngs, and uses scp to send them to the server. A server side script checks for new files and generates dreams. Another client side script uses ssh to look for new files, then uses scp to download them. The streamlit app, when told, waits for a new file to display.

The ultimate effect is that you can take a photo on your phone, airdrop it, and wait 10-20 seconds (mostly due to network latency/download times; only about 2-3 seconds are spent on "inference" - i.e. dreaming - on a GPU) 

# How to use this repository
## Software
Major dependencies: Tensorflow 2.0, keras-applications 1.0.9 (there is a bug in 1.0.8 that was fixed in the nightly build at the time installed)

## Files
The project is pretty diverse and is structured as a variety of relatively independent parts. Here they are, listed by folder:
- Classification. Classifier training and evaluation each as a seperate Jupyter notebook.
- Datset. Artist selection - used to create a list of which artwork to retain from the main dataset, then load that information and download and organize the dataset from kaggle. all_data_info is information for the main dataset, and artist-breakdown-annotated contains the information (hand-entered) of which artists to retain. Also contains the t-SNE embedding as a jupyter notebook
- Dreaming: contains a Jupyter notebook for dreaming as well a hybrid dream/style-transfer with target activations for a randomly selected artwork from each artist.
- Utilities: contains the streamlit app frontend as well as various code for re-use (e.g. Gram matrix layers)
- lit_app: contains the client/server backend for the streamlit app. (The organization is such due to issues with relative imports working differently in .ipynb and .py files)
- Style Extraction: contains an implementation of a neural algorithm for artistic style, an experimental branch of this project.

Note that the trained models are not provided, as they are large files.
