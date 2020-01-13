import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Display a tensor
def show_image(image):
    figsize = image.shape.as_list()[:-1]
    figsize = tuple(size / 72 for size in figsize)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis('off')
    plt.show()

def vgg19_process_image(image):
    return tf.keras.applications.vgg19.preprocess_input(image)

def vgg19_deprocess_image(image):
    image = image.numpy()
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    out = np.clip(image[:, :, ::-1], 0, 255).astype('uint8')
    return tf.convert_to_tensor(out)
