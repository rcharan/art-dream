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

def vgg19_deprocess_image(image, clip_and_cast = True):
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    if clip_and_cast:
        out = np.clip(image[:, :, ::-1], 0, 255).astype('uint8')
    else:
        out = image
    return tf.convert_to_tensor(out)

def inceptionV3_process_image(image):
    return tf.keras.applications.inception_v3.preprocess_input(image)

def inceptionV3_deprocess_image(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


def load_image(image_path, cast = tf.float32):
    img = tf.io.read_file(image_path)
    if image_path.endswith('.png'):
        img = tf.image.decode_png(img, channels = 3)
    elif image_path.endswith('.jpg'):
        img = tf.image.decode_jpeg(img, channels = 3)
    else:
        raise TypeError(f'File format for {path} not supported or detected')

    if cast is not None:
        img = tf.cast(img, cast)

    return img
