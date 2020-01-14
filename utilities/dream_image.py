import tensorflow as tf
from images import (
    load_image,
    show_image,
    vgg19_process_image,
    vgg19_deprocess_image
)

class DreamImage:
    def __init__(self, image_path, width, height, normalizer = 'min-max'):
        # Load the image from disk
        self.img = load_image(image_path)

        # Set the target width and height
        self.width  = width
        self.height = height

    def show_base(self):
        show_image(self.img)

    def prepare_base(self):
        # Resize to fit the model
        img = tf.image.resize(self.img, [self.width, self.height])

        # Pre-process for VGG19 architecture
        img = vgg19_process_image(img)

        # Record base image for later post-processing
        self.resized_base = img

        # Add noise
        # img = img + tf.random.normal(img.shape, stddev = .05)

        # Put the image into a batch of one
        img = tf.expand_dims(img, axis = 0)

        return img

    def decode_dream(self, dream, full_size = False):
        # Remove the dream from a batch of one
        dream    = tf.squeeze(dream)

        if full_size:
            # Resample only the change due to dreaming to preserve image quality
            dream_delta = dream - self.resized_base
            dream_delta = tf.image.resize(dream_delta, self.img.shape.as_list()[:-1])
            img         = dream_delta + vgg19_process_image(self.img)
        else:
            img         = dream

        img = vgg19_deprocess_image(img)
        return img

    def show_dream(self, dream, full_size = False):
        dream = self.decode_dream(dream, full_size = full_size)
        show_image(dream)
