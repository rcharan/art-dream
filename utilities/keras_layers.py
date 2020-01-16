import tensorflow as tf
from tensorflow.keras.layers import Layer

from images import vgg19_deprocess_image

# Gram Matrix Layers
def gram_matrix(activations):
    result        = tf.linalg.einsum('aijb,aijc->abc', activations, activations)
    input_shape   = tf.shape(activations)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Source layers (with no/fake inputs)
class Source(Layer):

    def __init__(self, output_dim, kernel_constraint = None, **kwargs):
        self.output_dim        = output_dim
        self.kernel_constraint = kernel_constraint
        super().__init__(**kwargs)

    def build(self, input_shapes):
        self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='uniform', trainable=True)
        super().build(input_shapes)

    def call(self, inputs):
        if self.kernel_constraint is not None:
            return self.kernel_constraint(self.kernel)
        else:
            return self.kernel

    def compute_output_shape(self):
        return self.output_dim

    def get_params(self):
        base_config = super().get_config()
        return {**base_config, 'output_dim' : self.output_dim}

def precomputed_loss(dummy, loss):
    return loss

def get_image_from_model(model, layer_name = 'image'):
    out_img = model.get_layer(layer_name).get_weights()[0]
    out_img = tf.squeeze(vgg19_deprocess_image(out_img, clip_and_cast = False))
    out_img = out_img - tf.reduce_min(out_img)
    out_img = out_img/tf.reduce_max(out_img) * 256
    out_img = tf.cast(out_img, tf.uint8)
    return out_img
