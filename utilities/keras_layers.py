import tensorflow as tf
from tensorflow.keras.layers import Layer

# Gram Matrix Layers
def gram_matrix(activations):
    result        = tf.linalg.einsum('aijb,aijc->abc', activations, activations)
    input_shape   = tf.shape(activations)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Source layers (with no/fake inputs)
class Source(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shapes):
        self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='uniform', trainable=True)
        super().build(input_shapes)

    def call(self, inputs):
        return self.kernel

    def compute_output_shape(self):
        return self.output_dim

    def get_params(self):
        base_config = super().get_config()
        return {**base_config, 'output_dim' : self.output_dim}
