import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

from tensorflow.keras.layers import Lambda, Dense, MaxPool2D, MaxPooling2D, AvgPool2D, Flatten, Layer, Dropout, Input, Subtract, Multiply, Add, InputLayer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils  import plot_model

import os

import sys
sys.path.append('../utilities')
from utilities import (
    load_image,
    show_image,
    vgg19_process_image,
    vgg19_deprocess_image,
    precomputed_loss,
    get_image_from_model,
    gram_matrix,
    Source,
    Timer,
    dummy,
    class_names,
)

# Must match in remote_dreamer.py
width = height = 896

# Should be moved to utilities
from tensorflow.keras.constraints import Constraint
class RemainImage(Constraint):
    def __init__(self, rate = 1.0):
        super().__init__()
        self.rate = rate

    def __call__(self, kernel):
        return (self.rate       * tf.clip_by_value(kernel, -150, 150) +
                (1 - self.rate) * kernel)


# Due to difficulties saving the model, just build it (only happens once)
def load_dream_style_model():
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    base_model = load_model('../classification/logs/models/vgg19-INet-down2-b.hdf5')
    style_content_weighting = 100.

    style_layer_weighting   = tf.constant(1/len(style_layers) * style_content_weighting * (1/4))

    input_       = Input(shape = (), batch_size = 1, name = 'dummy_input')
    image_layer  = Source((1, width, height, 3), name = 'image', kernel_constraint = RemainImage(0.9))
    signal       = image_layer(input_)

    output_layers = {
    #     'dense_2'            : 1.0,
        'block4_conv1'       : 0.1,
        'block5_conv1'       : 0.1,
    }
    max_pool_extracted_layers = False

    # Alias
    if 'dense_2' in output_layers:
        output_layers['fc2'] = output_layers['dense_2']
    if 'dense_1' in output_layers:
        output_layers['fc1'] = output_layers['dense_1']

    output_count = 0
    style_input_count  = 0
    outputs = []
    inputs  = [input_]
    def get_summary_function(weight):
        def summary_fn(activation):
            # activation = tf.square(activation)
            axes       = tf.range(1, tf.rank(activation))
            activation = tf.reduce_mean(activation, axis = axes)
            return activation * (-1) * weight
        return summary_fn
    base_model_layers = base_model.layers

    # Add resampling before flattening to allow larger images
    flatten_index     = next(filter(
                                lambda i : isinstance(base_model_layers[i], Flatten),
                                range(len(base_model_layers))))
    resize_layer      = Lambda(lambda tensor : tf.image.resize(tensor, (7,7), method = 'gaussian'),
                               name = 'resize')
    base_model_layers.insert(flatten_index, resize_layer)

    for layer in base_model_layers:
        layer.trainable = False
        if isinstance(layer, InputLayer):
            continue
        elif isinstance(layer, Dropout):
            continue
        elif isinstance(layer, MaxPooling2D):
            layer = AvgPool2D().from_config(layer.get_config())
        elif layer.name in ['dense_2', 'fc2']:
            # Top layer requires different activations
            top_config = layer.get_config()
            top_config['activation'] = 'elu'
            layer     = layer.from_config(top_config)
            top_layer = layer


        signal = layer(signal)

        if layer.name in output_layers:
            output_count += 1
            summarizer_layer = Lambda(get_summary_function(output_layers[layer.name]), name = f'compute_gain_{output_count}')
            # Optional: max pool the layer first (!!)
            if max_pool_extracted_layers:
                max_pool = GlobalMaxPooling2D(name = f'focus_{output_count}')
                total_activation = summarizer_layer(max_pool(signal))
            else:
                total_activation = summarizer_layer(signal)
            outputs.append(total_activation)


        if layer.name in style_layers:
            output_count += 1
            gram_matrix_layer = Lambda(gram_matrix, name = f'gram_{layer.name}')
            gram_signal = gram_matrix_layer(signal)
            batch_input_shape = gram_signal.shape
            input_     = Input(shape = batch_input_shape[1:], batch_size=batch_input_shape[0],
                               name = f'arr_{style_input_count}')
            style_input_count += 1
            inputs.append(input_)
            difference = Subtract()([input_, gram_signal])
            square     = Lambda(tf.square, name = f'square_{output_count}')(difference)
            reduce = Lambda(lambda t : tf.reduce_mean(t, axis = [1,2]), name = f'mean_{output_count}')(square)
            scale  = Lambda(lambda x : x * style_layer_weighting, name = f'weight_{output_count}')(reduce)
            outputs.append(scale)

    final_output = Add()(outputs)

    model = Model(inputs = inputs, outputs = final_output)

    # Weights for the top layer (with artist activations)
    #  were reset; restore them
    top_layer.set_weights(base_model.layers[-1].get_weights())

    return model


# Load the Style and set up the dataset for a call to model.fit
def load_style(artist):
    style_path   = f'../dreaming/style-activations/{artist}.npz'

    with np.load(style_path, allow_pickle = False) as style_data:
        style_inputs = [tf.convert_to_tensor(value) for value in style_data.values()]
    style_data.close()

    inputs = [dummy] + style_inputs
    ds = tf.data.Dataset.from_tensor_slices((tuple(inputs), dummy))
    ds = ds.batch(1)
    ds = ds.cache()
    return ds

def dream_style(model, image, style, nat_width, nat_height, strong = False):
    # Load the Image into the Model
    image_layer = model.get_layer('image')
    image_layer.set_weights([image.numpy()])

    # Compile the model
    adam = tf.optimizers.Adam(learning_rate = 20.0)
    model.compile(optimizer = adam, loss = precomputed_loss)
    model.fit(style, epochs = 7 if strong else 2)

    img = get_image_from_model(model)
    img = tf.image.resize(img, [nat_width, nat_height], method = 'gaussian')
    img = tf.cast(img, tf.uint8)
    return img
