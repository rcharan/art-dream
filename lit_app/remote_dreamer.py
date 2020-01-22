import tensorflow as tf
import time
import sys
sys.path.append('../utilities/')

from utilities import (
    Source,
    precomputed_loss,
    vgg19_process_image,
    Timer,
    load_image,
    dummy_input,
    get_image_from_model
)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

model_path = './test-model.hdf5'
dream_base_dir = '../dream-base-images/'
dreamt_dir     = '../dreamt-images/'

def load_dream_model():
    model = load_model(model_path,
           custom_objects={'Source'           : Source,
                           'precomputed_loss' : precomputed_loss}
          )
    image_layer = model.get_layer('image')
    width, height = list(image_layer.compute_output_shape())[1:3]
    opt = tf.optimizers.Adam(learning_rate = 100.0)
    return model, width, height

def dream(model, image, nat_width, nat_height):
    # Load the Image into the Model
    image_layer = model.get_layer('image')
    image_layer.set_weights([image.numpy()])

    # Compile the model
    opt = Adam(lr = 100.0)
    model.compile(optimizer = opt, loss = precomputed_loss)
    model.fit(dummy_input.repeat(1), epochs = 3)

    img = get_image_from_model(model)
    img = tf.image.resize(img, [nat_width, nat_height], method = 'gaussian')
    img = tf.cast(img, tf.uint8)

def load_lit_image(image_path, width, height, mode = 'vgg19'):
    nat_image = load_image(image_path, cast = tf.uint8)
    nat_size  = nat_image.shape[:-1]
    image = tf.image.resize(tf.cast(nat_image, tf.float32), [width, height])
    if mode == 'vgg19':
        image = vgg19_process_image(image)
    else:
        raise NotImplementedError('Other Network Formats not Supported')
    image = tf.expand_dims(image, axis = 0)
    return image, nat_size

def save_dream(image, file_name):
    image = tf.image.encode_jpeg(image)
    out_dir = dreamt_dir + 'dreamt-' + file_name[:-4] + '.jpg'
    tf.io.write_file(out_dir, image)
