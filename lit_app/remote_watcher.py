from watchdog.events import FileSystemEventHandler
from file_watcher import Watcher
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
import tensorflow as tf
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
    return img


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


model, width, height = load_dream_model()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            file_name = event.src_path.split('/')[-1]

            print(f'Detected file {file_name}')
            if file_name[-3:] not in ['jpg', 'png']:
                print(f'''File doesn't appear to be a jpeg or png, ignoring''')

            print('Waiting for the file to finish transmission')
            time.sleep(5)

            try:
                Timer.start()
                print(f'Loading the file')
                image, nat_size = load_lit_image(event.src_path, width, height)
                Timer.end()
            except:
                print(f'Error loading the file; will wait and try again')
                time.sleep(10)
                image, nat_size = load_lit_image(event.src_path, width, height)
                Timer.end()


            print(f'Dreaming')
            image = dream(model, image, *nat_size)
            Timer.end()

            print(f'Saving the file')
            save_dream(image, file_name)
            Timer.end()

        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print(f'Noticed file {event.src_path} was modified, ignoring')

watcher = Watcher(dream_base_dir, Handler(), 'dream-base-watcher')
watcher.run()
