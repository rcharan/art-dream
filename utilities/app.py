import streamlit as st
import tensorflow as tf
import time
import glob
import os
from tensorflow.keras.models import load_model
from utilities import (
    show_image,
    Source,
    get_image_from_model,
    load_image,
    vgg19_process_image,
    precomputed_loss,
    dummy,
    dummy_input
)
import SessionState
session = SessionState.get(run_id=0)


# Streamlit tends to run files from unknown locations so I have hardcoded these
dream_base_dir = '/Users/rcharan/Photos/'
dreamt_dir     = '/Users/rcharan/Dropbox/Flatiron/final-project/art-dream/dreamt-images/'

# This is an extremly hacky workaround to cache the model loading.
#  First, we get the type of the model (which does not seem be exposed
#    by tensorflow anywhere I can find it).
#  Later, we will tell streamlit to not try to hash objects of this type
@st.cache(allow_output_mutation = True)
def get_model_type():
    model = load_model('./test-model.hdf5',
           custom_objects={
               'Source'           : Source,
               'precomputed_loss' : precomputed_loss
           }
          )
    return type(model)
model_type = get_model_type()


# This is the main program
def main():
    # Set up the Sidebar
    st.sidebar.title("Welcome to Art Dream")

    # Reset will clear everything
    if st.sidebar.button('Reset'):
        session.run_id += 1

    null_file_name = '------'
    available_files = get_available_files()
    file_name = st.sidebar.selectbox('Pick a File (most recent on top)',
        [null_file_name] + available_files,
    index = 0, key = session.run_id)
    if file_name != null_file_name:
        file_path = dream_base_dir + file_name

    # Pick the action; used to avoid running app early
    # run_radio = st.sidebar.empty()
    # run  = run_radio.radio('Action',[
    #     'Pick a file',
    #     'View Image',
    #     'Dream'
    # ], key = session.run_id,
    # index = 0 if file_name == null_file_name else 1)

    # User ability to get the file
    st.sidebar.markdown('## Get Your Photo!')
    email = st.sidebar.text_input('Enter your email to get your photo!', '', key = session.run_id)
    # use   = st.sidebar.radio('Permissions', [
    #     'Destroy after reading',
    #     'Send me a copy',
    #     'Okay to use in demos/presentations',
    #     'Okay to post to social media'
    # ], key = session.run_id)
    save = st.sidebar.button('Save')

    # Debugging Information/other messages and actions
    st.sidebar.markdown('## Debugging')
    st.sidebar.markdown(f'App State: {session.run_id}')
    if save:
        st.sidebar.markdown('Saving is not implemented yet!')
    st.sidebar.button('Soft Refresh')

    if st.sidebar.button('Clear Cache'):
        st.caching.clear_cache()
        session.dreamt_files = []

    # Set up the main canvas
    if file_name == null_file_name:
        st.header('Get your own dream!')
        st.markdown('Take a photo and find it at left to get started.')
        st.markdown('Note: images may take a few seconds to load from the camera')
        st.button('Refresh image list')
    else:
        to_dream = st.button('Dream')
        image, _, _ = load_lit_image(file_path, width, height)
        if not to_dream:
            im_disp_a = st.image(nat_image, width = 300, use_column_width = False)
        if to_dream:
            load_dream_image(file_name)
            im_disp_b = st.image([nat_image, dream_img], width = 300, use_column_width = False)

@st.cache
def lit_load_image(image_path):
    nat_image = load_image(image_path, cast = tf.uint8)
    return nat_image.numpy()

def get_file_content_as_string(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def get_available_files(num_most_recent = 5):
    list_of_files = glob.glob(dream_base_dir + '*.jpg')
    list_of_files+= glob.glob(dream_base_dir + '*.png')
    list_of_files.sort(key = os.path.getctime, reverse = True)
    if num_most_recent is not None:
        list_of_files = list_of_files[:num_most_recent]

    list_of_files = [file_path.split('/')[-1] for file_path in list_of_files]
    return list_of_files

def get_dreamt_files():
    list_of_files = glob.glob(dreamt_dir + '*.jpg')
    list_of_files = [file_path.split('/')[-1] for file_path in list_of_files]
    return list_of_files

def load_dream_image(file_name, init_time = 5, timeout = 20):
    while True:
        available = get_dreamt_files()
        file_name = 'dreamt-' + file_name
        if file_name[-3:] != 'jpg':
            print(f'Warning: looking for {file_name[:-3]}.jpg instead')
            file_name = file_name[:-3]+'.jpg'
    if file_name in list_of_files:
        return lit_load_image()


if __name__ == '__main__':
    main()
