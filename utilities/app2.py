import streamlit as st
import tensorflow as tf
import time
import glob
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import SessionState
import numpy as np
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

################################################################################
#
# Config
#  Due to import issues, must match config in other files
#
################################################################################

# Streamlit tends to run files from unknown locations so I have hardcoded these
dream_base_dir  = '/Users/rcharan/Downloads/'
dreamt_dir      = '/Users/rcharan/Dropbox/Flatiron/final-project/art-dream/dreamt-images/'
dream_file_type = 'jpg'

################################################################################
#
# Main Program Loop
#
################################################################################

def main():

    #######################################################################
    # (1) Sidebar
    #######################################################################

    num_dreams_to_display = sidebar()

    dreams = get_dream_pairs(num_dreams_to_display)

    for base_name, dream_name in dreams:
        base    = lit_load_image(dream_base_dir + base_name )
        dream   = lit_load_image(dreamt_dir     + dream_name)
        im_disp = st.image([base, dream], width = 300, use_column_width = False)


################################################################################
#
# Utilities
#
################################################################################

@st.cache
def lit_load_image(image_path):
    image = load_image(image_path, cast = tf.uint8)
    return image.numpy()

def get_available_base_files(num_most_recent = None):
    list_of_files = glob.glob(dream_base_dir + '*.jpg')
    list_of_files+= glob.glob(dream_base_dir + '*.png')
    list_of_files.sort(key = os.path.getctime, reverse = True)
    if num_most_recent is not None:
        list_of_files = list_of_files[:num_most_recent]

    list_of_files = [file_path.split('/')[-1] for file_path in list_of_files]
    return list_of_files

def dreamt_file_name(file_name):
    file_name = 'dreamt-' + file_name
    # if file_name[-3:] != dream_file_type:
        # print(f'Warning: looking for {file_name[:-3]}.{dream_file_type} instead')
    file_name = '.'.join(file_name.split('.')[:-1])
    file_name = file_name + '.' + dream_file_type
    return file_name

def check_for_dream(file_name):
    return os.path.exists(dreamt_dir+dreamt_file_name(file_name))

@st.cache
def load_dream_image(file_name):
    file_name = dreamt_file_name(file_name)
    return lit_load_image(dreamt_dir+file_name)

def get_dream_time(file_name_pair):
    dream_path = dreamt_dir + file_name_pair[1]
    return os.path.getctime(dream_path)

def get_dream_pairs(max_pairs):
    found_pairs = []
    for file_name in get_available_base_files():
        if check_for_dream(file_name):
            found_pairs.append((file_name, dreamt_file_name(file_name)))
            if len(found_pairs) == max_pairs:
                break


    found_pairs.sort(key = get_dream_time, reverse = True)
    return found_pairs

################################################################################
#
# Orphaned Code; Scheduled for Garbage Collection
#
################################################################################

# def sidebar_footer():
#     st.sidebar.markdown('## Get Your Photo!')
#     st.sidebar.markdown('Warning: not actually implemented')
#     email = st.sidebar.text_input('Enter your email to get your photo!', '', key = session.run_id)
#     save = st.sidebar.button('Save')
#     return email, save

################################################################################
#
# Sidebar
#
################################################################################

def sidebar():
    st.sidebar.markdown(f'''
    # Welcome to Art Dream"
    ## About
    ArtDream is a computer vision system trained to recognize the artist given
    a photograph of their paintings. Watch what happens when it morphs photos so
    that they look more like whatever it's looking for.

    ## Get your own dream!
    Just Take a photo on an iPhone and airdrop it to "Ravi's Macbook." (1 at a time please)

    ## Technical Support
    - Airdrop is accessible in your phone's photo app, select a photo and choose "share"
    - You should have wifi and bluetooth turned on (no wifi connection necessary)
    - iPhone or iPad: iOS 8.0+
    - Macbook made after about 2010 running MacOS 10.7+
    - If you **really** want, you can email me (ravimcharan@gmail.com) with a jpeg or png file, or text me 646.591.0739
    - Note: this can take a hot second to process because your picture has to go to Oregon and back'
    ''')

    # Debugging Information/other messages and actions
    st.sidebar.markdown('## Menu')
    st.sidebar.button('Soft Refresh')

    if st.sidebar.button('Clear Cache'):
        st.caching.clear_cache()
        st.sidebar.button('Please press to refresh')

    num_dreams_to_display = st.sidebar.slider('Number of dreams to display', min_value = 1, max_value = 20, value = 10)
    return num_dreams_to_display

################################################################################
#
# Run the Page
#
################################################################################



if __name__ == '__main__':
    main()
