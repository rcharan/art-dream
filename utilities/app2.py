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
from app_utilities import (
    dream_base_dir,
    dreamt_dir,
    dream_file_type,
    get_available_base_files,
    dreamt_file_name,
    check_for_dream,
    get_dream_time,
    get_dream_pairs
)


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

@st.cache
def load_dream_image(file_name):
    file_name = dreamt_file_name(file_name)
    return lit_load_image(dreamt_dir+file_name)

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
