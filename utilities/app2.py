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
# Maintain state to clear buffers later
#
################################################################################

session = SessionState.get(cleared_pairs = [], run_id = 0)


################################################################################
#
# Main Program Loop
#
################################################################################

def main():

    print('Re-running')
    #######################################################################
    # (1) Sidebar
    #######################################################################

    refresh_button = st.empty()
    num_dreams_to_display, app_is_on = sidebar()

    if not app_is_on:
        return

    display_buffer  = [st.empty() for _ in range(num_dreams_to_display)]

    # See if we have filled up before getting started
    dreams = get_valid_dream_pairs(num_dreams_to_display)
    while len(dreams) == num_dreams_to_display:
        session.cleared_pairs += half_list(dreams)
        dreams = get_valid_dream_pairs(num_dreams_to_display)

    displayed_pairs = session.cleared_pairs.copy()

    for base_name, _ in session.cleared_pairs:
        print(f'cleared: {base_name}')

    while True:
        dreams = get_dream_pairs(num_dreams_to_display)
        for base_name, dream_name in dreams[::-1]:
            # Pass on anything already displayed
            if (base_name, dream_name) in displayed_pairs:
                continue
            else:
                base    = lit_load_image(dream_base_dir + base_name )
                dream   = lit_load_image(dreamt_dir     + dream_name)
                display_slot = display_buffer.pop()
                im_disp = display_slot.image([base, dream], width = 300, use_column_width = False)
                displayed_pairs.append((base_name, dream_name))
                if len(display_buffer) == 1:
                    break
        if refresh_button.button('Stop Searching'):
            session.run_id += 1
            break


        print('Looping (staying alive)')
        time.sleep(5)

        if len(display_buffer) == 1:
            print('Clearing images')
            display_buffer[0].markdown('Out of space! Please Refresh')
            # Blacklist half of the images
            session.cleared_pairs += half_list(dreams)
            break




    print('Exiting Loop. Soft Restart Required')






################################################################################
#
# Utilities
#
################################################################################

@st.cache(show_spinner = False)
def lit_load_image(image_path):
    image = load_image(image_path, cast = tf.uint8)
    return image.numpy()

@st.cache(show_spinner = False)
def load_dream_image(file_name):
    file_name = dreamt_file_name(file_name)
    return lit_load_image(dreamt_dir+file_name)

def half_list(list_):
    num = len(list_) // 2
    return list_[:-num]

def get_valid_dream_pairs(num_dreams_to_display):
    dreams = get_dream_pairs(num_dreams_to_display)
    dreams = filter(lambda pair : pair not in session.cleared_pairs, dreams)
    return list(dreams)


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
    - Note: this can take a hot second to process because your picture has to go to Oregon and back
    ''')

    # Debugging Information/other messages and actions
    st.sidebar.markdown('## Menu')
    st.sidebar.button('Soft Refresh')

    if st.sidebar.button('Hard Refresh'):
        session.cleared_pairs = []
        session.run_id       += 1
        st.sidebar.button('Plese press to refresh')

    app_is_on = st.sidebar.radio('Turn the App on or Off', ['Off', 'On'], key = session.run_id)
    app_is_on = app_is_on == 'On'

    if st.sidebar.button('Clear Cache'):
        st.caching.clear_cache()
        st.sidebar.button('Please press to refresh')


    num_dreams_to_display = st.sidebar.slider('Number of dreams to display', min_value = 10, max_value = 50, value = 30)
    return num_dreams_to_display, app_is_on

################################################################################
#
# Run the Page
#
################################################################################



if __name__ == '__main__':
    main()
