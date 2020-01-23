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
    dummy_input,
    class_names
)
from app_utilities import (
    dream_base_dir,
    dreamt_dir,
    dream_file_type,
    get_available_base_files,
    dreamt_file_name,
    check_for_dream,
    get_dream_time,
    get_dream_pairs,
    update_config
)

################################################################################
#
# Maintain state to clear buffers later
#
################################################################################

session = SessionState.get(config_state = ('dream', True, 'Pablo Picasso'), seen_before = [])


################################################################################
#
# Main Program Loop
#
################################################################################

def main():

    print('Refreshing Page')
    #######################################################################
    # (1) Sidebar
    #######################################################################

    style = st.radio('Style', ['Dream', 'Dream in Style'])
    style = 'dream' if style == 'Dream' else 'dream-style'

    if style == 'dream-style':
        strength = st.radio('Strength', ['Medium Roast', 'Dark Roast'])
        strength = (strength == 'Dark Roast')

        artist_names = [artist.strip() for artist in class_names]
        artist   = st.selectbox('Artist', artist_names)
        if artist == 'Wassily Kandinsky':
            artist = ' ' + artist
    else:
        strength = True
        artist   = 'Pablo Picasso'

    curr_config_state = (style, strength, artist)
    print(f'Config: {curr_config_state}')
    if curr_config_state != session.config_state:
        print('Updating Config')
        session.config_state = curr_config_state
        update_config(*curr_config_state)

    wait_for_dream = st.button('Look for a Dream')
    empty_spot     = st.empty()
    if wait_for_dream:
        empty_spot.markdown('Searching')
    # num_dreams_to_display, app_is_on = sidebar()
    num_dreams_to_display = sidebar()

    dreams = get_dream_pairs(num_dreams_to_display)
    for base_name, dream_name in dreams:
        # Pass on anything already displayed
        if (base_name, dream_name) not in session.seen_before:
            session.seen_before.append((base_name, dream_name))
        base    = lit_load_image(dream_base_dir + base_name )
        dream   = lit_load_image(dreamt_dir     + dream_name)
        # display_slot = display_buffer.pop()
        im_disp = st.image([base, dream], width = 300, use_column_width = False)

    if wait_for_dream:
        print('Entering search mode')
        max_loops  = 60
        loop_count = 0
        while True:
            loop_count += 1
            dreams = get_dream_pairs(num_dreams_to_display)
            dreams = list(filter(lambda pair : pair not in session.seen_before, dreams))
            print(f'New dreams: {dreams}')
            if len(dreams) > 0:
                base_name, dream_name = dreams[-1]
                while True:
                    try:
                        print('Attempting to load images')
                        base    = lit_load_image(dream_base_dir + base_name )
                        dream   = lit_load_image(dreamt_dir     + dream_name)
                        success = True
                        break
                    except:
                        time.sleep(2)
                        success = False
                        print('Issue loading image')
                        break
                # display_slot = display_buffer.pop()
                if success:
                    im_disp = empty_spot.image([base, dream], width = 300, use_column_width = False)
                    session.seen_before.append((base_name, dream_name))
                    print('Dream found!')
                    break
            elif loop_count >= max_loops:
                empty_spot.markdown('''Didn't see a dream... try again?''')
                print('Exiting Waiting Loop')
                break
            else:
                time.sleep(2)
                print(f'Looping (staying alive) -- loop {loop_count} completed')


    print('Done loading page')





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

    ## Social Media
    Insta: @ravimcharan
    LinkedIn: ravimcharan


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

    # if st.sidebar.button('Hard Refresh'):
    #     session.cleared_pairs = []
    #     session.run_id       += 1
    #     st.sidebar.button('Plese press to refresh')

    # app_is_on = st.sidebar.radio('Turn the App on or Off', ['Off', 'On'], key = session.run_id)
    # app_is_on = app_is_on == 'On'

    if st.sidebar.button('Clear Cache'):
        st.caching.clear_cache()
        st.sidebar.button('Please press to refresh')


    num_dreams_to_display = st.sidebar.slider('Number of dreams to display', min_value = 1, max_value = 50, value = 5)
    return num_dreams_to_display#, app_is_on

################################################################################
#
# Run the Page
#
################################################################################



if __name__ == '__main__':
    main()
