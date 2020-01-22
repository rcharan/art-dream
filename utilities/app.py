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
# Create a session for persistence and refresh capabilities
#
################################################################################

session = SessionState.get(run_id=0,
                            seen_files = [],
                            mode = 'lazy',
                            wait_times = [20],
                            mode_id = 0)

################################################################################
#
# Main Program Loop
#
################################################################################

def main():

    clearable_elements = []
    #######################################################################
    # (1) Decide app mode.
    #     Terrible UI but required by streamlit paradism
    #######################################################################

    st.sidebar.markdown('### Application Mode')
    eager = st.sidebar.empty()
    mode = eager.radio('App Mode', ['Lazy', 'Eager'],
                index = (0 if session.mode == 'lazy' else 1), key = session.mode_id)
    if session.mode != mode.lower():
        session.mode_id += 1
        session.mode = mode.lower()

    if session.mode == 'lazy':
        st.sidebar.markdown('In lazy mode, you will have to refresh '
                    'the page to see your file and select it yourself'
                    ' (the computer gets to be lazy, not you!)')
    elif session.mode == 'eager':
        st.sidebar.markdown('In eager mode, computer will auto-detect files')



    #######################################################################
    # (1) Sidebar Header
    #######################################################################
    sidebar_header()

    #######################################################################
    # (2) Detect Available Files and either proceed automatically or prompt
    #        the user
    #######################################################################

    # Program attempts to automatically load and dream new files.
    #  Use this to clear the backlog
    if st.sidebar.button('Mark all images seen'):
        session.seen_files = get_available_base_files()

    # In eager mode, loop until a file is detected
    #  Break the loop eventually if nothing is found
    look_count = 0
    # max_looks  = 60
    max_looks  = 120
    display_files = None
    null_file_name = '------'
    while True:
        # Detect Available files to dream on
        available_files = get_available_base_files()

        # See which files are new
        new_files = list(filter(lambda f : f not in session.seen_files,
                                available_files))

        # Wiat for new files in eager mode
        if len(new_files) == 0 and session.mode == 'eager':
            print(f'Polling for new files: try {look_count}/{max_looks}')
            time.sleep(1)
            if look_count == max_looks:
                session.mode = 'lazy'
                file_name = null_file_name
                mode = eager.radio('App Mode', ['Lazy', 'Eager'],
                    index = (0 if session.mode == 'lazy' else 1), key = session.mode_id)
                st.markdown('Failed to find a file; try picking one manually?')
                _ = st.button('Refresh image list')
            else:
                if look_count == 0:
                    clearable_elements += welcome_screen()
                look_count += 1

        # Pick the least recent new file in eager mode (for stability)
        elif session.mode == 'eager':
            clearable_elements += welcome_screen()
            file_name = new_files[-1]
            break

        # In other mode, show a dropdown bar
        else:
            if len(new_files) == 0:
                display_files = available_files[:5]
            else:
                display_files = new_files

            if look_count != max_looks:
                clearable_elements += welcome_screen()
            file_name = st.selectbox('Pick a File (most recent on top)',
                [null_file_name] + display_files,
            index = 0, key = session.run_id)
            break

    #######################################################################
    # (5) File Selected Behavior
    #      If the dream is there, then load both; otherwise just the base
    #      image
    #######################################################################
    if file_name != null_file_name:
        for elt in clearable_elements:
            elt.empty()
        file_path        = dream_base_dir + file_name
        base_image       = lit_load_image(file_path)
        dream_done       = check_for_dream(file_name)
        # If done, display the dream
        if dream_done:
            dream_image = load_dream_image(file_name)
            im_disp     = st.image([base_image, dream_image], width = 300, use_column_width = False)
            next_dream_button()
        # Otherwise, wait for it to be done
        else:
            # Display the base image while waiting
            im_disp_a = st.image(base_image, width = 300, use_column_width = False)
            st.markdown('Waiting for Dream')

            # Compute expected and max/timeout times based on recent history
            expected_wait = session.wait_times
            if len(expected_wait) > 10:
                expected_wait = expected_wait[-10:]
            expected_wait = np.array(expected_wait).sum() * 1.1
            max_wait      = expected_wait * 1.5

            # Set up timing and progress bar
            start_time    = datetime.now()
            bar           = st.progress(0)

            # Periodically poll for completion and update progress bar
            apology_message_printed = False
            while not dream_done:
                # Compute time taken
                time_elapsed = (datetime.now() - start_time).seconds
                pct_elapsed  = time_elapsed/expected_wait

                # Fail gracefully and apologetically
                if pct_elapsed > 100:
                    bar.progress(100)
                    if not apology_message_printed:
                        st.markdown('Sorry... this is taking longer than expected')
                        apology_message_printed = True
                    if time_elapsed > max_wait:
                        st.markdown('Timeout error')
                        session.mode = 'lazy'
                        st.button('Please press this button to save the computer'
                                  ' from boredom.')
                        break

                # If we haven't failed yet, see if we have suceeded!
                else:
                    bar.progress(int(pct_elapsed))
                    time.sleep(1)
                    dream_done = check_for_dream(file_name)
                    if dream_done:
                        st.balloons()
                        session.wait_times.append(time_elapsed)
                        dream_image = load_dream_image(file_name)
                        im_disp     = st.image([base_image, dream_image], width = 300, use_column_width = False)
                        next_dream_button()
                        break

    #######################################################################
    # (3) Sidebar footer
    #######################################################################
    email, save = sidebar_footer()
    debugging_sidebar(save)

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

# def get_dreamt_files():
#     list_of_files = glob.glob(dreamt_dir + f'*.{dream_file_type}')
#     list_of_files = [file_path.split('/')[-1] for file_path in list_of_files]
#     return list_of_files

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

################################################################################
#
# Ancillary Sidebar Behavior
#
################################################################################

def sidebar_header():
    # Set up the Sidebar

    # Reset will clear sliders, fields, etc.
    if st.sidebar.button('Reset'):
        session.run_id += 1

def sidebar_footer():
    st.sidebar.markdown('## Get Your Photo!')
    st.sidebar.markdown('Warning: not actually implemented')
    email = st.sidebar.text_input('Enter your email to get your photo!', '', key = session.run_id)
    save = st.sidebar.button('Save')
    return email, save

def debugging_sidebar(save):
    # Debugging Information/other messages and actions
    st.sidebar.markdown('## Debugging')
    st.sidebar.markdown(f'App State: {session.run_id}')
    st.sidebar.markdown(f'App Mode: {session.mode}')
    if save:
        st.sidebar.markdown('Saving is not implemented yet!')
    st.sidebar.button('Soft Refresh')

    if st.sidebar.button('Clear Cache'):
        st.caching.clear_cache()
        session.dreamt_files = []

################################################################################
#
# Welcome Screen
#
################################################################################

def welcome_screen():
    elts = []
    elts.append(st.markdown(f'''
    # Welcome to Art Dream"
    ## Get your own dream!'
    Take a photo on an iPhone and airdrop it to "Ravi's Macbook."
    {'Then wait for your dream'
     if session.mode == 'eager'
     else 'Then select the file at left'}

    - Airdrop is accessible in your phone's photo app, under sharing
    - You should have wifi and bluetooth turned on (no wifi connection necessary)
    - Must have an iPhone or iPad running iOS8.0+ or a Macbook not from the aughts or earlier with Lion (10.7)+

    Note: this can take a hot second to process because your picture has to go to Oregon and back')

    {'Set the app to Eager Mode at top left to auto-detect' if session.mode == 'lazy' else ''}
    '''))
    if session.mode == 'lazy':
        button = st.empty()
        _ = button.button('Refresh image list')
        elts.append(button)

    return elts

################################################################################
#
# Conclusion/Reset Interface
#
################################################################################

def next_dream_button():
    if st.button('Next Dream'):
        session.run_id += 1

################################################################################
#
# Run the Page
#
################################################################################



if __name__ == '__main__':
    main()
