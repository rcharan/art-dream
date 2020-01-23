from remote_dreamer import (
    dream,
    load_lit_image,
    save_dream,
    dream_base_dir,
    dreamt_dir,
    load_dream_model
)

from remote_dreamer_in_style import (
    load_dream_style_model,
    load_style,
    dream_style
)

import sys
sys.path.append('../utilities/')
from app_utilities import (
    get_undreamt_files
)
from datetime import datetime
import tensorflow as tf
import time

blacklist = []

def get_config():
    with open('dreamer_config.txt', 'r') as f:
        line = f.readline()
    line = line.strip()
    line = line.split(',')
    line = [l.strip() for l in line]
    type, strength, artist = line
    if artist == 'Wassily Kandinsky':
        artist = ' ' + artist
    if strength == 'true':
        strength = True
    elif strength == 'false':
        strength = False
    return type, strength, artist

def process_new_dreams():
    new_dreams = get_undreamt_files(dream_base_dir, dreamt_dir)
    for file_name in new_dreams:
        start = datetime.now()
        if file_name in blacklist:
            continue
        print(f'{file_name} -- detected')
        if file_name[-3:] not in ['jpg', 'png']:
            print(f'''{file_name} -- doesn't appear to be a jpeg or png, ignoring''')
            blacklist.append(file_name)
            continue
        try:
            image, nat_size = load_lit_image(dream_base_dir+file_name, width, height)
        except Exception as e:
            print(f'''{file_name} -- error loading file''')
            print(file_name, '--', e)
            continue

        type, strength, artist = get_config()
        print(f'{file_name} -- loaded, dreaming type {type} strength {strength}')
        if type == 'dream':
            image = dream(model, image, *nat_size)
        elif type == 'dream-style':
            style = load_style(artist)
            image = dream_style(style_model, image, style, *nat_size, strong = strength)

        print(f'{file_name} -- saving')
        save_dream(image, file_name)

        time_elapsed = (datetime.now() - start).seconds
        print(f'{file_name} -- {time_elapsed}s elapsed')

model, width, height = load_dream_model()
style_model          = load_dream_style_model()

def main():
    print(f'\n-----------------------------------------\n'
        f'Watcher remote-watcher all set up')
    while True:
        process_new_dreams()
        time.sleep(5)

if __name__ == '__main__':
    main()
