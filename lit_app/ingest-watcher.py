'''
    Watch for incoming files and post them to the server
'''

import sys
import os
import time
from datetime import datetime
from watchdog.events import FileSystemEventHandler
from file_watcher import Watcher
sys.path.append('../utilities')
from app_utilities import (
    dream_file_type,
    dreamt_dir,
    dream_base_dir,
    dreamt_file_name,
    post_file,
    fetch_file
)

# Alias I am so so sorry
local_dreamt_dir       = dreamt_dir
local_dream_base_dir   = dream_base_dir
local_ingest_dir       = dream_base_dir

class Handler(FileSystemEventHandler):

    def __init__(self):
        self.seen_files = []

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            file_name = event.src_path.split('/')[-1]
            bare_name = '.'.join(file_name.split('.')[:-1])
            file_type = file_name.split('.')[-1].lower()

            print(f'{bare_name} -- Detected as {file_type.upper()}. Waiting for the file to finish loading locally')
            time.sleep(1)

            if file_type in ['jpg', 'png']:
                print(f'{bare_name} -- posting file')
                success = post_file(event.src_path, bare_name)
            elif file_type in ['heic', 'heis']:
                print(f'{bare_name} -- Converting HEIC to jpg')
                safe_bare_name = bare_name.replace(' ', '_')
                command = f'magick "{event.src_path}" "{local_ingest_dir}{safe_bare_name}.jpg"'
                os.system(command)

            else:
                print(f'''File {bare_name} doesn't appear to be a jpeg, png, or HEIC, ignoring''')



        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print(f'Noticed file {event.src_path} was modified, ignoring')

if __name__ == '__main__':
    watcher = Watcher(local_ingest_dir, Handler(), 'dream-base-watcher')
    watcher.run()
