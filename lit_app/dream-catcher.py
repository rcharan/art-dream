'''
    Look for files that have been processed and are
    available on the server
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
    fetch_file,
    get_undreamt_files,
    remote_dreamt_dir,
    remote_monitor_command
)


# Internal Config
dream_poll_results_loc       = f'{dreamt_dir}../lit_app/remote_dreams.txt'
base_poll_results_loc        = f'{dreamt_dir}../lit_app/remote_base.txt'


def check_for_new_dreams():
    # Target files to find
    undreamt_files = get_undreamt_files()

    # Poll the remote
    response_code = os.system(remote_monitor_command(remote_dreamt_dir, dream_poll_results_loc))

    # Look at the poll results
    with open(dream_poll_results_loc, 'r') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    for file_name in undreamt_files:
        dreamt_name = dreamt_file_name(file_name)
        if dreamt_name in lines:
            print(f'{file_name} -- dream detected remotely')
            success = fetch_file(dreamt_name)
            if success:
                print(f'{file_name} -- dream fetched')
            else:
                print(f'{file_name} -- WARNING: failed to fetch file')

def main():
    print(f'\n-----------------------------------------\n'
        f'Watcher dream-catcher all set up')
    while True:
        check_for_new_dreams()
        time.sleep(5)

if __name__ == '__main__':
    main()
