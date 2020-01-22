import os
import time
from datetime import datetime
from watchdog.events import FileSystemEventHandler
from file_watcher import Watcher

# Remote directories
remote_dreamt_dir      = '~/art-dream/dreamt-images/'
remote_dream_base_dir  = '~/art-dream/dream-base-images/'

# Local directories
_local_dir             = '/Users/rcharan/Dropbox/Flatiron/final-project/art-dream/'
local_dreamt_dir       = _local_dir + 'dreamt-images/'
local_dream_base_dir   = _local_dir + 'dream-base-images/'

local_ingest_dir       = '/Users/rcharan/Downloads/'

poll_results_loc       = f'{_local_dir}lit_app/remote_dreams.txt'

# Configuration
dream_file_type       = 'jpg'

# Utilities
def parse_file_path(file_path):
    parts     = file_path.split('/')
    base_dir  = '/'.join(parts[:-1]) + '/'
    file_name_parts = parts[-1].split('.')
    file_name = '.'.join(file_name_parts[:-1])
    file_ending = '.' + file_name_parts[-1].lower()
    return base_dir, file_name, file_ending


# Command to monitor the remote file system
def _remote_monitor_command():
    ssh_command     = 'gcloud compute ssh jupyter@flatiron'
    monitor_command = f" --command 'ls {remote_dreamt_dir}'"
    cat_command     = f' > {poll_results_loc}'

    return ssh_command + monitor_command + cat_command

# Command to fetch a file from the remote file system
def _remote_fetch_command(file_name):
    scp_command = f'gcloud compute scp jupyter@flatiron:'
    return scp_command + remote_dreamt_dir + file_name + f' {local_dreamt_dir}' + ' --compress'

# Command to put a file onto the remote file system
def _remote_put_command(file_path):
    scp_command = f'gcloud compute scp {file_path}'
    return scp_command + f' jupyter@flatiron:{remote_dream_base_dir}' + ' --compress'



# Functions to post and fetch files
def post_file(file_path, bare_name):
    return_code = os.system(_remote_put_command(file_path))
    if return_code != 0:
        print(f'{bare_name} -- WARNING: failed to post with return code {return_code}')
        return False
    else:
        print(f'{bare_name} -- Posted')
        return True

def fetch_file(file_name):
    return_code = os.system(_remote_fetch_command(file_name))
    if return_code != 0:
        print(f'WARNING: failed to fetch file {file_name} with return code {return_code}')

def dreamt_file_name(file_name):
    file_name = 'dreamt-' + file_name
    # if file_name[-3:] != dream_file_type:
        # print(f'Warning: looking for {file_name[:-3]}.{dream_file_type} instead')
    file_name = '.'.join(file_name.split('.')[:-1])
    file_name = file_name + '.' + dream_file_type
    return file_name

def wait_for_file(file_name, poll_freq = 2, initial_wait = 10, timeout = 60):
    '''
        Waits initial_wait, then starts polling every poll_freq (approx) to see
        if the file exists. If it exists, fetch the file for the local system.
        After timeout, cease attempts.

        Returns: (1) status code True if file was fetched; False otherwise.
                 (2) the time elapsed
    '''

    file_name = dreamt_file_name(file_name)

    start = datetime.now()
    time.sleep(initial_wait)
    success = False
    while True:
        # Poll the remote
        response_code = os.system(_remote_monitor_command())
        # print(f'Polled ssh with response code {response_code}')

        # Look at the poll results
        with open(poll_results_loc, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]
        if file_name in lines:
            success = True
            break
        else:
            # print(f'Files detected: {lines}')
            pass

        time.sleep(poll_freq)

        time_elapsed = (datetime.now() - start).seconds
        if time_elapsed > timeout:
            break

    if not success:
        return False, time_elapsed
    else:
        fetch_file(file_name)
        time_elapsed = (datetime.now() - start).seconds
        return True, time_elapsed


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
                if success:
                    success, time_elapsed = wait_for_file(file_name, poll_freq = 2, initial_wait = 3, timeout = 20)
                    if success:
                        print(f'{bare_name} -- Fetched in {time_elapsed}s')
                    else:
                        print(f'{bare_name} -- WARNING: failed to fetch; took {time_elapsed}s')
                print(f'\n-------------------------------------\n')
            elif file_type in ['heic', 'heis']:
                print(f'{bare_name} -- Converting HEIC to jpg')
                safe_bare_name = bare_name.replace(' ', '_')
                command = f'magick "{event.src_path}" "{local_ingest_dir}{safe_bare_name}.jpg"'
                os.system(command)

            else:
                print(f'''File doesn't appear to be a jpeg, png, or HEIC, ignoring''')



        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print(f'Noticed file {event.src_path} was modified, ignoring')

watcher = Watcher(local_ingest_dir, Handler(), 'dream-base-watcher')
watcher.run()



# if __name__ == '__main__':
    # wait_for_file('marco3.jpg', initial_wait = 0)
