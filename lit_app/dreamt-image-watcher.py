import os
import time
from datetime import datetime

# Remote directories
remote_dreamt_dir      = '~/art-dream/dreamt-images/'
remote_dream_base_dir  = '~/art-dream/dream-base-images/'

# Local directories
_local_dir             = '/Users/rcharan/Dropbox/Flatiron/final-project/art-dream/'
local_dreamt_dir       = _local_dir + 'dreamt-images/'
local_dream_base_dir   = _local_dir + 'dream-base-images/'

poll_results_loc       = f'{_local_dir}lit_app/remote_dreams.txt'
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
def _remote_put_command(file_name):
    scp_command = f'gcloud compute scp {local_dream_base_dir}{file_name}'
    return scp_command + f' jupyter@flatiron:{remote_dream_base_dir}' + ' --compress'



# Functions to post and fetch files
def post_file(file_name):
    return_code = os.system(_remote_put_command(file_name))
    if return_code != 0:
        print(f'WARNING: failed to post file {file_name} with return code {return_code}')

def fetch_file(file_name):
    return_code = os.system(_remote_fetch_command(file_name))
    if return_code != 0
        print(f'WARNING: failed to fetch file {file_name} with return code {return_code}')
    else:
        print(f'Fetched file {file_name}')

def wait_for_file(file_name, poll_freq = 2, initial_wait = 5, timeout = 20):
    '''
        Waits initial_wait, then starts polling every poll_freq (approx) to see
        if the file exists. If it exists, fetch the file for the local system.
        After timeout, cease attempts.

        Returns: (1) status code True if file was fetched; False otherwise.
                 (2) the time elapsed
    '''

    if not file_name.endswith('jpg'):
        print(f'Warning: looking for {file_name} as a jpg instead')
    file_name = file_name[:-3] + 'jpg'
    file_name = 'dreamt-' + file_name

    start = datetime.now()
    time.sleep(initial_wait)
    success = False
    while True:
        # Poll the remote
        response_code = os.system(_remote_monitor_command())
        print(f'Polled ssh with response code {response_code}')

        # Look at the poll results
        with open(poll_results_loc, 'r') as f:
            lines = f.readlines()

        lines = [l.strip() for l in lines]
        if file_name in lines:
            success = True
            break
        else:
            print(f'Files detected: {lines}')

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

if __name__ == '__main__':
    # print(_remote_monitor_command())
    # response_code = os.system(_remote_monitor_command())
    wait_for_file('marco3.jpg', initial_wait = 0)
