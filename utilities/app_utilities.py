import glob
import os

# Streamlit tends to run files from unknown locations so I have hardcoded these
dream_base_dir         = '/Users/rcharan/Downloads/'
dreamt_dir             = '/Users/rcharan/Dropbox/Flatiron/final-project/art-dream/dreamt-images/'
remote_dreamt_dir      = '~/art-dream/dreamt-images/'
remote_dream_base_dir  = '~/art-dream/dream-base-images/'
dream_file_type = 'jpg'

# Alias I am so so sorry
local_dreamt_dir       = dreamt_dir
local_dream_base_dir   = dream_base_dir
local_ingest_dir       = dream_base_dir

def get_available_base_files(num_most_recent = None, base_dir = dream_base_dir):
    list_of_files = glob.glob(base_dir + '*.jpg')
    list_of_files+= glob.glob(base_dir + '*.png')
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

def check_for_dream(file_name, dreamt_dir = dreamt_dir):
    return os.path.exists(dreamt_dir+dreamt_file_name(file_name))

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

def get_undreamt_files(base_dir = dream_base_dir, dreamt_dir = dreamt_dir):
    undreamt = []
    for file_name in get_available_base_files(base_dir = base_dir):
        if not check_for_dream(file_name, dreamt_dir = dreamt_dir):
            undreamt.append(file_name)
    return undreamt


# Command to monitor the remote file system
def remote_monitor_command(remote_dir, poll_results_loc):
    ssh_command     = 'gcloud compute ssh jupyter@flatiron'
    monitor_command = f" --command 'ls {remote_dir}'"
    cat_command     = f' > {poll_results_loc}'

    return ssh_command + monitor_command + cat_command

# Command to fetch a file from the remote file system
def _remote_fetch_command(file_name):
    scp_command = f'gcloud compute scp jupyter@flatiron:'
    return scp_command + f'"{remote_dreamt_dir}{file_name}"' + f' {local_dreamt_dir}' + ' --compress'

# Command to put a file onto the remote file system
def _remote_put_command(file_path):
    scp_command = f'gcloud compute scp "{file_path}"'
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
        return False
    else:
        return True
