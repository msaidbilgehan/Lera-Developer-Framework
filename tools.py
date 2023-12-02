"""
    Refs:
        - init_logging
            -- https://realpython.com/python-logging
        -

"""


import os
import re
import time
import shutil
import natsort
# import logging
# from glob import glob
from inspect import currentframe, getframeinfo
from stdo import stdo, get_time
from enum import Enum
import secrets

# Camera Instance Flags
class TIME_FLAGS(Enum):
    START = -1
    END = 0
    PASSED = 1


time_struct = {
    TIME_FLAGS.START: 0, 
    TIME_FLAGS.END: 0, 
    TIME_FLAGS.PASSED: 0
}
time_list = {}


def seppuku():

    # TODO: Only for temporary solution, Solve the Undestroyed GUI and Progress!!!
    # https://stackoverflow.com/questions/19782075/how-to-stop-terminate-a-python-script-from-running
    stdo(2, "Forced to close the app. Do not forget to close clean!!!")
    
    import signal, os, sys

    # https://docs.python.org/3/library/sys.html#sys.platform
    """
    System          |   platform value
    -----------------------
    AIX             |   'aix'
    Linux           |   'linux'
    Windows         |   'win32'
    Windows/Cygwin  |   'cygwin'
    macOS           |   'darwin'
    """
    platform = sys.platform
    try:
        if platform.startswith('win32'):
            pid = os.getpid()
            stdo(1, "切腹 ! (https://en.wikipedia.org/wiki/Seppuku)")
            os.kill(int(pid), signal.SIGTERM)
            """
            import subprocess
            print(sys.argv[0].split("\\"))
            subprocess = subprocess.Popen(
                ['ps', '-A'], stdout=subprocess.PIPE)
            output, error = subprocess.communicate()
            stdo(1, output)
            target_process = sys.argv[0]
            for line in output.splitlines():
                if target_process in str(line):
                    pid = int(line.split(None, 1)[0])
                    os.kill(pid, 9)
            """
        else:
            #for line in os.popen("ps ax | grep " + sys.argv[0] + " | grep -v grep"):
            #fields = line.split()
            #pid = fields[0]
            pid = os.getpid()
            stdo(1, "切腹 ! (https://en.wikipedia.org/wiki/Seppuku)")
            os.kill(int(pid), signal.SIGKILL)
    except Exception as error:
        stdo(3, "Exit Error: {}".format(error))

def get_OS():
    # https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on
    """
    Linux: Linux
    Mac: Darwin
    Windows: Windows
    """
    from platform import system
    return system()

    """
    OS = ""

    if os.name == "nt":
        OS = "W"  # Windows
    else:
        OS = "P"  # Posix (Linux etc...)

    return OS
    """
    
def time_log(id="id", flag=TIME_FLAGS.END):
    global time_list, time_struct

    if len(time_list) == 0:
        time_list[id] = time_struct.copy()
    elif id != "id" and id not in time_list:
        time_list[id] = time_struct.copy()

    if time_list[id][TIME_FLAGS.START] == 0:
        #stdo(2, "Start time is not initialized. Taken time will be saved as start time.")
        flag = TIME_FLAGS.START

    time_list[id][flag] = time.time()  # To count program start time

    if flag == TIME_FLAGS.END:
        time_list[id][TIME_FLAGS.PASSED] = (
            time_list[id][TIME_FLAGS.END] - time_list[id][TIME_FLAGS.START]
        ) * 1000


def save_to_json(path, data, sort_keys=True, indent=4):
    global json
    import json

    with open(path, "w") as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=indent)
    return 0

def load_from_json(path):
    data = None
    if path_control(path, is_file=True)[0]:
        global json
        import json

        with open(path, "r") as json_file:
            data = json.load(json_file)
    return data


def file_open(file_path, is_open_nonexist=True, is_return_file=False):

    if file_path == "":
        file_path = "output.log"

    if is_open_nonexist and not path_control(file_path, is_file=True)[0]:
        stdo(2, "File not found: {}".format(file_path))
        # os.path.isfile(file_path)
        first_create = open(file_path, "w+")
        if not is_return_file:
            first_create.close()
    return file_path, first_create

def file_save(path, data):
    with open(path, "w") as outfile:
        outfile.write(data)


def list_folders(path="", name="", recursive=False):
    # https://dzone.com/articles/listing-a-directory-with-python
    # https://stackoverflow.com/questions/2783969/compare-string-with-all-values-in-list

    if not recursive:
        dir_list = filter(
            lambda dir_name: os.path.isdir(dir_name)
            and name in dir_name,
            os.listdir(path)
        )

    else:
        dir_list = []
        rx = re.compile("(" + name + ")")

        for path, dnames, fnames in os.walk('.'):
            dir_list.extend([os.path.join(path, x)
                             for x in dnames if rx.search(x)])

    return dir_list

def list_files(path="", name="", extensions=[".png"], recursive=False, is_sorted=False, reverse_sorted=False):
    # https://dzone.com/articles/listing-a-directory-with-python
    # https://stackoverflow.com/questions/2783969/compare-string-with-all-values-in-list
    # https://www.geeksforgeeks.org/python-remove-empty-strings-from-list-of-strings/
    if not recursive:
        walk = [sub_files for _, _, sub_files in os.walk(path)]
        if len(walk) > 0:
            sub_files = walk[0]
        else:
            sub_files = walk
        files = [
            path + sub_file 
            for sub_file in sub_files
            for extension in extensions
            if sub_file.endswith(extension) and name in sub_file
        ]
    else:
        files = [
            sub_files for sub_files in [
                [
                    sub_path + "/" + sub_file 
                    for extension in extensions
                    for sub_file in sub_files
                    if sub_file.endswith(extension) and name in sub_file
                ]
                for sub_path, sub_folders, sub_files in os.walk(path)
            ] 
            if sub_files != []
        ]
    
    if is_sorted:
        files = natsort.natsorted(files, reverse=False)
    elif reverse_sorted:
        files = natsort.natsorted(files, reverse=True)
    return files

def list_files_to_dict(themes_subtree, themes_dict = dict()):
    for themes in themes_subtree:
        for theme in themes:
            key = theme.split("/")[-1].split(".")[0]
            themes_dict[key] = theme
    return themes_dict

def dict_to_list(dict, is_only_value):

    list_keys_passed = list()
    if is_only_value:
        [list_keys_passed.append([key, value]) for key, value in dict.items()]
    else:
        [list_keys_passed.append([value]) for key, value in dict.items()]

    """
    dict_list = list()
    for key, value in dict.items():
        if is_only_value:
            dict_list.append(value)
        else:
            dict_list.append([key, value])
    return dict_list
    """
    return list_keys_passed

def list_to_dict(list_data): # , is_key_value=False):
    dictionary = dict()

    """
    if is_key_value:
        [dictionary[index] = list_data[index] for index in range(list_data)]
        [dictionary.append([list_data_pack]) for list_data_pack in list_data]
    else:
    """
    id = 1
    for data_pack in list_data:
        dictionary[id] = data_pack
        id += 1
    return dictionary


def path_control(path, is_file=True, is_directory=True):
    bool_list = list()
    if is_file:
        bool_list.append(os.path.isfile(path))
    if is_directory:
        bool_list.append(os.path.isdir(path))
    return bool_list

def remove(path_list, only_if_empty=True, ignore_errors=False):
    if type(path_list) is list:
        file_path_list = list()
        dir_path_list = list()
        
        for path in path_list:
            is_file, is_dir = path_control(path)
            if is_file:
                file_path_list.append(path)
            elif is_dir:
                dir_path_list.append(path)

        remove_file(file_path_list)
        remove_dir(dir_path_list, only_if_empty=only_if_empty, ignore_errors=ignore_errors)

        return 0
    else:
        stdo(3, string="'{}' path parametes should be in type of list format".format(path_list))
        return -1

def remove_file(file_path_list):
    last_path = None
    try:
        if type(file_path_list) is list:
            for path in file_path_list:
                last_path = path
                os.remove(last_path)
            return 0
        else:
            stdo(3, string="'{}' path parametes should be in type of list format".format(file_path_list))
            return -1
    except IsADirectoryError:
        stdo(3, string="'{}' path is a directory".format(last_path))
        return -1
    except FileNotFoundError:
        stdo(3, string="'{}' path is not available".format(last_path))
        return -1

def remove_dir(dir_path_list, only_if_empty=True, ignore_errors=False):
    last_dir_path = None
    try:
        if type(dir_path_list) is list:
            for dir_path in dir_path_list:
                last_dir_path = dir_path
                stdo(1, string="'{}' path deleting...".format(last_dir_path))
                if only_if_empty:
                    os.rmdir(last_dir_path)
                else:
                    shutil.rmtree(last_dir_path, ignore_errors, onerror=None)
                # stdo(1, string="'{}' path deleted".format(last_dir_path))
            return 0
        else:
            stdo(3, string="'{}' path parametes should be in type of list format".format(dir_path_list))
            return -1

    except FileNotFoundError:
        stdo(3, string="'{}' path is not available".format(last_dir_path))
        return -1

    except OSError as error:
        stdo(
            3,
            "Error Occurred while working in 'remove_dir': {}".format(
                error.__str__()
            ),
            getframeinfo(currentframe()),
        )
        return -1

def rename_file(path, old_name, new_name):
    if path[-1] != '/' or path[-1] != '\\':
        path += '/'
       
    if path_control(path + new_name, is_file=True)[0]:
        stdo(2, f"New file name '{new_name}' already exists in '{path}'")
        return path + old_name
    """
    elif new_name.isalnum():
        stdo(2, f"New file name '{new_name}' should be alphanumeric.")
        return path + old_name
    """
    os.rename(path + old_name, path + new_name)
    return path + new_name


def move_file(location_file_path, move_file_path):
    last_path = None
    try:
        os.replace(location_file_path, move_file_path)
        last_path = location_file_path
        return 0
    except IsADirectoryError:
        stdo(3, string="'{}' path is a directory".format(last_path))
        return -1
    except FileNotFoundError:
        stdo(3, string="'{}' path is not available".format(last_path))
        return -1

def get_file_name(
    name="file",
    extension=None,
    with_time=True,
    random_seed=None,
    random_min_max=[0, 1024],
):
    if with_time:
        if extension is not None:
            file_name = "{}_{}.{}".format(name, get_time(level=2), extension)
        else:
            file_name = "{}_{}".format(name, get_time(level=2))
    else:
        if random_seed is None:
            secrets.SystemRandom().seed(time.time())
        else:
            secrets.SystemRandom().seed(random_seed)
        if extension is not None:
            file_name = "{}_{}.{}".format(
                name, secrets.SystemRandom().randint(random_min_max[0], random_min_max[1]), extension
            )
        else:
            file_name = "{}_{}".format(
                name, secrets.SystemRandom().randint(random_min_max[0], random_min_max[1])
            )
    return file_name

def name_parsing(
    filePath, separate=False, separator=".", maxSplit=-1
):  # parsing name from file path
    file = filePath.split("/")[-1]
    extension = file.split(".")[-1]
    name = file.strip("." + extension)
    if separate:
        # https://www.programiz.com/python-programming/methods/string/strip
        name = name.split(separator, maxSplit)
    return name, extension

def get_temp_name(seed=None):
    if seed is None:
        seed = get_time(level=0)
    secrets.SystemRandom().seed(seed * 1000000)

    name = get_time(2)

    if name[-1] == "_":
        name = "{}0_{}".format(name[:-1], secrets.SystemRandom().randint(1000, 9999))
    else:
        name = "{}_{}".format(name, secrets.SystemRandom().randint(1000, 9999))

    return "temp_image-{}".format(name)

def str_to_bool(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str.lower() in ("no", "false", "f", "n", "0"):
        return False
    # else:
    #     raise argparse.ArgumentTypeError("Boolean value expected.")

def get_Functions_From_Library(library):
    from inspect import isfunction
    return [f for f in vars(library).values() if isfunction(f)]
