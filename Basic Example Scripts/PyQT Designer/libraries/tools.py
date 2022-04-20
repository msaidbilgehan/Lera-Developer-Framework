#cython: language_level=3, boundscheck=False
"""
    Refs:
        - init_logging
            -- https://realpython.com/python-logging
        -

"""

# import libs

import os
import re
import time
import random
import shutil
import natsort
# import logging
# from glob import glob
from inspect import currentframe, getframeinfo
from stdo import stdo, get_time


time_struct = {"start": 0, "end": 0, "passed": 0}

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

def time_log(option, id="id"):
    global time_list, time_struct

    if len(time_list) == 0:
        time_list[id] = time_struct.copy()
    elif id != "id" and id not in time_list:
        time_list[id] = time_struct.copy()

    if option == "end" and time_list[id]["start"] == 0:
        stdo(2, "Start time is not initialized. Taken time will be saved as start time.")
        option = "start"

    time_list[id][option] = time.time()  # To count program start time
    if option == "end":
        time_list[id]["passed"] = (
            time_list[id]["end"] - time_list[id]["start"]) * 1000


def save_to_json(path, data, sort_keys=True, indent=4):
    global json
    import json

    with open(path, "w") as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=indent)
    return 0

def load_from_json(path):
    global json
    import json

    data = None
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
        """
        import fnmatch
        files = [
            fnmatch.filter(
                fnmatch.filter(
                    [
                        file_or_folder 
                        for file_or_folder in os.listdir(path)
                        if os.path.isfile(path + file_or_folder) 
                    ],
                    extension
                ),
                name
            ) 
            for extension in extensions
        ]
        """
        
        sub_files = [sub_files for _, _, sub_files in os.walk(path)][0]
        
        files = [
            path + sub_file 
            for sub_file in sub_files
            for extension in extensions
            if sub_file.endswith(extension) and name in sub_file
        ]
        
    else:

        """
        files = list()
        sub_files_list = [
            [
                sub_path + sub_file 
                for sub_file in sub_files
            ]
            for sub_path, sub_folders, sub_files in os.walk(path)
        ]
        
        files += [sub_files for sub_files in sub_files_list if sub_files != []]
        """
        files = list()
        files += [
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

"""
def list_files_old_1(path="", name="*", extensions=["png"], recursive=False, verbose=True): 
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    # TODO: https://dzone.com/articles/listing-a-directory-with-python

    files = list()
    if recursive:
        if path[-1] != "/":
            path = path + "/"
    else:
        if path[-1] == "/":
            path = path[:-1]

    for extension in extensions:
        files.extend(
            [
                f
                for f in glob(
                    path + "**/{}.{}".format(name, extension), recursive=recursive
                )
            ]
        )

    # RECURSİVE
    #for path in files:
    #    if path.split("/")[-2] != ""
    if verbose:
        output = "- {}".format(path)
        for subPath in files:
            subPath = subPath.replace(path, "")
            output += "\n"
            for i in range(len(path.split("/"))):
                output += "\t"
            output += "\t|- {}".format(subPath)
        stdo(1, output)
        stdo(1, "{} files found".format(len(files)))

    return files


def list_files_old_2(path="", name="", extensions=["png"], recursive=False):
    # https://dzone.com/articles/listing-a-directory-with-python
    # https://stackoverflow.com/questions/2783969/compare-string-with-all-values-in-list

    if not recursive:
        # print("!!path",path)
        import pdb
        pdb.set_trace()
        # for file in filter( lambda file_name: os.path.isfile(file_name) and name in file_name and [file_name.endswith(ext) for ext in extensions][0], os.listdir(path) ) : print(file)
        files = filter(
            lambda file_name: os.path.isfile(file_name)
            and name in file_name
            and [file_name.endswith(ext) for ext in extensions][0],
            os.listdir(path)
        )

    else:
        or_operand = "|"

        if len(extensions) == 1:
            extensions_filter = extensions[0]
        elif len(extensions) > 1:
            extensions_filter = ""
            for index in range(1, len(extensions)):
                extensions_filter += or_operand + extensions[index]
        else:
            extensions_filter = ""

        rx = re.compile(r"\.(" + extensions_filter + ")")
        files = []

        for path, dnames, fnames in os.walk(path):
            files.extend([os.path.join(path, x)
                          for x in fnames if rx.search(x)])

    return files
"""

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
            random.seed(time.time())
        else:
            random.seed(random_seed)
        if extension is not None:
            file_name = "{}_{}.{}".format(
                name, random.randint(
                    random_min_max[0], random_min_max[1]), extension
            )
        else:
            file_name = "{}_{}".format(
                name, random.randint(random_min_max[0], random_min_max[1])
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
    random.seed(seed * 1000000)

    name = get_time(2)

    if name[-1] == "_":
        name = "{}0_{}".format(name[:-1], random.randint(1000, 9999))
    else:
        name = "{}_{}".format(name, random.randint(1000, 9999))

    return "temp_image-{}".format(name)
