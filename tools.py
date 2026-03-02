"""
    Refs:
        - init_logging
            -- https://realpython.com/python-logging
        -

"""


import os
import re
import time
import random
from glob import glob
import platform
import shutil
import natsort
# import logging
# from glob import glob
from inspect import currentframe, getframeinfo
from stdo import stdo, get_time
from enum import Enum
import numpy as np

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
    data = dict()
    if path_control(path, is_file=True)[0]:
        global json
        import json

        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
    return data


def file_open(file_path, is_open_noexist=True):
    if file_path == "":
        file_path = "output.log"

    if is_open_noexist and not path_control(file_path, is_file=True)[0]:
        stdo(2, "File not found: {}".format(file_path))
        # os.path.isfile(file_path)
        first_create = open(file_path, "w+")
    else:
        first_create = open(file_path, "r")
    return file_path, first_create

def file_save(path, data):
    with open(path, "w") as outfile:
        outfile.write(data)


def list_folders(path="", name="", recursive=False, return_list=False):
    # https://dzone.com/articles/listing-a-directory-with-python
    # https://stackoverflow.com/questions/2783969/compare-string-with-all-values-in-list

    if not recursive:
        if return_list:

            if platform.system() == "Linux" or platform.system() == "Darwin":
            # Linux veya macOS'ta
                try:
                    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                    folders = sorted(folders)
                    return folders
                except FileNotFoundError:
                    return []

            elif platform.system() == "Windows":
                # Windows'ta
                try:
                    # output = os.popen(f'dir /b /ad "{path}"').read()
                    # folders = output.strip().split('\n')
                    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                    folders = sorted(folders)
                    return folders
                except FileNotFoundError:
                    return []
            else:
                return []  # Diğer işletim sistemleri için destek yok

        else:
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

def list_files(path="", name="", extensions=[".png"], recursive=False, is_sorted=False, reverse_sorted=False, os_sorted=False):
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

    if os_sorted:
        if is_sorted:
            files = natsort.os_sorted(files, reverse=False)
        elif reverse_sorted:
            files = natsort.os_sorted(files, reverse=True)
    else:
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

def list_Element_Remove(list_data, index=[]):

    if isinstance(index, list):
        if len(index) > 0:
            for id in index:
                list_data.pop(id)

    elif isinstance(index, np.ndarray):
        if index.any() and isinstance(list_data, list):
            for id in index:
                list_data.pop(id)

        elif index.any() and isinstance(list_data, np.ndarray):
            list_data = np.delete(list_data, index)

    return list_data

def path_control(path, is_file=False, is_directory=False):
    bool_list = list()
    if is_file:
        bool_list.append(os.path.isfile(path))
    if is_directory:
        bool_list.append(os.path.isdir(path))
    return bool_list

def make_Directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        stdo(3, string="'{}' path parameters should be in type of list format".format(path_list))
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
            stdo(3, string="'{}' path parameter should be in type of list format".format(file_path_list))
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
            stdo(3, string="'{}' path parameters should be in type of list format".format(dir_path_list))
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
        if not os.path.exists(os.path.dirname(move_file_path)):
            os.makedirs(os.path.dirname(move_file_path), exist_ok=True, mode=0o777)
        os.replace(location_file_path, move_file_path)
        last_path = location_file_path
        return 0
    except IsADirectoryError:
        stdo(3, string="'{}' path is a directory".format(last_path))
        return -1
    except FileNotFoundError:
        stdo(3, string="'{}' path is not available".format(last_path))
        return -1

def move_folder(location_folder_path, move_folder_path):
    last_path = location_folder_path

    try:
        if not os.path.exists(location_folder_path):
            raise FileNotFoundError(f"Location folder '{location_folder_path}' doesn't exist.")

        # Hedef klasör varsa, önce sil
        if os.path.exists(move_folder_path):
            shutil.rmtree(move_folder_path)

        # Hedef klasörün üst dizinleri yoksa oluştur
        os.makedirs(os.path.dirname(move_folder_path), exist_ok=True)

        # Taşıma işlemi
        shutil.move(location_folder_path, move_folder_path)

        print(f"Klasör başarıyla taşındı: {location_folder_path} -> {move_folder_path}")
        return 0

    except IsADirectoryError:
        stdo(3, string="'{}' path is a directory".format(last_path))
        return -1
    except FileNotFoundError:
        stdo(3, string="'{}' path is not available".format(last_path))
        return -1
    except Exception as e:
        stdo(3, f"Unexpected error: {e}")
        return -1

def copy_File(location_file_path, copy_file_path):
    try:
        if not os.path.exists(os.path.dirname(copy_file_path)):
            os.makedirs(os.path.dirname(copy_file_path), exist_ok=True, mode=0o777)
        shutil.copyfile(location_file_path, copy_file_path)
        return 0
    except shutil.SameFileError:
        stdo(3, string="'{}' src path and dst path are a same file".format(location_file_path))
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


def rdata_to_csv(path_read="./r_data.rdata", path_write="./csv_data.csv"):
    # https://github.com/ofajardo/pyreadr
    global pyreadr
    import pyreadr
    result = pyreadr.read_r(path_read)  # also works for Rds
    result.to_csv(path_write, index=False)

    return result  # extract the pandas data frame for object df1


def csv_to_rdata(path_read="./csv_data.csv", path_write="./rdata.rdata"):
    # https://github.com/ofajardo/pyreadr
    global pyreadr
    import pyreadr
    pyreadr.write_rdata(
        path_write,
        pd.read_csv(path_read),
        df_name="dataset"
    )
    # pyreadr.write_rds("test.Rds", df)

def generate_Matrix_Prefix(rows=0, cols=0, base_prefix="", base_number=0):

    list_generate_prefix = [[f'{base_prefix}{base_number + row * cols + col+1:02d}' for col in range(cols)] for row in range(rows)]

    """
    list_generate_prefix = []
    for row in range(rows):
        row_leds = []
        for col in range(cols):
            if isinstance(base_prefix, list):
                prefix_str = base_prefix[col % len(base_prefix)]
            else:
                prefix_str = base_prefix
            led_label = f'{prefix_str}{row * cols + col + 2:02d}'
            row_leds.append(led_label)
        list_generate_prefix.append(led_label)
    """

    return list_generate_prefix

def filter_Dictionary_Elements(data, probability_threshold=0.5):
    filtered_data = {}
    for key, value in data.items():
        if value['probability'] > probability_threshold:
            label = value['label']
            if label not in filtered_data:
                filtered_data[label] = []
            filtered_data[label].append(value)

    average_probabilities = {}
    for label, entries in filtered_data.items():
        average_probabilities[label] = sum(entry['probability'] for entry in entries) / len(entries)

    highest_average_label = max(average_probabilities, key=average_probabilities.get)

    final_group = {key: value for key, value in data.items() if value['label'] == highest_average_label and value['probability'] > probability_threshold}
    return final_group

def match_Dictionaries_Elements(conf=None, pred=None, axiscoords=[], tolerance=10):

    matched_detectedCoords = np.array([])

    if conf or pred:

        if str(axiscoords) in conf["fiducial_area"]:

            for id_conf, [classes, centers] in enumerate(conf["fiducial_area"][str(axiscoords)].items()):

                for id_pred, pred_data in enumerate(pred.values()):

                    if (
                        classes.split("_")[0] == pred_data['predicted_class'].split("_")[0]
                        and
                        abs(centers['coord_center'][0] - pred_data["predicted_center_coords"][0]) <= tolerance
                        and
                        abs(centers['coord_center'][1] - pred_data["predicted_center_coords"][1]) <= tolerance
                    ):
                        matched_detectedCoords = np.append(
                            matched_detectedCoords,
                            [
                                str(pred_data['predicted_class']) + ":" + str(pred_data["predicted_center_coords"]),
                                str(pred_data['predicted_bbox'])
                            ]
                        )

    matched_detectedCoords = matched_detectedCoords.flatten().tolist()

    return matched_detectedCoords

