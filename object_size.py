import os
import time

import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np

import libs
from image_manipulation import threshold, image_Make_Border, image_Calculate_Border_Range, draw_Circle, grayscale_Conversion
from image_tools import show_image, open_image, save_image
from stdo import stdo, get_time
from tools import time_log, list_files, list_folders, save_to_json, load_from_json, path_control, seppuku, remove_dir
from math_tools import coordinate_Scaling, extraction_Pixel_Values, find_Peak_Points, line_Intersection
from preprocess_image_processing import preprocess_of_Component_Direction

import torch

def detect_Object_Size(image, torch_version=str(torch.__version__), size='2', process_methods=None, process_params=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):
    
    dict_socket_pin_size = {
        '2': "2_pin",
        '3': "3_pin",
        '4': "4_pin",
        '5': "5_pin",
        '6': "6_pin",
        '8': "8_pin" 
    }
    
    start_time = time.time()
    pred_size_decision = 0
    decision = False
    result_dashboard = "Size:{} - Pred:[N/A] - T:{}ms".format(decision, 0)
    
    if int(torch_version.split(".")[0]) < 2: 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
        image_reshape = np.transpose(image_resize, (2, 0, 1))
        image_torch = torch.from_numpy(image_reshape).unsqueeze(0)
    else:
        image_torch = image
    
    prediction = process_methods.predict(source=image_torch)
    probabilities = prediction[0].probs
    class_names = prediction[0].names
    max_prob_index = probabilities.data.argmax()
    pred_size_decision = class_names[int(max_prob_index)]
    
    if pred_size_decision == dict_socket_pin_size[size]:
        decision = True
    else:
        decision = False
    
    if flag_activate_debug_images:
        image_pack = [image]
        title_pack = [
            str(counter) + "max_matched_frame" + configuration_id
        ]
        save_image(image_pack, path="temp_files/detect_Object_Size/", filename=title_pack, format="png")
        
    stop_time = (time.time() - start_time) * 1000
    result_dashboard = "Size:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f}ms".format(decision, *list(map(lambda x: x, probabilities.data.tolist())), stop_time)
    
    pred_size_decision = [key for key, value in dict_socket_pin_size.items() if value == pred_size_decision][0]

    return pred_size_decision, decision, result_dashboard