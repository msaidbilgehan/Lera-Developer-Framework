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


def detect_Object_Size(image, size='2', process_methods=None, process_params=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):
    
    dict_socket_pin_size = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "8": 5 
    }
    
    start_time = time.time()
    pred_size_decision = False
    decision = False
    result_dashboard = "Size:{} - Pred:[N/A] - T:{}ms".format(decision, 0)
    
    max_w = process_params[0]
    max_h = process_params[1]
    
    padded_image = cv2.resize(image, (max_w, max_h))
    padded_image = grayscale_Conversion(padded_image)
    # padded_image = padded_image / 255
    # prediction = process_methods.predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
    prediction = process_methods.predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
    prediction = np.array(prediction)
    index = np.where(prediction > 0.5)[0]

    if len(index) > 0:
        pred_size_decision = index[0]
    else:
        pred_size_decision = np.where(prediction == prediction.max())[0][0]
    
    if pred_size_decision == dict_socket_pin_size[size]:
        decision = True
    else:
        decision = False
    
    if flag_activate_debug_images:
        image_pack = [image, padded_image]
        title_pack = [
            str(counter) + "_0_max_matched_frame_" + configuration_id, 
            str(counter) + "_1_padded_image_" + configuration_id
        ]
        save_image(image_pack, path="temp_files/detect_Object_Size/", filename=title_pack, format="png")
        
        # stdo(1, "[{}][{}]: Size:{} | Prediction:[{:.2f} {:.2f} {:.2f}] | Time:{:.2f} ms".format(pattern_id, configuration_id, decision, prediction[0], prediction[1], prediction[2], stop_time))
        
    stop_time = (time.time() - start_time) * 1000
    
    result_dashboard = "Size:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f}ms".format(decision, *np.round(prediction, 1), stop_time)
    # result_dashboard = "Size:{} - Pred:[{}] - T:{:.2f} ms".format(decision, str(*np.round(prediction, 1)), stop_time)
    
    return pred_size_decision, decision, result_dashboard