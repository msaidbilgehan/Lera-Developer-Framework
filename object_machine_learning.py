# import os
import time

import cv2
# from skimage.metrics import structural_similarity
# import matplotlib.pyplot as plt
import numpy as np

import libs
from image_manipulation import threshold, image_Make_Border, image_Calculate_Border_Range, draw_Circle, grayscale_Conversion
from image_tools import show_image, open_image, save_image
from stdo import stdo, get_time
# from tools import time_log, list_files, list_folders, save_to_json, load_from_json, path_control, seppuku, remove_dir
# from math_tools import coordinate_Scaling, extraction_Pixel_Values, find_Peak_Points, line_Intersection
# from preprocess_image_processing import preprocess_of_Component_Direction


def detect_Object_Machine_Learning(image, process_methods=None, process_params=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):

    start_time = time.time()
    result_dashboard = ""

    max_w = process_params[0]
    max_h = process_params[1]
    """
    if (image.shape[1] > max_w) or (image.shape[0] > max_h):
        image = cv2.resize(image, (max_w, max_h))

    top, bottom, left, right = image_Calculate_Border_Range(image, max_w=max_w, max_h=max_h)
    padded_image = image_Make_Border(image, border_range=[top, bottom, left, right])
    if (padded_image.shape[1] > max_w) or (padded_image.shape[0] > max_h):
        padded_image = cv2.resize(padded_image, (max_w, max_h))

    padded_image = grayscale_Conversion(padded_image)
    """
    padded_image = cv2.resize(image, (max_w, max_h))
    padded_image = grayscale_Conversion(padded_image)
    padded_image = padded_image / 255
    prediction = process_methods.predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
    prediction = np.array(prediction)[0]
    """
    index = np.where(prediction > 0.5)[0]

    if len(index) > 0:
        pred_ml_decision = index[0]
    else:
        pred_ml_decision = np.where(prediction == prediction.max())[0][0]

    if pred_ml_decision == 0:
        decision = True
    else:
        decision = False
    """
    if prediction > 0.5:
        decision = True
    else:
        decision = False
    pred_ml_decision = prediction

    if flag_activate_debug_images:
        image_pack = [image, padded_image]
        title_pack = [
            str(counter) + "_0_max_matched_frame_" + configuration_id,
            str(counter) + "_1_padded_image_" + configuration_id
        ]
        save_image(image_pack, path="temp_files/detect_Object_Machine_Learning/", filename=title_pack, format="png")

        stop_time = (time.time() - start_time) * 1000
        # stdo(1, "[{}][{}]: Size:{} | Prediction:[{:.2f} {:.2f} {:.2f}] | Time:{:.2f} ms".format(pattern_id, configuration_id, decision, prediction[0], prediction[1], prediction[2], stop_time))
        result_dashboard = "[{}][{}] ML:{} - Pred:[{:.1f}] - T:{:.2f}ms".format(pattern_id, configuration_id, decision, prediction, stop_time)

    return pred_ml_decision, decision, result_dashboard