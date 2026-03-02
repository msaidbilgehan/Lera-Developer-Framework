import os
import time

import cv2
import numpy as np
import torch

import libs
from tools import load_from_json
from image_manipulation import draw_Rectangle, draw_Text, get_Bbox_Position_Into_Image, color_Range_Mask_Using_Palette
from image_tools import save_image
from stdo import stdo
from math_tools import coordinate_Scaling


def detect_Fiducial_Marker(image, process_methods=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):

    start_time = time.time()
    predicted_class = None
    decision = False
    result_dashboard = "Fiducial Marker:{} - Pred:[N/A] - T:{}ms".format(decision, 0)

    dict_counter_fiducial_class = {
        "circle":1,
        "plus":1,
        "square":1
    }

    temp_image = image.copy()
    dict_predicted_results = dict()
    prediction = process_methods.predict(source=image, verbose=False)[0]

    for id, cls in enumerate(prediction.boxes.cls):
        class_names = prediction.names
        predicted_class = class_names[int(cls)]
        predicted_prob = np.round(float(prediction.boxes.conf[id]), 2)
        predicted_bbox = list(map(int, prediction.boxes.xyxy[id].tolist()))
        center_x = predicted_bbox[0] + ((predicted_bbox[2] - predicted_bbox[0]) // 2)
        center_y = predicted_bbox[1] + ((predicted_bbox[3] - predicted_bbox[1]) // 2)

        dict_predicted_results[id] = dict()
        dict_predicted_results[id]['predicted_bbox'] = predicted_bbox
        dict_predicted_results[id]['predicted_class'] = predicted_class + "_" + str(dict_counter_fiducial_class[predicted_class])
        dict_predicted_results[id]['predicted_prob'] = predicted_prob
        dict_predicted_results[id]['predicted_center_coords'] = [center_x, center_y]

        start_point = [predicted_bbox[0], predicted_bbox[1]]
        end_point = [predicted_bbox[2]-predicted_bbox[0], predicted_bbox[3]-predicted_bbox[1]]
        temp_image = draw_Rectangle(temp_image, start_point, end_point, color=(0, 255, 0), thickness=1)
        temp_image = draw_Text(temp_image, text=[str(predicted_class) +"-"+ str(predicted_prob)], center_point=(predicted_bbox[0], predicted_bbox[1]-20), fontscale=1, color=(0,255,0), thickness=3)

        dict_counter_fiducial_class[predicted_class] += 1

        stdo(1, "[{}] dict_predicted_results:{}".format(
            id,
            dict_predicted_results[id]
        ))

    if predicted_class is not None:
        decision = True
    else:
        decision = False

    if flag_activate_debug_images:
        image_pack = [temp_image]
        title_pack = [
            str(counter) + "_detected_image_" + str(configuration_id)
        ]
        save_image(image_pack, path="temp_files/detect_Fiducial_Marker/", filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000
    result_dashboard = "Fiducial Marker:{} - Pred:[{}] - T:{:.2f}ms".format(decision, str(predicted_class), stop_time)

    return predicted_class, decision, result_dashboard, dict_predicted_results

def detect_Terminal_Component(image, torch_version=str(torch.__version__), process_methods=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):

    start_time = time.time()
    pred_size_decision = 0
    decision = False
    predicted_class = None
    predicted_prob = 0
    result_dashboard = "Detection:{} - Pred:[N/A] - T:{}ms".format(decision, 0)

    if int(torch_version.split(".")[0]) < 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
        image_reshape = np.transpose(image_resize, (2, 0, 1))
        image_torch = torch.from_numpy(image_reshape).unsqueeze(0)

    else:
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_resize = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
        # image_reshape = np.transpose(image_resize, (2, 0, 1))
        # image_torch = torch.from_numpy(image_reshape).unsqueeze(0)
        # print("else:::::HERE::::",  image_torch.shape)
        image_torch = image

    pred = process_methods.predict(source=image_torch, verbose=False)[0]

    list_predicted_class = list()
    for id, cls in enumerate(pred.boxes.cls):
        class_names = pred.names
        predicted_class = class_names[int(cls)]
        predicted_prob = np.round(float(pred.boxes.conf[id]), 2)
        predicted_bbox = list(map(int, pred.boxes.xyxy[id].tolist()))

        if int(torch_version.split(".")[0]) < 2:
            predicted_bbox = np.array(predicted_bbox)
            predicted_bbox = predicted_bbox.reshape(-1,2)
            predicted_bbox[:,0], predicted_bbox[:,1] = coordinate_Scaling(
                x=predicted_bbox[:,0], y=predicted_bbox[:,1],
                old_w=224, old_h=224,
                new_w=image.shape[1], new_h=image.shape[0],
                task='RESIZE',
                is_dual=False
            )
            predicted_bbox = predicted_bbox.reshape(-1)

        ## The model predicted wrong decision in some images. So bellow codes  ##
        if str(predicted_class) == 'blank':
            if (
                (predicted_bbox[2]-predicted_bbox[0]) > image.shape[1] // 2
            ) or (
                (predicted_bbox[3]-predicted_bbox[1]) > image.shape[0] // 2
            ):

                print("::::::Shape-Mismatch:::::", predicted_bbox, " | ", image.shape)

                decision = False
            else:

                print("::::::Real-Blank:::::", predicted_bbox, " | ", image.shape)

                decision = False
        elif str(predicted_class) == 'terminal':

            print("::::::Real-Terminal:::::", predicted_bbox, " | ", image.shape)

            decision = True

        list_predicted_class.append(decision)
    """
    list_predicted_class.append(decision)
    list_predicted_class = np.array(list_predicted_class)
    list_index = np.where(list_predicted_class==False)
    matched_list_index = len(list_index)
    if matched_list_index > 0:
        decision = False
    else:
        decision = True
    """

    if flag_activate_debug_images:
        image_pack = [image]
        title_pack = [
            str(counter) + "max_matched_frame" + configuration_id
        ]
        save_image(image_pack, path="temp_files/detect_Terminal_Component/", filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000

    result_dashboard = "[{}][{}] Terminal Detection:{} - Pred:[{}] - T:{:.2f}ms".format(pattern_id, configuration_id, predicted_class, predicted_prob, stop_time)

    return predicted_prob, decision, result_dashboard

def detect_Samb78_Defect(image, process_methods=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):

    start_time = time.time()
    predicted_class = None
    decision = False
    result_dashboard = "Samb78 Defect:{} - Pred:[N/A] - T:{}ms".format(decision, 0)

    dict_counter_samb78_defect_class = {
        "scratch":1,
        "sandblast":1,
        "stain":1
    }

    dict_predicted_results = dict()


    list_left = []
    list_right = []
    defect_left = 0
    defect_right = 0

    prediction = process_methods.predict(source=image, verbose=False)[0]
    for id, cls in enumerate(prediction.boxes.cls):
        class_names = prediction.names
        predicted_class = class_names[int(cls)]
        predicted_prob = np.round(float(prediction.boxes.conf[id]), 2)
        predicted_bbox = list(map(int, prediction.boxes.xyxy[id].tolist()))

        dict_predicted_results[id] = dict()
        dict_predicted_results[id]['predicted_bbox'] = predicted_bbox
        dict_predicted_results[id]['predicted_class'] = predicted_class + "_" + str(dict_counter_samb78_defect_class[predicted_class])
        dict_predicted_results[id]['predicted_prob'] = predicted_prob

        dict_counter_samb78_defect_class[predicted_class] += 1

        stdo(1, "[{}] dict_predicted_results:{}".format(
            id,
            dict_predicted_results[id]
        ))

        # xyxy'den köşe noktalarını oluştur
        x_min, y_min, x_max, y_max = map(int, prediction.boxes.xyxy[id].tolist())
        bbox_coords = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

        position = get_Bbox_Position_Into_Image(image, bbox_coords)

        if position == 'left':
            list_left.append(position)
            dict_predicted_results[id]['defect_position'] = [1, 0]
        elif position == 'right':
            list_right.append(position)
            dict_predicted_results[id]['defect_position'] = [0, 1]

    if len(list_left) == 0:
        defect_left = 1
    else:
        defect_left = 0
    if len(list_right) == 0:
        defect_right = 1
    else:
        defect_right = 0

    if predicted_class == None:
        decision = True
    else:
        decision = False

    if flag_activate_debug_images:
        image_pack = [image]
        title_pack = [
            str(counter) + "_detected_image_" + str(configuration_id)
        ]
        save_image(image_pack, path="temp_files/detect_Defect/", filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000
    predicted_classes = [v['predicted_class'] for v in dict_predicted_results.values()]
    result_dashboard = "Samb78 Defect:{} - Pred:[{}] - T:{:.2f}ms".format(decision, predicted_classes, stop_time)

    return decision, [defect_left, defect_right], result_dashboard, dict_predicted_results

def detect_Samb78_Led(image, dict_color_palette=None, configuration_id='N/A', pattern_id='0', counter=0, flag_activate_debug_images=False):

    start_time = time.time()
    decision = False
    result_dashboard = "Samb78 Led:{} - Led:[N/A] - T:{}ms".format(decision, 0)
    dict_predicted_results = dict()

    defect_left = 0
    defect_right = 0

    h, w = image.shape[:2]
    mid = w // 2
    image_left = image[:, :mid]
    image_right = image[:, mid:]

    ### Left ###
    max_matched_ratio, max_matched_frame_coords, method_dashboard = color_Range_Mask_Using_Palette(
        img=image_left,
        dict_color_palette=dict_color_palette,
        type_color='BGR',
        ranged_color='red',
        morph_kernel=[3, 3],
        rso_ratio=200,
        area_threshold=500,
        configuration_id=configuration_id,
        pattern_id='left',
        show_result=flag_activate_debug_images,
        show_specified_component=pattern_id
    )

    if max_matched_ratio > 500: # area_threshold
        defect_left = 1
        bbox_coords = np.array([
            [max_matched_frame_coords[0], max_matched_frame_coords[1]],
            [max_matched_frame_coords[0]+max_matched_frame_coords[2], max_matched_frame_coords[1]],
            [max_matched_frame_coords[0]+max_matched_frame_coords[2], max_matched_frame_coords[1]+max_matched_frame_coords[3]],
            [max_matched_frame_coords[0], max_matched_frame_coords[1]+max_matched_frame_coords[3]]
        ])
        dict_predicted_results[0] = dict()
        dict_predicted_results[0]['predicted_bbox'] = bbox_coords
        dict_predicted_results[0]['predicted_class'] = 'led_left'
        dict_predicted_results[0]['predicted_prob'] = max_matched_ratio
        dict_predicted_results[0]['led_position'] = [1, 0]
    else:
        defect_left = 0
    ### ### # ###

    ### Right ###
    max_matched_ratio, max_matched_frame_coords, method_dashboard = color_Range_Mask_Using_Palette(
        img=image_right,
        dict_color_palette=dict_color_palette,
        type_color='BGR',
        ranged_color='red',
        morph_kernel=[3, 3],
        rso_ratio=200,
        area_threshold=500,
        configuration_id=configuration_id,
        pattern_id='right',
        show_result=flag_activate_debug_images,
        show_specified_component=pattern_id
    )

    if max_matched_ratio > 500: # area_threshold
        defect_right = 1
        bbox_coords = np.array([
            [max_matched_frame_coords[0], max_matched_frame_coords[1]],
            [max_matched_frame_coords[0]+max_matched_frame_coords[2], max_matched_frame_coords[1]],
            [max_matched_frame_coords[0]+max_matched_frame_coords[2], max_matched_frame_coords[1]+max_matched_frame_coords[3]],
            [max_matched_frame_coords[0], max_matched_frame_coords[1]+max_matched_frame_coords[3]]
        ])
        dict_predicted_results[1] = dict()
        dict_predicted_results[1]['predicted_bbox'] = bbox_coords
        dict_predicted_results[1]['predicted_class'] = 'led_right'
        dict_predicted_results[1]['predicted_prob'] = max_matched_ratio
        dict_predicted_results[1]['led_position'] = [0, 1]
    else:
        defect_right = 0
    ### ### # ###


    if defect_left == 1 and defect_right == 1:
        decision = True
    else:
        decision = False

    if flag_activate_debug_images:
        image_pack = [image_left, image_right]
        title_pack = [
            str(counter) + "_detected_image_" + str(pattern_id),
            str(counter) + "_detected_image_" + str(pattern_id)
        ]
        save_image(image_pack, path="temp_files/detect_Samb78_Led/", filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000
    result_dashboard = "Samb78 Led:{} Led-Left:{} - Led-Right:{} - T:{:.2f}ms".format(decision, defect_left, defect_right, stop_time)

    return decision, [defect_left, defect_right], result_dashboard, dict_predicted_results



def detect_Product_CL1_AB(image, process_methods=None, configuration_id='N/A', counter=0, flag_activate_debug_images=False):
    start_time = time.time()
    predicted_class = None
    decision = False
    result_dashboard = "Product_AB:{} - Pred:[N/A] - T:{}ms".format(decision, 0)
    dict_predicted_results = dict()

    prediction = process_methods.predict(source=image, verbose=False)[0]
    for id, cls in enumerate(prediction.boxes.cls):
        class_names = prediction.names
        predicted_class = class_names[int(cls)]
        predicted_prob = np.round(float(prediction.boxes.conf[id]), 2)
        predicted_bbox = list(map(int, prediction.boxes.xyxy[id].tolist()))
        center_x = predicted_bbox[0] + ((predicted_bbox[2] - predicted_bbox[0]) // 2)
        center_y = predicted_bbox[1] + ((predicted_bbox[3] - predicted_bbox[1]) // 2)

        dict_predicted_results[id] = dict()
        dict_predicted_results[id]['predicted_bbox'] = predicted_bbox
        dict_predicted_results[id]['predicted_class'] = predicted_class
        dict_predicted_results[id]['predicted_prob'] = predicted_prob
        dict_predicted_results[id]['predicted_center_coords'] = [center_x, center_y]

        if flag_activate_debug_images:
            image = draw_Rectangle(image, [predicted_bbox[0], predicted_bbox[1]], [predicted_bbox[2]-predicted_bbox[0], predicted_bbox[3]-predicted_bbox[1]], color=(0,255,0), thickness=2)
            image = draw_Text(image, text=[str(predicted_class)+"-"+str(predicted_prob)], center_point=(predicted_bbox[0], predicted_bbox[1]+10), fontscale=0.5, color=(0,255,0), thickness=1)

    if len(dict_predicted_results) > 0:
        decision = True
    else:
        decision = False

    if flag_activate_debug_images and decision:
        image_pack = [image]
        title_pack = [
            f"{counter}_{int(time.time() * 1000)}_detected_product_AB_{configuration_id}"
        ]
        save_image(image_pack, path="temp_files/detect_Product_AB/", filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000
    result_dashboard = "Product_AB:{} - Pred:[{}] - T:{:.2f}ms".format(decision, str(predicted_class), stop_time)

    return predicted_class, decision, result_dashboard, dict_predicted_results
