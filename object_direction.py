import time

import cv2
import numpy as np

import libs
from image_manipulation import grayscale_Conversion
from image_tools import save_image
from stdo import stdo

import torch

def detect_Object_Direction(
        list_image,
        torch_version=str(torch.__version__),
        direction='up',
        method='C',
        process_methods=None,
        process_params=None,
        configuration_id='N/A',
        pattern_id='0',
        counter=0,
        flag_activate_debug_images=False
    ):

    dict_model = {key: process_methods[0] for key in ['C', 'c']}
    dict_model.update({key: process_methods[1] for key in ['K', 'k']})
    dict_model.update({key: process_methods[2] for key in ['D', 'd', 'Z', 'z']})
    dict_model.update({key: process_methods[3] for key in ['T', 't', 'TR', 'tr', 'L', 'l']})
    dict_model.update({key: process_methods[3] for key in ['S', 's']})  # TODO:process_methods[3] -> Change process_methods[4] after training bitron sensor s.

    dict_direction = {
        "0": 'down',
        "1": 'left',
        "2": 'right',
        "3": 'up'
    }

    dict_direction_binary = {
        "0": 'left',
        "1": 'right'
    }

    start_time = time.time()
    pred_direction_decision = 'up'
    decision = False
    result_dashboard = "Direct:{} - Pred:[N/A] - T:{} ms".format(decision, 0)

    # method =>  C:capacitor | K:socket | D:diot | S:sensor | TR/L:transformer

    if (method == 'D' or method == 'd') or (method == 'Z' or method == 'z'):

        dict_direction = {
            "0": 'left',
            "1": 'up',
            "2": 'down',
            "3": 'right'
        }

        image_rgb = cv2.cvtColor(list_image[1], cv2.COLOR_BGR2RGB)
        image_resize = cv2.resize(image_rgb, (80, 80)).astype(np.float32) / 255.0
        image_gray = grayscale_Conversion(image_resize)
        image_reshape = np.transpose(image_gray, (0, 1))
        image_torch = torch.from_numpy(image_reshape).unsqueeze(0)

        prediction = dict_model[method](image_torch)[0]
        max_prob_index = int(prediction.argmax())
        pred_direction_decision = dict_direction[str(max_prob_index)]

    else:
        if int(torch_version.split(".")[0]) < 2:
            image_rgb = cv2.cvtColor(list_image[1], cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_rgb, (224, 224)).astype(np.float32) / 255.0
            image_reshape = np.transpose(image_resize, (2, 0, 1))
            image_torch = torch.from_numpy(image_reshape).unsqueeze(0)
        else:
            image_torch = list_image[1]

        prediction = dict_model[method].predict(source=image_torch, verbose=False)
        probabilities = prediction[0].probs
        class_names = prediction[0].names
        max_prob_index = probabilities.data.argmax()
        pred_direction_decision = class_names[int(max_prob_index)]

    if pred_direction_decision == direction:
        decision = True
    else:
        decision = False

    if flag_activate_debug_images:
        image_pack = [list_image[1]]
        title_pack = [
            str(counter) + "_max_matched_frame_" + configuration_id
        ]
        save_image(image_pack, path="temp_files/detect_Object_Direction/" + method, filename=title_pack, format="png")

    stop_time = (time.time() - start_time) * 1000
    # result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f}ms".format(decision, probabilities.data[0], probabilities.data[1], probabilities.data[2], probabilities.data[3], stop_time)

    if (method == 'D' or method == 'd') or (method == 'Z' or method == 'z'):
        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f} ms".format(decision, *prediction.tolist(), stop_time)
    else:
        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f} ms".format(decision, *list(map(lambda x: x, probabilities.data.tolist())), stop_time)

    """
    if (method == 'S' or method == 's') or (method == 'S' or method == 's'):

        max_w = process_params[0]
        max_h = process_params[1]

        padded_image = cv2.resize(list_image[1], (max_w, max_h))
        padded_image = grayscale_Conversion(padded_image)
        # padded_image = padded_image / 255
        # prediction = process_methods[1].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
        prediction = process_methods[4].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
        prediction = np.array(prediction)
        index = np.where(prediction > 0.5)[0]

        if len(index) > 0:
            pred_direction_decision = dict_direction_binary[str(index[0])]
        else:
            pred_direction_decision = dict_direction_binary[str(np.where(prediction == prediction.max())[0][0])]

        if pred_direction_decision == direction:
            decision = True
        else:
            decision = False

        if flag_activate_debug_images:
            image_pack = [list_image[1], padded_image]
            title_pack = [
                str(counter) + "_0_max_matched_frame_" + configuration_id,
                str(counter) + "_1_padded_image_" + configuration_id
            ]
            save_image(image_pack, path="temp_files/detect_Object_Direction/" + method, filename=title_pack, format="png")

            # stdo(1, "[{}][{}]: Direction:{} | Prediction:[{:.2f} {:.2f} {:.2f} {:.2f}] | Time:{:.2f} ms".format(pattern_id, configuration_id, decision, prediction[0], prediction[1], prediction[2], prediction[3], stop_time))

        stop_time = (time.time() - start_time) * 1000

        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f}] - T:{:.2f}ms".format(decision, prediction[0], prediction[1], stop_time)
        # result_dashboard = "Direct:{} - Pred:[{}] - T:{:.2f} ms".format(decision, str(*np.round(prediction, 1)), stop_time)
    """

    return pred_direction_decision, decision, result_dashboard