import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style

import libs
from tools import load_from_json
from image_manipulation import create_Mask, draw_Rectangle
from image_tools import save_image
from stdo import stdo
from math_tools import coordinate_Scaling, bbox_IOU
from preprocess_image_processing import preprocess_of_Template_Matching
from template_matching import MATCHING, APPLICABLE_ALGORITHMS as applicable_algorithms_template_matching
from window_slider import operation_Window_Slider
from object_detection import detect_Product_CL1_AB
from shapely.geometry import Polygon, box


def crop_Roi(image, bbox):

    startx = bbox[0] - 20
    starty = bbox[1] - 20
    endx = bbox[2] + bbox[0] + 5
    endy = bbox[3] + bbox[1] + 20

    sample = image[starty:endy, startx:endx]
    return sample

IP_COLORS = {
    "192.168.60.70": Fore.BLUE,
    "192.168.60.71": Fore.GREEN,
    "192.168.60.72": Fore.MAGENTA,
    # Diğer IP'ler için ekleme yapabilirsin
}

def object_Tracking(org_image=None, processed_image=None, template_matching_image=None, cam_ip="192.168.60.70", product_id="0", counter=0, detected=False, cycle_time=0.0, current_dict_value={}, flag_activate_debug_images=False, template_result=None, matcher=None, process_methods=None, person_bboxes=None):
    start_time = time.time()
    display_image = org_image.copy()

    if 'last_count_time' not in current_dict_value:
        current_dict_value['last_count_time'] = 0

    def any_person_overlaps(target_bbox_xyxy, iou_threshold=0.10):
        if person_bboxes is None or not person_bboxes:
            return 0.0
        tb = target_bbox_xyxy  # sade: xyxy formatı varsayılır
        max_iou = 0.0
        for pb in person_bboxes:
            nb = None
            # Destek: dict veya liste, sade: yalnızca xyxy kabul
            if isinstance(pb, dict):
                bb = pb.get('bbox') or pb.get('person_bbox') or None
                if bb is not None:
                    nb = bb
            elif isinstance(pb, (list, tuple)):
                # CSV tarzı: [ts, cam_ip, id, bbox, posture]
                if len(pb) >= 4 and isinstance(pb[3], (list, tuple)):
                    nb = pb[3]
                elif len(pb) == 4:
                    nb = pb
            if nb is None or len(nb) != 4:
                continue
            iou = bbox_IOU(tb, nb)
            if iou > max_iou:
                max_iou = iou
        return max_iou

    ############################
    ########### YOLO ###########
    ############################

    if current_dict_value.get("use_YOLO", False):
        yolo_overlap_suppressed = False
        yolo_overlap_iou = 0.0
        yolo_overlap_flag = False
        predicted_class, decision, result_dashboard, dict_predicted_results = detect_Product_CL1_AB(
            org_image,
            process_methods=process_methods,
            configuration_id="CL1_AB",
            counter=counter,
            flag_activate_debug_images=False
        )

        # Threshold filtresi
        prob_threshold = float(current_dict_value.get("yolo_prob_threshold", 0.5))
        threshold_results = {k: v for k, v in dict_predicted_results.items() if float(v['predicted_prob']) >= prob_threshold}

        # Alan filtresi parametreleri
        bbox_coords = np.array(current_dict_value['bbox_coords'], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(bbox_coords)
        gt_bbox_rect = [x, y, x+w, y+h]

        # Class filtresi
        expected_class = current_dict_value.get("expected_class")

        filtered_results = {}
        for k, v in threshold_results.items():

            # Ürün Filtresi       
            predicted_class = v['predicted_class']
            if predicted_class == expected_class:
                # filtered_results[k] = v
            
                # Alan Filtresi
                # if filtered_results:
                if expected_class == 'A':
                    iou = bbox_IOU(v['predicted_bbox'], gt_bbox_rect)
                elif expected_class == 'B':
                    iou = bbox_IOU(v['predicted_bbox'], gt_bbox_rect)
                if iou > 0.4:
                    if expected_class == 'A':
                        filtered_results[k] = v
                    elif expected_class == 'B':
                        filtered_results[k] = v
                
                    # if cam_ip == '192.168.60.70' and expected_class == 'B':
                    #     stdo(1, f"[object_Tracking][{cam_ip}][{product_id}] IOU:{iou:.2f} | p-class:{predicted_class} - e-class:{expected_class} | prob:{v['predicted_prob']:.2f} -> D:{detected} | C:{counter}")

        if len(filtered_results) > 1:
            max_id = max(filtered_results, key=lambda k: float(filtered_results[k]['predicted_prob']))
            filtered_results = {max_id: filtered_results[max_id]}

        decision = len(filtered_results) > 0
        if decision and not detected:
            detected = True

        if not decision and detected:
            now = time.time()
            ignore_overlap = True
            # YOLO yokken ROI ile, varsa son ROI ile kontrol edelim
            # YOLO kararsızsa ürün ROI'yi kullan
            bbox_coords = np.array(current_dict_value['bbox_coords'], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(bbox_coords)
            product_roi_xyxy = [x, y, x + w, y + h]
            overlap_iou = any_person_overlaps(product_roi_xyxy) if ignore_overlap else 0.0
            overlapping = overlap_iou >= 0.10
            yolo_overlap_iou = overlap_iou
            yolo_overlap_flag = overlapping
            if ignore_overlap and overlapping:
                # Çakışma varken sayma ve detected durumunu koru
                detected = True
                yolo_overlap_suppressed = True
            else:
                if now - current_dict_value['last_count_time'] > 10:
                    counter += 1
                    current_dict_value['last_count_time'] = now
                    detected = False
                else:
                    detected = False

        # predicted_classes = [v['predicted_class'] for v in filtered_results.values()]
        # predicted_probs = [float(v['predicted_prob']) for v in filtered_results.values()]
        # stdo(1, f"[object_Tracking][YOLO] cam_ip: {cam_ip}, product_id: {product_id}, predicted_classes: {predicted_classes}, predicted_probs: {predicted_probs}, decision: {decision}, detected: {detected}")

        # all_predicted_classes = [v['predicted_class'] for v in dict_predicted_results.values()]
        # all_predicted_probs = [float(v['predicted_prob']) for v in dict_predicted_results.values()]
        # stdo(1, f"[object_Tracking][YOLO][ALL] cam_ip: {cam_ip}, product_id: {product_id}, predicted_classes: {all_predicted_classes}, predicted_probs: {all_predicted_probs}")

        for id in filtered_results.keys():
            display_image = draw_Rectangle(
                display_image,
                [filtered_results[id]['predicted_bbox'][0], filtered_results[id]['predicted_bbox'][1]],
                [filtered_results[id]['predicted_bbox'][2] - filtered_results[id]['predicted_bbox'][0], filtered_results[id]['predicted_bbox'][3] - filtered_results[id]['predicted_bbox'][1]],
                color=(0, 255, 0), thickness=2
            )

        # if detected:
        #     save_image(display_image, f"output/yolo_detected_{id}.png")
        # cv2.polylines(display_image, [cv2.convexHull(np.array(current_dict_value['bbox_coords']))], True, (255,0,0), 5)
        draw_Rectangle(
            display_image,
            (x, y), (w, h),
            color=(255,0,0), thickness=2
        )


        ip_color = IP_COLORS.get(cam_ip, Fore.WHITE)
        if current_dict_value.get("use_YOLO", False):
            pred_prob = "-"
            if filtered_results:
                pred_prob = max([float(v['predicted_prob']) for v in filtered_results.values()])
            # Daha önce hesaplanan değerleri kullan
            yolo_log_iou = yolo_overlap_iou if 'yolo_overlap_iou' in locals() else 0.0
            yolo_log_overlap = yolo_overlap_flag if 'yolo_overlap_flag' in locals() else False
            stdo(
                1,
                f"{Fore.WHITE}[object_Tracking]{Style.RESET_ALL}"
                f"[{ip_color}{cam_ip}{Style.RESET_ALL}][{Fore.YELLOW}{product_id}{Style.RESET_ALL}] "
                f"{Fore.MAGENTA}[YOLO]{Style.RESET_ALL} "
                f"decision:{Fore.GREEN if decision else Fore.RED}{decision}{Style.RESET_ALL}, "
                f"detected:{Fore.GREEN if detected else Fore.RED}{detected}{Style.RESET_ALL}, "
                f"pred_prob:{Fore.CYAN}{pred_prob}{Style.RESET_ALL}, "
                f"overlap_iou:{Fore.CYAN}{yolo_log_iou:.2f}{Style.RESET_ALL}, "
                f"overlap:{Fore.GREEN if yolo_log_overlap else Fore.RED}{yolo_log_overlap}{Style.RESET_ALL}, "
                f"counter:{Fore.BLUE}{counter}{Style.RESET_ALL}"
            )
            return counter, detected, result_dashboard, display_image

    ############################
    ############################
    ############################


    masked_roi = create_Mask(bbox_coords=current_dict_value['bbox_coords'], image_shape=processed_image.shape, is_line=current_dict_value['is_line'])
    bitwise_roi = cv2.bitwise_and(processed_image, masked_roi)
    count_pixel = cv2.countNonZero(bitwise_roi) 

    # Template matching opsiyonel
    template_threshold = current_dict_value.get("template_threshold", 0.7)  
    template_result = None
    template_detected = False 

    if current_dict_value.get("use_template_matching", False):
        ref_template = current_dict_value.get("template_image", None)
        template_bbox = current_dict_value.get("template_bbox", None)

        if ref_template is not None and template_bbox is not None:
            try:
                bbox_coords = np.array(template_bbox, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(bbox_coords)
                mask = np.zeros(ref_template.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [bbox_coords], 255)
                combine = cv2.bitwise_and(ref_template, ref_template, mask=mask)
                ref_roi = combine[y:y+h, x:x+w]

                bbox = [x, y, w, h]
                sample_roi = crop_Roi(template_matching_image, bbox)

                ref_pp = preprocess_of_Template_Matching(ref_roi, method=7, pattern_id="ref", configuration_id=0, counter=0, flag_activate_debug_images=False)
                sample_pp = preprocess_of_Template_Matching(sample_roi, method=7, pattern_id="sample", configuration_id=0, counter=0, flag_activate_debug_images=False)
                max_matched_ratio, _, _ = operation_Window_Slider(
                    refImage=ref_pp,
                    sampleImage=sample_pp,
                    window_step_size=4,
                    scale_percent=100,
                    process_methods=matcher.compute,
                    process_params=[applicable_algorithms_template_matching.NORMALIZED_COSINE_COEFFICIENT],
                    ratio_threshold=template_threshold, 
                    configuration_id="config_01",
                    pattern_id="product_roi",
                )
                template_result = max_matched_ratio
                template_detected = template_result > template_threshold
            except Exception as e:
                stdo(1, "[ERROR] Template matching exception:", e)
                template_result = None
        else:
            stdo(1, "[ERROR] ref_template or template_bbox is None!")

    pixel_detected = count_pixel > current_dict_value['min_white_pixels'] 

    # stdo(1, f"[object_Tracking] pixel_detected: {pixel_detected}, template_detected: {template_detected}")

    tm_overlap_suppressed = False
    tm_overlap_iou = 0.0
    tm_overlap_flag = False
    if current_dict_value.get("use_template_matching", False):
        if (pixel_detected and template_detected) and not detected:
            detected = True
        if not (pixel_detected and template_detected) and detected:
            now = time.time()
            ignore_overlap = True
            bbox_coords = np.array(current_dict_value['bbox_coords'], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(bbox_coords)
            product_roi_xyxy = [x, y, x + w, y + h]
            overlap_iou = any_person_overlaps(product_roi_xyxy) if ignore_overlap else 0.0
            overlapping = overlap_iou >= 0.10
            tm_overlap_iou = overlap_iou
            tm_overlap_flag = overlapping
            if ignore_overlap and overlapping:
                detected = True
                tm_overlap_suppressed = True
            else:
                if now - current_dict_value['last_count_time'] > 5:
                    counter += 1
                    current_dict_value['last_count_time'] = now
                    detected = False
                else:
                    detected = False
    else:
        pix_overlap_suppressed = False
        pix_overlap_iou = 0.0
        pix_overlap_flag = False
        if pixel_detected and not detected:
            detected = True
        if not pixel_detected and detected:
            now = time.time()
            ignore_overlap = True
            bbox_coords = np.array(current_dict_value['bbox_coords'], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(bbox_coords)
            product_roi_xyxy = [x, y, x + w, y + h]
            overlap_iou = any_person_overlaps(product_roi_xyxy) if ignore_overlap else 0.0
            overlapping = overlap_iou >= 0.10
            pix_overlap_iou = overlap_iou
            pix_overlap_flag = overlapping
            if ignore_overlap and overlapping:
                detected = True
                pix_overlap_suppressed = True
            else:
                if now - current_dict_value['last_count_time'] > 3:
                    counter += 1
                    current_dict_value['last_count_time'] = now
                    detected = False
                else:
                    detected = False

    # stdo(1, f"[object_Tracking] cam_ip: {cam_ip}, product_id: {product_id}, count_pixel: {count_pixel}, min_white_pixels: {current_dict_value['min_white_pixels']}, detected: {detected}, template_ratio: {template_result}")
    stop_time = time.time() - start_time
    result_dashboard = "[{}] Counter:{} - Detect:{} - Count Pixel:{} - Template:{:.2f} - T:{:.2f}ms".format(
        product_id, counter, detected, count_pixel, template_result if template_result is not None else -1, stop_time*1000
    )

    # cv2.polylines(display_image, [cv2.convexHull(np.array(current_dict_value['bbox_coords']))], True, (255,0,0), 2)

    # if flag_activate_debug_images:
    #     processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    #     cv2.polylines(processed_image, [cv2.convexHull(np.array(current_dict_value['bbox_coords']))], True, (255,0,0), 2)
    #     image_pack = [processed_image]
    #     title_pack = [str(counter) + "_detected_image_"]
    #     save_image(image_pack, path="/temp_files/object_Tracking/", filename=title_pack, format="png")

    if not current_dict_value.get("use_YOLO", False):
        bbox_coords = np.array(current_dict_value['bbox_coords'], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(bbox_coords)
        draw_Rectangle(
            display_image,
            (x, y), (w, h),
            color=(255, 0, 0), thickness=2
        )


    ip_color = IP_COLORS.get(cam_ip, Fore.WHITE)  
    if current_dict_value.get("use_YOLO", False):
        pred_prob = "-"
        if filtered_results:
            pred_prob = max([float(v['predicted_prob']) for v in filtered_results.values()])
        stdo(
            1,
            f"{Fore.WHITE}[object_Tracking]{Style.RESET_ALL}"
            f"[{ip_color}{cam_ip}{Style.RESET_ALL}][{Fore.YELLOW}{product_id}{Style.RESET_ALL}] "
            f"{Fore.MAGENTA}[YOLO]{Style.RESET_ALL} "
            f"decision:{Fore.GREEN if decision else Fore.RED}{decision}{Style.RESET_ALL}, "
            f"detected:{Fore.GREEN if detected else Fore.RED}{detected}{Style.RESET_ALL}, "
            f"pred_prob:{Fore.CYAN}{pred_prob}{Style.RESET_ALL}, "
            f"counter:{Fore.BLUE}{counter}{Style.RESET_ALL}"
        )
        return counter, detected, result_dashboard, display_image

    if current_dict_value.get("use_template_matching", False):
        tm_log_iou = tm_overlap_iou
        tm_log_overlap = tm_overlap_flag
        stdo(
            1,
            f"{Fore.WHITE}[object_Tracking]{Style.RESET_ALL} "
            f"[{ip_color}{cam_ip}{Style.RESET_ALL}][{Fore.YELLOW}{product_id}{Style.RESET_ALL}] "
            f"template:{Fore.CYAN}{template_result if template_result is not None else '-'}{Style.RESET_ALL}, "
            f"detected:{Fore.GREEN if detected else Fore.RED}{detected}{Style.RESET_ALL}, "
            f"overlap_iou:{Fore.CYAN}{tm_log_iou:.2f}{Style.RESET_ALL}, "
            f"overlap:{Fore.GREEN if tm_log_overlap else Fore.RED}{tm_log_overlap}{Style.RESET_ALL}, "
            f"counter:{Fore.BLUE}{counter}{Style.RESET_ALL}"
        )
    else:
        pix_log_iou = pix_overlap_iou
        pix_log_overlap = pix_overlap_flag
        stdo(
            1,
            f"{Fore.WHITE}[object_Tracking]{Style.RESET_ALL} "
            f"[{ip_color}{cam_ip}{Style.RESET_ALL}][{Fore.YELLOW}{product_id}{Style.RESET_ALL}] "
            f"pixel:{Fore.CYAN}{count_pixel}{Style.RESET_ALL} (>{current_dict_value.get('min_white_pixels', '-')}) "
            f"detected:{Fore.GREEN if detected else Fore.RED}{detected}{Style.RESET_ALL}, "
            f"overlap_iou:{Fore.CYAN}{pix_log_iou:.2f}{Style.RESET_ALL}, "
            f"overlap:{Fore.GREEN if pix_log_overlap else Fore.RED}{pix_log_overlap}{Style.RESET_ALL}, "
            f"counter:{Fore.BLUE}{counter}{Style.RESET_ALL}"
        )

    return counter, detected, result_dashboard, display_image