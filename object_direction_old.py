# import os
import time

import cv2
# from skimage.metrics import structural_similarity
# import matplotlib.pyplot as plt
import numpy as np

import libs
from image_manipulation import grayscale_Conversion  # , threshold, remove_Small_Object, circular_Hough_Transform, calculation_Crop_Parameters_Of_Image_To_Focused_Interest_Area, image_Make_Border, image_Calculate_Border_Range, draw_Circle
from image_tools import save_image  # , show_image, open_image
# from stdo import stdo, get_time
# from tools import time_log, list_files, list_folders, save_to_json, load_from_json, path_control, seppuku, remove_dir
# from math_tools import coordinate_Scaling, extraction_Pixel_Values, find_Peak_Points, line_Intersection
# from preprocess_image_processing import preprocess_of_Component_Direction


def detect_Object_Direction(
        list_image,
        direction='up',
        method='C',
        process_methods=None,
        process_params=None,
        configuration_id='N/A',
        pattern_id='0',
        counter=0,
        flag_activate_debug_images=False
    ):

    dict_direction = {
        "0": 'left',
        "1": 'up',
        "2": 'down',
        "3": 'right'
    }

    dict_direction_binary = {
        "0": 'left',
        "1": 'right'
    }

    start_time = time.time()
    pred_direction_decision = 'up'
    decision = False
    result_dashboard = "Direct:{} - Pred:[N/A] - T:{} ms".format(decision, 0)

    # method =>  C:capacitor | K:socket | D:diot#

    if method == 'C' or method == 'c':

        """
        max_matched_frame_gray_masked, max_matched_frame_sobel_masked, max_matched_frame_equhist_masked = preprocess_of_Component_Direction(
            image=list_image[1],
            method=method,
            configuration_id=configuration_id,
            counter=counter,
            flag_activate_debug_images=flag_activate_debug_images
        )


        ######
        temp_indices = np.where(max_matched_frame_equhist_masked == 0)
        new_image = max_matched_frame_equhist_masked.copy()
        new_image[temp_indices] = max_matched_frame_equhist_masked[temp_indices] + 255

        center = (int(new_image.shape[1]/2), int(new_image.shape[0]/2))
        radius = min(center[0], center[1], new_image.shape[1]-center[0], new_image.shape[0]-center[1])
        outer_mask_white = draw_Circle(new_image.copy(), center_point=(center[0], center[1]), radius=radius-2, color=(0,0,0), thickness=1)
        ######


        # The Signal Line: TopLeft -> BottomRight
        signal_max_matched_frame_equhist = extraction_Pixel_Values(
            image=outer_mask_white,
            start_point=(0,0),
            end_point=(max_matched_frame_equhist_masked.shape[1]-1, max_matched_frame_equhist_masked.shape[0]-1)
        )

        signal_max_matched_frame_equhist_norm = np.concatenate(
            (signal_max_matched_frame_equhist[:,0].reshape(-1,1), 255-signal_max_matched_frame_equhist[:,1].reshape(-1,1)),
            axis=1
        )
        peak_max_1, peak_distances_1, peak_indices_1 = find_Peak_Points(
            image=max_matched_frame_equhist_masked,
            signal=signal_max_matched_frame_equhist_norm,
            is_scale_x_axis=False,
            is_scale_y_axis=False
        )

        # The Signal Line: BottomLeft -> TopRight
        signal_max_matched_frame_equhist = extraction_Pixel_Values(
            image=outer_mask_white,
            start_point=(0,max_matched_frame_equhist_masked.shape[0]-1),
            end_point=(max_matched_frame_equhist_masked.shape[1]-1, 0)
        )

        signal_max_matched_frame_equhist_norm = np.concatenate(
            (signal_max_matched_frame_equhist[:,0].reshape(-1,1), 255-signal_max_matched_frame_equhist[:,1].reshape(-1,1)),
            axis=1
        )
        peak_max_2, peak_distances_2, peak_indices_2 = find_Peak_Points(
            image=max_matched_frame_equhist_masked,
            signal=signal_max_matched_frame_equhist_norm,
            is_scale_x_axis=False,
            is_scale_y_axis=False
        )

        """"""
        signal1_start_curve = peak_indices_1[np.where(peak_distances_1 == peak_max_1)[0][0]]
        signal1_stop_curve = peak_indices_1[np.where(peak_distances_1 == peak_max_1)[0][0] + 1]
        signal2_start_curve = peak_indices_2[np.where(peak_distances_2 == peak_max_2)[0][0]]
        signal2_stop_curve = peak_indices_2[np.where(peak_distances_2 == peak_max_2)[0][0] + 1]
        max_radius = min(signal1_stop_curve, signal2_stop_curve) - max(signal1_start_curve, signal2_start_curve)
        """"""

        max_radius = max(peak_max_1, peak_max_2)
        range_norm, _ = coordinate_Scaling(
            x=max_radius,
            y=0,
            old_w=max_matched_frame_equhist_masked.shape[1],
            old_h=max_matched_frame_equhist_masked.shape[0],
            task="PROJECTION_TO_XY_AXIS",
            is_dual=True
        )

        x,y,r, detected_image = circular_Hough_Transform(
            image=max_matched_frame_sobel_masked.copy(),
            max_radius=range_norm,  # np.max([peak_max_1, peak_max_2]),
            is_circle_detected_image=flag_activate_debug_images
        )
        if (
                (x >= max_matched_frame_sobel_masked.shape[1]) or (y >= max_matched_frame_sobel_masked.shape[0])
            ) or (
                (r >= max_matched_frame_sobel_masked.shape[0] // 2) or (r >= max_matched_frame_sobel_masked.shape[1] // 2)
            ) or (
                (r <= 10)
            ) or (
                (x - r <= 0) or (x + r > max_matched_frame_sobel_masked.shape[1]) or (y - r <= 0) or (y + r > max_matched_frame_sobel_masked.shape[0])
            ):
            sum_of_all_contour_area = -1
            decision = False

        else:
            max_matched_frame_gray_nested_masked = draw_Circle(max_matched_frame_gray_masked.copy(), center_point=(x, y), radius=r+3, color=(0,0,0), thickness=-1)

            # First progress as determined "direction"
            crop_params = calculation_Crop_Parameters_Of_Image_To_Focused_Interest_Area(image=max_matched_frame_gray_nested_masked, inner_circle=(x,y,r), direction=direction)
            max_matched_frame_gray_nested_masked_crop = max_matched_frame_gray_nested_masked[
                crop_params['start_y']:crop_params['end_y'],
                crop_params['start_x']:crop_params['end_x']
            ]

            max_matched_frame_gray_nested_masked_crop_threshold = threshold(max_matched_frame_gray_nested_masked_crop.copy(), configs=[150, 255, cv2.THRESH_BINARY])
            if max_matched_frame_gray_nested_masked_crop_threshold is None:
                return -1, False

            _, _, all_contour_area, _, _ = remove_Small_Object(
                max_matched_frame_gray_nested_masked_crop_threshold.copy(),
                ratio=1,
            )
            sum_of_all_contour_area = np.sum(all_contour_area)
            if sum_of_all_contour_area > 50:
                decision = True
            else:
                decision = False

            # Second progress to decide as determined "direction" with opposite one.
            if direction == 'up':
                opposite_direction = 'down'
            elif direction == 'down':
                opposite_direction = 'up'
            elif direction == 'left':
                opposite_direction = 'right'
            elif direction == 'right':
                opposite_direction = 'left'

            crop_params = calculation_Crop_Parameters_Of_Image_To_Focused_Interest_Area(image=max_matched_frame_gray_nested_masked, inner_circle=(x,y,r), direction=opposite_direction)
            max_matched_frame_gray_nested_masked_crop = max_matched_frame_gray_nested_masked[
                crop_params['start_y']:crop_params['end_y'],
                crop_params['start_x']:crop_params['end_x']
            ]

            max_matched_frame_gray_nested_masked_crop_threshold = threshold(max_matched_frame_gray_nested_masked_crop.copy(), configs=[150, 255, cv2.THRESH_BINARY])
            if max_matched_frame_gray_nested_masked_crop_threshold is None:
                return -1, False

            _, _, all_contour_area, _, _ = remove_Small_Object(
                max_matched_frame_gray_nested_masked_crop_threshold.copy(),
                ratio=1,
            )
            sum_of_all_contour_area_opposite_direction = np.sum(all_contour_area)

            coefficient_direction_decision = (sum_of_all_contour_area + sum_of_all_contour_area_opposite_direction) / 2
            decision = False
            if sum_of_all_contour_area > coefficient_direction_decision:
                decision = True
            else:
                decision = False

        if flag_activate_debug_images:
            stop_time = (time.time() - start_time) * 1000
            stdo(1, "[{}][{}]: Direction:{} | Direction-Area:{:.2f} | Time:{:.2f} ms".format(pattern_id, configuration_id, decision, sum_of_all_contour_area, stop_time))

            image_pack = [detected_image]
            title_pack = [
                str(counter) + "_0_detected_image_" + configuration_id
            ]
            save_image(image_pack, path="temp_files/detect_Object_Direction/" + method, filename=title_pack, format="png")

            # Figure #
            """"""
            fig, ax = plt.subplots(3, 5, figsize=(14, 8))
            ax[0,0].imshow(list_image[1], cmap='gray')
            ax[0,1].imshow(max_matched_frame_sobel_masked, cmap='gray')
            ax[0,2].imshow(max_matched_frame_equhist_masked, cmap='gray')

            ax[1,2].plot(signal_max_matched_frame_equhist_norm[:,0], signal_max_matched_frame_equhist_norm[:,1])

            ax[2,2].imshow(max_matched_frame_gray_nested_masked, cmap='gray')
            ax[2,3].imshow(max_matched_frame_gray_nested_masked_crop, cmap='gray')
            ax[2,4].imshow(max_matched_frame_gray_nested_masked_crop_threshold, cmap='gray')
            ax[2,4].set_title("[{}] all_contour_area: {}".format(counter, sum_of_all_contour_area), fontsize=8)

            if not path_control(path='temp_files/detect_Object_Direction', is_file=False, is_directory=True):
                os.makedirs('temp_files/detect_Object_Direction', exist_ok=True, mode=0o777)
            plt.savefig("temp_files/detect_Object_Direction/detect_Object_Direction.png")
            """"""

            """

        max_w = process_params[0]
        max_h = process_params[1]

        padded_image = cv2.resize(list_image[1], (max_w, max_h))
        padded_image = grayscale_Conversion(padded_image)
        # padded_image = padded_image / 255
        # prediction = process_methods[0].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
        prediction = process_methods[0].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
        prediction = np.array(prediction)
        index = np.where(prediction > 0.5)[0]

        if len(index) > 0:
            pred_direction_decision = dict_direction[str(index[0])]
        else:
            pred_direction_decision = dict_direction[str(np.where(prediction == prediction.max())[0][0])]

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

            # stdo(1, "[{}][{}]: Direction:{} | Prediction:[{:.1f} {:.1f} {:.1f} {:.1f}] | Time:{:.2f} ms".format(pattern_id, configuration_id, decision, prediction[0], prediction[1], prediction[2], prediction[3], stop_time))

        stop_time = (time.time() - start_time) * 1000
        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f} ms".format(decision, prediction[0], prediction[1], prediction[2], prediction[3], stop_time)
        # result_dashboard = "Direct:{} - Pred:[{}] - T:{:.2f} ms".format(decision, str(*np.round(prediction, 1)), stop_time)

    elif method == 'K' or method == 'k':

        """
        _, _, ref_image_threshold = preprocess_of_Component_Direction(
            image=list_image[0],
            method=method,
            counter=counter,
            flag_activate_debug_images=flag_activate_debug_images
        )

        _, _, sample_image_threshold = preprocess_of_Component_Direction(
            image=list_image[1],
            method=method,
            counter=counter+20,
            flag_activate_debug_images=flag_activate_debug_images
        )

        """"""
        score, _ = structural_similarity(ref_image_threshold, sample_image_threshold, full=True)
        sum_of_all_contour_area = float("{:.2f}".format(score))

        decision = False
        if sum_of_all_contour_area > 0.7:
            decision = True
        else:
            decision = False
        """"""

        _, _, good, list_kp = process_methods.finder(
            reference_image=ref_image_threshold.copy(),
            sample_image=sample_image_threshold.copy(),
            algorithm=process_params,
            min_match_count=5,
            configuration_id=configuration_id,
            pattern_id=pattern_id,
            counter=counter
        )
        if good == -1:
            sum_of_all_contour_area = -1
            decision = False

        else:
            connection_coordinates = list()
            for match in good:
                idx1 = match.queryIdx
                idx2 = match.trainIdx

                x1, y1 = list_kp[0][idx1].pt
                x2, y2 = list_kp[1][idx2].pt

                connection_coordinates.append(( (int(x1), int(y1)), (int(x2), int(y2)) ))
            connection_coordinates = np.array(connection_coordinates)

            connection_coordinates_norm = list()
            for points in connection_coordinates:
                connection_coordinates_norm.append(( (points[0][0], points[0][1]), (points[1][0]+points[0][0], points[1][1]) ))
            connection_coordinates_norm = np.array(connection_coordinates_norm)

            list_decision_line_intersection = list()
            for id, _ in enumerate(connection_coordinates_norm):
                if id < len(connection_coordinates_norm)-1:

                    cross_x, cross_y = line_Intersection(connection_coordinates_norm[id], connection_coordinates_norm[id+1], method='1')
                    if (
                        (cross_x > sample_image_threshold.shape[1]) or (cross_y > sample_image_threshold.shape[0])
                    ) or (
                        (cross_x < 0) or (cross_y < 0)
                    ):
                        cross_x = False; cross_y = False
                    list_decision_line_intersection.append((cross_x, cross_y))

            list_decision_line_intersection = np.array(list_decision_line_intersection)
            count_False = np.count_nonzero(list_decision_line_intersection != False)
            count_True = np.count_nonzero(list_decision_line_intersection == False)
            if count_True > count_False:
                decision = True
            else:
                decision = False
            sum_of_all_contour_area = count_False
        """

        max_w = process_params[2]
        max_h = process_params[3]

        padded_image = cv2.resize(list_image[1], (max_w, max_h))
        padded_image = grayscale_Conversion(padded_image)
        # padded_image = padded_image / 255
        # prediction = process_methods[1].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
        prediction = process_methods[1].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
        prediction = np.array(prediction)
        index = np.where(prediction > 0.5)[0]

        if len(index) > 0:
            pred_direction_decision = dict_direction[str(index[0])]
        else:
            pred_direction_decision = dict_direction[str(np.where(prediction == prediction.max())[0][0])]

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
        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f}ms".format(decision, prediction[0], prediction[1], prediction[2], prediction[3], stop_time)
        # result_dashboard = "Direct:{} - Pred:[{}] - T:{:.2f} ms".format(decision, str(*np.round(prediction, 1)), stop_time)

    elif (method == 'D' or method == 'd') or (method == 'Z' or method == 'z'):

        max_w = process_params[4]
        max_h = process_params[5]

        padded_image = cv2.resize(list_image[1], (max_w, max_h))
        padded_image = grayscale_Conversion(padded_image)
        # padded_image = padded_image / 255
        # prediction = process_methods[1].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
        prediction = process_methods[2].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
        prediction = np.array(prediction)
        index = np.where(prediction > 0.5)[0]

        if len(index) > 0:
            pred_direction_decision = dict_direction[str(index[0])]
        else:
            pred_direction_decision = dict_direction[str(np.where(prediction == prediction.max())[0][0])]

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

        result_dashboard = "Direct:{} - Pred:[{:.1f} {:.1f} {:.1f} {:.1f}] - T:{:.2f}ms".format(decision, prediction[0], prediction[1], prediction[2], prediction[3], stop_time)
        # result_dashboard = "Direct:{} - Pred:[{}] - T:{:.2f} ms".format(decision, str(*np.round(prediction, 1)), stop_time)

    elif (method == 'S' or method == 's') or (method == 'S' or method == 's'):

        max_w = process_params[6]
        max_h = process_params[7]

        padded_image = cv2.resize(list_image[1], (max_w, max_h))
        padded_image = grayscale_Conversion(padded_image)
        # padded_image = padded_image / 255
        # prediction = process_methods[1].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1).astype(np.float32))[0]
        prediction = process_methods[3].predict(padded_image.reshape(-1, padded_image.shape[0], padded_image.shape[1], 1))[0]
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

    return pred_direction_decision, decision, result_dashboard