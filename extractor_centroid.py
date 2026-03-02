import cv2
import numpy as np
import math
import importlib
import libs

import time
import itertools

from stdo import stdo
from tools import filter_Dictionary_Elements
from image_tools import show_image, save_image
from math_tools import coordinate_Scaling
from image_manipulation import remove_Small_Object, contour_Centroids, contour_Extreme_Points, detect_Pallette, sharpening, draw_Rectangle, draw_Text, edge_Detection, average_Color, detect_Color, draw_Circle
import extractor_difference


def preProcess(src_frame, is_label=True, is_bigger_text=False, object_color='white', bbox_invert=True, kernel=20, control_scratch=False, gpu_obj=None, is_inlay=False, counter=0, activate_debug_images=False):

    if object_color == 'white':
        if bbox_invert:
            src_frame = cv2.bitwise_not(src_frame)
        # src_frame = src_frame
    """
    if is_bigger_text:
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    else:
        if type(kernel) is not int:
            kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
        else:
            kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

    if is_label:
        flag_morph = cv2.MORPH_CLOSE
    else:
        flag_morph = cv2.MORPH_OPEN

    if gpu_obj:
        gpu_obj.upload(src_frame)
        if object_color == 'black':
            iterationss = 1
        elif object_color == 'gray' and is_inlay:
            iterationss = 1
        else:
            iterationss = 1 #3
        gpu_obj.morph_Operations(method=flag_morph, kernel=kernel_closing, iterations=iterationss)
        closing = gpu_obj.download()
    else:
        closing = cv2.morphologyEx(src_frame, flag_morph, kernel_closing)
        closing = cv2.morphologyEx(closing, flag_morph, kernel_closing)
        closing = cv2.morphologyEx(closing, flag_morph, kernel_closing)

    if is_bigger_text:
        #show_image([src_frame, closing], open_order=1)

        return -1, -1, closing
    """
    closing = src_frame.copy()
    fill = closing.copy()
    h, w = src_frame.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
    fill_inv = cv2.bitwise_not(fill)

    combine = closing | fill_inv
    #combine = cv2.add(closing, fill_inv)

    if control_scratch:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        combine = cv2.erode(combine, kernel_erode)

    if activate_debug_images:
        image_pack = [src_frame, closing, fill, fill_inv, combine]
        tittle_pack = [str(counter) + "_1_src_frame", str(counter) + "_2_closing", str(counter) + "_3_fill", str(counter) + "_4_fill_inv", str(counter) + "_5_combine"]
        save_image(image_pack, path="temp_files/extractor_centroid/preProcess", filename=tittle_pack, format="png")
        # show_image(image_pack, title='EC', open_order=1)

    return closing, fill_inv, combine


def process_Inlay_Button_Text(
    model_easy_ocr=None,
    model_kmeans=None,
    frame=None,
    probability_threshold=0.5,
    color_table=None,
    counter=0,
    flag_activate_debug_images=False,
):

    start_easy_ocr = time.time()
    prediction_easy_ocr = model_easy_ocr.readtext(frame)
    stop_easy_ocr = time.time()- start_easy_ocr


    # ############## PREDICTION PARSING ############
    start_pp = time.time()

    clustering_prediction_easy_ocr = []
    coords_prediction_easy_ocr = []
    text_prediction_easy_ocr = []
    proba_prediction_easy_ocr = []
    for id, data in enumerate(prediction_easy_ocr):
        clustering_prediction_easy_ocr.append([data[0][0][1]])
        coords_prediction_easy_ocr.append(data[0])
        text_prediction_easy_ocr.append(data[1])
        proba_prediction_easy_ocr.append(data[2])
    clustering_prediction_easy_ocr = np.array(clustering_prediction_easy_ocr)
    text_prediction_easy_ocr = np.array(text_prediction_easy_ocr)
    proba_prediction_easy_ocr = np.array(proba_prediction_easy_ocr)

    stop_pp = time.time() - start_pp
    ###############################################

    start_kmean = time.time()
    model_kmeans.fit(clustering_prediction_easy_ocr)
    labels_kmeans = model_kmeans.labels_
    stop_kmeans = time.time() - start_kmean

    # ######## FILTER PREDICTION RESULTS ###########
    start_fpr = time.time()
    if flag_activate_debug_images:
        temp_draw_image = frame.copy()
        temp_draw_image_filtered = frame.copy()

    dict_cluster = dict()
    for i in range(len(prediction_easy_ocr)):
        dict_cluster[i] = dict()
        dict_cluster[i]["label"] = labels_kmeans[i]
        dict_cluster[i]["probability"] = proba_prediction_easy_ocr[i]
        dict_cluster[i]["coords"] = coords_prediction_easy_ocr[i]
        dict_cluster[i]["text"] = text_prediction_easy_ocr[i]

        if flag_activate_debug_images:
            temp_draw_image = draw_Rectangle(
                temp_draw_image,
                start_point=(dict_cluster[i]["coords"][0][0], dict_cluster[i]["coords"][0][1]),
                end_point=(dict_cluster[i]["coords"][2][0]-dict_cluster[i]["coords"][0][0], dict_cluster[i]["coords"][2][1]-dict_cluster[i]["coords"][0][1]),
                color=(0, 255, 0),
                thickness=3
            )
            temp_draw_image = draw_Text(
                temp_draw_image,
                text=[dict_cluster[i]["text"]],
                center_point=(int(dict_cluster[i]["coords"][0][0]), int(dict_cluster[i]["coords"][0][1]-20)),
                fontscale=2,
                color=(255,0,0),
                thickness=2
            )

    filtered_dict_cluster = filter_Dictionary_Elements(dict_cluster, probability_threshold=probability_threshold)

    dict_text_info = dict()
    for id, data in enumerate(filtered_dict_cluster.values()):
        cropped_image = frame[int(data["coords"][0][1]):int(data["coords"][2][1]), int(data["coords"][0][0]):int(data["coords"][2][0])]
        canny_image = edge_Detection(cropped_image, method='Canny', configs=[100,200])
        contours, _ = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_x = []
        cnt_y = []
        for cnt in contours:
            cnt_x.append(cnt[:,0,0])
            cnt_y.append(cnt[:,0,1])
        flatten_cnt_x = list(itertools.chain(*cnt_x))
        flatten_cnt_y = list(itertools.chain(*cnt_y))

        all_numbers_x = np.arange(1, cropped_image.shape[1]).tolist()
        available_numbers_x = [item for item in all_numbers_x if item not in flatten_cnt_x]
        if not any(available_numbers_x):
            continue
        random_x = np.random.choice(available_numbers_x, 1)

        all_numbers_y = np.arange(1, cropped_image.shape[0]).tolist()
        available_numbers_y = [item for item in all_numbers_y if item not in flatten_cnt_y]
        if not any(available_numbers_y):
            continue
        random_y = np.random.choice(available_numbers_y, 1)

        avg_b, avg_g, avg_r = average_Color(cropped_image, (random_x, random_y), size=4)
        background_color = detect_Color(avg_b, avg_g, avg_r, color_table)

        dict_text_info[id] = dict()
        dict_text_info[id]['text'] = data["text"]
        dict_text_info[id]['background_color'] = background_color
        dict_text_info[id]['coords'] = data["coords"]

        if flag_activate_debug_images:

            temp_draw_image_filtered = draw_Rectangle(
                temp_draw_image_filtered,
                start_point=(data["coords"][0][0], data["coords"][0][1]),
                end_point=(data["coords"][2][0]-data["coords"][0][0], data["coords"][2][1]-data["coords"][0][1]),
                color=(0, 255, 0),
                thickness=3
            )

            temp_draw_image_filtered = draw_Text(
                temp_draw_image_filtered,
                text=[data["text"]],
                center_point=(int(data["coords"][0][0]), int(data["coords"][0][1]-20)),
                fontscale=2,
                color=(255,0,0),
                thickness=2
            )

            random_x, random_y = coordinate_Scaling(
                x=random_x, y=random_y,
                old_w=cropped_image.shape[1], old_h=cropped_image.shape[0],
                new_w=temp_draw_image_filtered.shape[1], new_h=temp_draw_image_filtered.shape[0],
                crop_x=data["coords"][0][0], crop_y=data["coords"][0][1],
                task='CROP',
                is_dual=True
            )

            temp_draw_image = draw_Circle(
                temp_draw_image,
                center_point=(int(random_x), int(random_y)),
                radius=3,
                color=(0, 0, 255),
                thickness=-1
            )

    if flag_activate_debug_images:
        list_temp_save_image = [temp_draw_image, temp_draw_image_filtered]
        filename = [str(counter)+"_1_detected_text", str(counter)+"_2_filtered_text"]
        save_image(list_temp_save_image, path="temp_files/process_Inlay_Button_Text", filename=filename, format="png")

    stop_fpr = time.time() - start_fpr

    stdo(1, "[{}] - process_Inlay_Button_Text - TIME: EOCR:{:.2f} | PP:{:.2f} | KMEANS:{:.2f} | FPR:{:.2f} | Total:{:.2f}".format(
        counter,
        stop_easy_ocr,
        stop_pp,
        stop_kmeans,
        stop_fpr,
        stop_easy_ocr + stop_pp + stop_kmeans + stop_fpr
    ))

    return dict_text_info
    ###############################################

def match_Inlay_Button_Text(
    frame_reference,
    frame_sample,
    ref_button_info={},
    sample_button_info={},
    obj_feature_matching=None,
    algorithm_of_feature_matching=None,
    obj_string_similarity=None,
    algorithm_of_string_similarity=None,
    flag_activate_debug_images=False
):

    dict_matched_frames = dict()
    for id_ref, text_ref in enumerate(ref_button_info['text']):
        for id_sample, text_sample in enumerate(sample_button_info['text']):

            if text_ref == text_sample:
                dict_matched_frames[id_ref] = dict()

                stdo(1, "[{}]-[{}] | Ref:{} | Sample:{}".format(
                    id_ref,
                    id_sample,
                    ref_button_info[id_ref]['background_color'],
                    sample_button_info[id_sample]['background_color']
                ))

                masked_reference_image, homography_sample_image = obj_feature_matching.finder(
                    reference_image = frame_reference[
                        int(ref_button_info[id_ref]['coords'][0][1]):int(ref_button_info[id_ref]['coords'][2][1]),
                        int(ref_button_info[id_ref]['coords'][0][0]):int(ref_button_info[id_ref]['coords'][2][0])
                    ],
                    sample_image = frame_sample[
                        int(sample_button_info[id_sample]['coords'][0][1]):int(sample_button_info[id_sample]['coords'][2][1]),
                        int(sample_button_info[id_sample]['coords'][0][0]):int(sample_button_info[id_sample]['coords'][2][0])
                    ],
                    algorithm=algorithm_of_feature_matching.FLANN_based_SIFT,
                    min_match_count=1,
                    is_mask_reference_image_by_sample_image=True,
                    counter=id_ref,
                    flag_activate_debug_images=flag_activate_debug_images
                )

                dict_matched_frames[id_ref]['ref_symbol'] = masked_reference_image
                dict_matched_frames[id_ref]['sample_symbol'] = homography_sample_image
                break

            else:
                similarity_match_ratio = obj_string_similarity.compute(
                    ref_button_info[id_ref]['text'], sample_button_info[id_sample]['text'], algorithm=algorithm_of_string_similarity.Levenshtein
                )

                if similarity_match_ratio > 0.5:
                    masked_reference_image, homography_sample_image = obj_feature_matching.finder(
                        reference_image = frame_reference[
                            int(ref_button_info[id_ref]['coords'][0][1]):int(ref_button_info[id_ref]['coords'][2][1]),
                            int(ref_button_info[id_ref]['coords'][0][0]):int(ref_button_info[id_ref]['coords'][2][0])
                        ],
                        sample_image = frame_sample[
                            int(sample_button_info[id_sample]['coords'][0][1]):int(sample_button_info[id_sample]['coords'][2][1]),
                            int(sample_button_info[id_sample]['coords'][0][0]):int(sample_button_info[id_sample]['coords'][2][0])
                        ],
                        algorithm=algorithm_of_feature_matching.FLANN_based_SIFT,
                        min_match_count=1,
                        is_mask_reference_image_by_sample_image=True,
                        counter=id_ref,
                        flag_activate_debug_images=flag_activate_debug_images
                    )

                    dict_matched_frames[id_ref]['ref_symbol'] = masked_reference_image
                    dict_matched_frames[id_ref]['sample_symbol'] = homography_sample_image
                    break

        else:
            masked_reference_image, homography_sample_image = obj_feature_matching.finder(
                reference_image = frame_reference[
                    int(ref_button_info[id_ref]['coords'][0][1]):int(ref_button_info[id_ref]['coords'][2][1]),
                    int(ref_button_info[id_ref]['coords'][0][0]):int(ref_button_info[id_ref]['coords'][2][0])
                ],
                sample_image = frame_sample[
                    int(sample_button_info[id_sample]['coords'][0][1]):int(sample_button_info[id_sample]['coords'][2][1]),
                    int(sample_button_info[id_sample]['coords'][0][0]):int(sample_button_info[id_sample]['coords'][2][0])
                ],
                algorithm=algorithm_of_feature_matching.FLANN_based_SIFT,
                min_match_count=1,
                is_mask_reference_image_by_sample_image=True,
                counter=id_ref,
                flag_activate_debug_images=flag_activate_debug_images
            )

            dict_matched_frames[id_ref]['ref_symbol'] = masked_reference_image
            dict_matched_frames[id_ref]['sample_symbol'] = homography_sample_image

    return dict_matched_frames








def in_Of_Inlay_Button_Extract_Symbols(
    frame_reference,
    frame_sample,
    threshold_config=[],
    object_color='white',
    counter=0,
    is_middle_object=False,
    middle_object_roi=[],
    middle_object_roi_2=[],
    kernel=50,
    inlay_button_detect_text_model=None,
    inlay_button_cluster_model=None,
    color_table=None,
    flag_activate_debug_images=False,
    inlay_button_info=[],
    obj_feature_matching=None,
    algorithm_of_feature_matching=None,
    obj_string_similarity=None,
    algorithm_of_string_similarity=None
):

    #### #### #### #### #### ####
    #### # BUTTONS SECTION # ####
    #### #### #### #### #### ####

    sample_dict_inlay_button_info = process_Inlay_Button_Text(
        model_easy_ocr = inlay_button_detect_text_model,
        model_kmeans = inlay_button_cluster_model,
        frame = frame_sample[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]],
        probability_threshold=0.3,
        color_table=color_table,
        counter=counter+1,
        flag_activate_debug_images=flag_activate_debug_images,
    )

    match_Inlay_Button_Text(
        ref_button_info=inlay_button_info,
        sample_button_info=sample_dict_inlay_button_info,
        obj_feature_matching=obj_feature_matching,
        algorithm_of_feature_matching=algorithm_of_feature_matching,
        obj_string_similarity=obj_string_similarity,
        algorithm_of_string_similarity=algorithm_of_string_similarity,
        flag_activate_debug_images=flag_activate_debug_images
    )

    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####



    #### #### #### #### #### ####
    ####  SCREEN SECTION #### ###
    #### #### #### #### #### ####

    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####

def out_Of_Inlay_Extract_Symbols(src_frame, object_color='white', counter=0, kernel=30, threshold=[40, 255, cv2.THRESH_BINARY], enable_rso=True):
    gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)

    _, th = cv2.threshold(gray, threshold[0], threshold[1], eval(threshold[2])) #40, 255, cv2.THRESH_BINARY

    _, th_invert = cv2.threshold(invert, 210-threshold[0], 255, cv2.THRESH_BINARY) #135
    # th = cv2.bitwise_not(th)

    combine = th_invert & th
    # combine = cv2.bitwise_not(combine)

    if object_color == 'gray':
        elim_coord_xmin = 650 #400
        elim_coord_xmax = 1250 #1800
        elim_coord_ymin = 0
        elim_coord_ymax = combine.shape[0]
        elim_coord = [elim_coord_xmin, elim_coord_xmax, elim_coord_ymin, elim_coord_ymax]

        combine_elim = remove_Small_Object(
            combine.copy(),
            ratio= 1,
            aspect = 'lower',
            elim_coord = elim_coord
        )[0]
        # show_image(source=combine_elim, open_order=1, window=True)
        combine = combine_elim

    if type(kernel) is not int:
        kernel_1 = kernel[0]
        kernel_2 = kernel[1]
    else:
        kernel_1 = kernel
        kernel_2 = kernel

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_1, kernel_2))
    closing = cv2.morphologyEx(combine, cv2.MORPH_CLOSE, kernel_closing)

    fill = closing.copy()
    h, w = src_frame.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
    fill_inv = cv2.bitwise_not(fill)
    combine = closing | fill_inv

    if enable_rso:
        rso = remove_Small_Object(combine.copy(), ratio=7)[0]
    else:
        rso = combine

    image_pack = [gray, invert, th, th_invert, combine, closing, combine, rso]
    tittle_pack = [str(counter)+"_1gray", str(counter)+"_2invert", str(counter)+"_3th", str(counter)+"_4thinvert", str(counter)+"_5combine", str(counter)+"_6closing", str(counter)+"_7fill", str(counter)+"_8rso"]
    save_image(image_pack, path="temp_files/extractor_centroid/out_of_Inlay_Extract_symbols", filename=tittle_pack, format="png")
    #show_image(source=image_pack, title=tittle_pack, open_order=2, window=True)

    """22.05.2021
    contours, _ = cv2.findContours(rso, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids_of_symbols = contour_Centroids(contours)

    contours_poly = [None]*len(contours)
    bb_cnt = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bb_cnt[i] = cv2.boundingRect(contours_poly[i])

    display_image_symbols = cv2.cvtColor(rso.copy(), cv2.COLOR_GRAY2BGR)

    for i, value in enumerate(centroids_of_symbols):
        cv2.circle(display_image_symbols, (value[0], value[1]) , 5, (0,255,0), -1)
        cv2.rectangle(display_image_symbols, (int(bb_cnt[i][0]), int(bb_cnt[i][1]), int(bb_cnt[i][2]), int(bb_cnt[i][3])), (24,132,255), 2)
    """

    display_image_symbols, bb_cnt, _, centroids_of_symbols, _ = detect_Centroid_Of_Segmented_Symbols(rso)


    return display_image_symbols, bb_cnt, np.array(centroids_of_symbols)





def detect_Centroid_Of_Segmented_Symbols(src_frame, rgb_frame=None, configuration_id='0', flag_activate_debug_images=False):
    contours, _ = cv2.findContours(src_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    """
    diss = cv2.cvtColor(src_frame, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours):
        for k in c:
            cv2.circle(diss, (int(k[0][0]), int(k[0][1])) , 2, (0,255,0), -1)
    show_image(diss, title='COORDS', open_order=1)
    """
    """
    ### CENTROID ###
    contours_poly = [None]*len(contours)
    bb_cnt = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bb_cnt[i] = cv2.boundingRect(contours_poly[i])
    """
    # centroids_of_symbols = contour_Centroids(contours, get_bbox_centroid=False)
    # centroids_of_symbols = contour_Centroids(bb_cnt, get_bbox_centroid=True)

    """
    ### EXTREME ###
    coords_ext = list()
    bb_ext = list()
    for i in range(len(contours)):
        temp = contour_Extreme_Points(contours[i], get_centroid=True, is_width_height=True)
        bb_ext.append(tuple(temp[:len(temp)-1]))
        coords_ext.append(temp[-1])
    coords_ext = np.array(coords_ext)

    # import pdb; pdb.set_trace()


    ### NORMALIZATION EXTREME TO CENTROID ###
    norm_bb_cnt = []
    for i in range(len(bb_cnt)):
        norm_bb_cnt.append(list())

        if bb_cnt[i][0] < bb_ext[i][0]:
            norm_bb_cnt[-1].append(bb_cnt[i][0])
        else:
            norm_bb_cnt[-1].append(bb_ext[i][0])

        if bb_cnt[i][1] < bb_ext[i][1]:
            norm_bb_cnt[-1].append(bb_cnt[i][1])
        else:
            norm_bb_cnt[-1].append(bb_ext[i][1])

        if bb_cnt[i][2] > bb_ext[i][2]:
            norm_bb_cnt[-1].append(bb_cnt[i][2])
        else:
            norm_bb_cnt[-1].append(bb_ext[i][2])

        if bb_cnt[i][3] > bb_ext[i][3]:
            norm_bb_cnt[-1].append(bb_cnt[i][3])
        else:
            norm_bb_cnt[-1].append(bb_ext[i][3])

        """"""
        norm_bb_cnt[-1].append(bb_cnt[i][0])
        norm_bb_cnt[-1].append(bb_ext[i][1])
        norm_bb_cnt[-1].append(bb_cnt[i][2])
        norm_bb_cnt[-1].append(bb_ext[i][3])
        """"""

    bb_cnt = norm_bb_cnt
    """

    ###DISPLAY###
    display_image_symbols = cv2.cvtColor(src_frame.copy(), cv2.COLOR_GRAY2BGR)
    coords_ext = list()
    bb_ext = list()
    bb_cnt = list()
    centroids_of_symbols = list()

    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    for i, value in enumerate(sorted_contours):
        sec_bbox = cv2.boundingRect(sorted_contours[i])
        bb_cnt.append(sec_bbox)
        # stdo(1, "[{}] SEC BBOX: {}".format(configuration_id, sec_bbox))
        contour_centroids = contour_Centroids(sec_bbox, get_bbox_centroid=True, is_single=True)[0]
        centroids_of_symbols.append(contour_centroids)

        if flag_activate_debug_images:
            cv2.circle(display_image_symbols, (contour_centroids[0], contour_centroids[1]) , 1, (0,255,0), -1)
            draw_Rectangle(display_image_symbols, start_point=(sec_bbox[0], sec_bbox[1]), end_point=(sec_bbox[2], sec_bbox[3]), color=(24,132,255), thickness=1)
            #draw_Text(display_image_symbols, text=[str(i)], center_point=(int(bb_cnt[i][0]), int(bb_cnt[i][1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)
            draw_Text(display_image_symbols, text=[str(i)], center_point=(sec_bbox[0], sec_bbox[1]+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)

            cv2.circle(rgb_frame, (contour_centroids[0], contour_centroids[1]) , 1, (0,255,0), -1)
            draw_Rectangle(rgb_frame, start_point=(sec_bbox[0], sec_bbox[1]), end_point=(sec_bbox[2], sec_bbox[3]), color=(24,132,255), thickness=1)
            #draw_Text(rgb_frame, text=[str(i)], center_point=(int(bb_cnt[i][0]), int(bb_cnt[i][1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)
            draw_Text(rgb_frame, text=[str(i)], center_point=(sec_bbox[0], sec_bbox[1]+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)

            # cv2.circle(display_image_symbols, (int(coords_ext[i][0]), int(coords_ext[i][1])) , 5, (0,0,255), -1)
            # cv2.rectangle(display_image_symbols, (int(bb_ext[i][0]), int(bb_ext[i][1]), int(bb_ext[i][2]), int(bb_ext[i][3])), (255,132,24), 1)

        # stdo(1, "[{}][{}] detect_Centroid_Of_Segmented_Symbols: center:({},{}) | bbox:({},{},{},{})".format(
        #     i, configuration_id,
        #     contour_centroids[0], contour_centroids[1],
        #     sec_bbox[0], sec_bbox[1], sec_bbox[2], sec_bbox[3]
        # ))

    if flag_activate_debug_images:
        image_pack = [display_image_symbols, rgb_frame]
        title_pack = [
            str(configuration_id) + "_display_image_symbols",
            str(configuration_id) + "_rgb_frame"
        ]
        save_image(image_pack, path="temp_files/detect_Centroid_Of_Segmented_Symbols/", filename=title_pack, format="png")

    return display_image_symbols, bb_cnt, bb_ext, centroids_of_symbols, coords_ext, rgb_frame


def in_of_Inlay_Extract_Symbols_old(
    src_frame,
    threshold_config=[],
    object_color='white',
    counter=0,
    is_middle_object=False,
    middle_object_roi=[],
    middle_object_roi_2=[],
    kernel=50
):

    gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    if object_color == 'white':
        # k_th = 40 # BETTER-ANKASTRE
        k_th = 30 # CAMASIR
    elif (object_color == 'gray') or (object_color == 'black'):
        #k_th = 50 # eski panolar
        k_th = 40 #60 09.07.2021 eski gray panolar
    elif object_color == 'piano-black':
        k_th = 60

    _, pre_threshold = cv2.threshold(gray, k_th, 255, cv2.THRESH_BINARY) #40,255,cv2.THRESH_BINARY#
    inner_mask = pre_threshold.copy()
    not_inner_mask = cv2.bitwise_not(inner_mask)

    if middle_object_roi_2:
        if (object_color == 'white') or (object_color == 'gray'):
            inner_mask[middle_object_roi_2[0]:middle_object_roi_2[1], middle_object_roi_2[2]:middle_object_roi_2[3]] = 255
            not_inner_mask = cv2.bitwise_not(inner_mask)
        else:
            inner_mask[middle_object_roi_2[0]:middle_object_roi_2[1], middle_object_roi_2[2]:middle_object_roi_2[3]] = 0
            not_inner_mask = inner_mask.copy()


    if is_middle_object:
        startx = middle_object_roi[2]
        endx = middle_object_roi[3]
        starty = middle_object_roi[0]
        endy = middle_object_roi[1]

    if (object_color == 'white') or (object_color == 'gray'):

        fill = not_inner_mask.copy()
        h, w = src_frame.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
        inlay_symbols = cv2.bitwise_not(fill)


        ####################FOR MASKING INLAY ZONE######################
        combine = not_inner_mask | inlay_symbols
        rso_inlay_zone = remove_Small_Object(combine.copy(), ratio=20000)[0]
        contours, _ = cv2.findContours(rso_inlay_zone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        """
        #rso_inlay_zone = cv2.bitwise_not(rso_inlay_zone)
        contours_poly = [None]*len(contours)
        bound_rect = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            bound_rect[i] = cv2.boundingRect(contours_poly[i])

        ratio = 10
        startx = int(bound_rect[i][0]) - ratio
        starty = int(bound_rect[i][1]) - ratio
        endx = int(bound_rect[i][0]+bound_rect[i][2]) + ratio
        endy = int(bound_rect[i][1]+bound_rect[i][3]) + ratio
        rso_inlay_zone[starty:endy, startx:endx] = 255
        rso_inlay_zone = cv2.bitwise_not(rso_inlay_zone)
        """

        """ratio = 30
        startx = 150 - ratio
        endx = 1750 + (ratio*2)
        starty = 130 - ratio
        endy = 720"""

        """if startx < 0:
            startx = 0
        if endx > rso_inlay_zone.shape[1]:
            endx = rso_inlay_zone.shape[1] - 1
        if starty < 0:
            starty = 0
        if endy > rso_inlay_zone.shape[0]:
            endy = rso_inlay_zone.shape[0] - 1"""

        rso_inlay_zone[starty:endy, startx:endx] = 255
        rso_inlay_zone = cv2.bitwise_not(rso_inlay_zone)


        # show_image([gray, inner_mask, not_inner_mask, inlay_symbols, combine, rso_inlay_zone], open_order=1, window=True)
        ################################################################

    else:
        inlay_symbols = not_inner_mask.copy()

        rso_inlay_zone = pre_threshold.copy()
        rso_inlay_zone[starty:endy, startx:endx] = 0

        contours, _ = cv2.findContours(rso_inlay_zone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if type(kernel) is not int:
        kernel_1 = kernel[0]
        kernel_2 = kernel[1]
    else:
        kernel_1 = kernel
        kernel_2 = kernel

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_1, kernel_2)) # 50,50
    closing = cv2.morphologyEx(inlay_symbols, cv2.MORPH_CLOSE, kernel_closing)
    #closing = cv2.dilate(inlay_symbols, kernel_closing)
    rso = remove_Small_Object(closing.copy(), is_chosen_max_area=False, ratio=300)[0] # 500

    """
    fill = rso.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fill, mask, (0,0), 255)
    fill_inv = cv2.bitwise_not(fill)
    combine = closing | fill_inv
    """

    """22.05.2021 Öncesi
    contours, _ = cv2.findContours(rso, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #display_image_symbols = src_frame.copy()
    display_image_symbols = cv2.cvtColor(rso.copy(), cv2.COLOR_GRAY2BGR)

    contours_poly = [None]*len(contours)
    bound_rect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    centroids_of_symbols = contour_Centroids(contours, get_bbox_centroid=True)


    for i, value in enumerate(centroids_of_symbols):
        cv2.circle(display_image_symbols, (value[0], value[1]) , 10, (0,255,0), -1)
        cv2.rectangle(display_image_symbols, (int(bound_rect[i][0]), int(bound_rect[i][1]), int(bound_rect[i][2]), int(bound_rect[i][3])), (24,132,255), 2)
    """

    display_image_symbols, bound_rect, _, centroids_of_symbols, _ = detect_Centroid_Of_Segmented_Symbols(rso)


    #display_image_symbols = cv2.cvtColor(display_image_symbols, cv2.COLOR_BGR2RGB)

    #cv2.imwrite(absolute_path + "../../test_area/test_image_results/rso.png", display_image_symbols)

    #show_image([gray, inner_mask, not_inner_mask, inlay_symbols, closing, rso], open_order=1, window=True)


    list_temp_save_image = [gray, pre_threshold, inner_mask, not_inner_mask, inlay_symbols, closing, rso, display_image_symbols]
    filename = [str(counter)+"_1gray", str(counter)+"_2pre_threshold", str(counter)+"_3inner_mask", str(counter)+"_4not_inner_mask", str(counter)+"_5inlay_symbols", str(counter)+"_6closing", str(counter)+"_7rso", str(counter)+"_8display"]
    save_image(list_temp_save_image, path="temp_files/extractor_centroid/in_Of_Inlay_Button_Extract_Symbols/", filename=filename, format="png")

    return np.array(centroids_of_symbols), rso_inlay_zone, contours, bound_rect, rso, display_image_symbols
    #return rso, np.array(centroids_of_symbols), bound_rect, rso_inlay_zone





def inlay_Process_Sequence(
    frame_reference=None,
    frame_sample=None,
    object_color='white',
    pano_is_middle_mask=False,
    pano_is_middle_roi=[],
    pano_is_middle_roi_2=[],
    pano_kernel=20,
    pano_coord_thresh=15,
    pano_crop_ratio=0,
    counter_data_collector_pano_index=0,
    object_id=0,
    pano_sector_index=0,
    pano_sobel_scale=0,
    pano_sharp_scale=0,
    pano_rso_ratio=0,
    flag_activate_debug_images=False,
    shape_list=[],
    pano_threshold_config=[225, 255, eval("cv2.THRESH_BINARY")],
    crop_param_secondary=[],
    obj_feature_matching=None,
    algorithm_of_feature_matching=None,
    rpw_ratio=2,
    inlay_button_detect_text_model=None,
    inlay_button_cluster_model=None,
    color_table=None,
    inlay_button_info=[],
    obj_string_similarity=None,
    algorithm_of_string_similarity=None
):

    """
    ref_coords, ref_inlay_zone_mask, ref_contours, ref_bbox, ref_rso, display_in_inlay_symbols_r = in_Of_Inlay_Button_Extract_Symbols(
        frame_reference,
        object_color=object_color,
        counter=0,
        is_middle_object=pano_is_middle_mask,
        middle_object_roi=pano_is_middle_roi,
        middle_object_roi_2=pano_is_middle_roi_2,
        kernel=pano_kernel,
        inlay_button_detect_text_model=inlay_button_detect_text_model,
        inlay_button_cluster_model=inlay_button_cluster_model,
        color_table=color_table,
        flag_activate_debug_images=flag_activate_debug_images,
        inlay_button_info=inlay_button_info,
        obj_feature_matching=obj_feature_matching,
        algorithm_of_feature_matching=algorithm_of_feature_matching,
        obj_string_similarity=obj_string_similarity,
        algorithm_of_string_similarity=algorithm_of_string_similarity
    )
    sample_coords, sample_inlay_zone_mask, sample_contours, sample_bbox, sample_rso, display_in_inlay_symbols_s = in_Of_Inlay_Button_Extract_Symbols(
        frame_sample,
        object_color=object_color,
        counter=1,
        is_middle_object=pano_is_middle_mask,
        middle_object_roi=pano_is_middle_roi,
        middle_object_roi_2=pano_is_middle_roi_2,
        kernel=pano_kernel,
        inlay_button_detect_text_model=inlay_button_detect_text_model,
        inlay_button_cluster_model=inlay_button_cluster_model,
        color_table=color_table,
        flag_activate_debug_images=flag_activate_debug_images,
        inlay_button_info=inlay_button_info,
        obj_feature_matching=obj_feature_matching,
        algorithm_of_feature_matching=algorithm_of_feature_matching,
        obj_string_similarity=obj_string_similarity,
        algorithm_of_string_similarity=algorithm_of_string_similarity
    )
    """


    #### EXTRACT SAMPLE IMAGE ###
    #### ## BUTTONS SYMBOLS  ####
    #### #### #### #### #### ####

    ref_coords = inlay_button_info

    sample_coords, sample_inlay_zone_mask, sample_contours, sample_bbox, sample_rso, display_in_inlay_symbols_s = in_Of_Inlay_Button_Extract_Symbols(
        frame_reference,
        frame_sample,
        object_color=object_color,
        counter=1,
        is_middle_object=pano_is_middle_mask,
        middle_object_roi=pano_is_middle_roi,
        middle_object_roi_2=pano_is_middle_roi_2,
        kernel=pano_kernel,
        inlay_button_detect_text_model=inlay_button_detect_text_model,
        inlay_button_cluster_model=inlay_button_cluster_model,
        color_table=color_table,
        flag_activate_debug_images=flag_activate_debug_images,
        inlay_button_info=inlay_button_info,
        obj_feature_matching=obj_feature_matching,
        algorithm_of_feature_matching=algorithm_of_feature_matching,
        obj_string_similarity=obj_string_similarity,
        algorithm_of_string_similarity=algorithm_of_string_similarity
    )

    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####



    primary_crop_w, primary_crop_h, secondary_crop_w, secondary_crop_h = shape_list


    #### #### #### #### #### ####
    #### ####  IN-INLAY #### ####
    #### #### #### #### #### ####

    list_cropped_symbols, frame_not_found = obj_feature_matching.finder(
        frame_reference,
        frame_sample,
        algorithm = algorithm_of_feature_matching.CWCFT,
        coord_thresh = pano_coord_thresh,
        ref_coords = ref_coords,
        sample_coords = sample_coords,
        ref_bb = ref_bbox,
        sample_bb = sample_bbox,
        is_inlay_activated = False,
        crop_ratio = pano_crop_ratio
    )

    list_counters_diffs_of_symbols = []
    counter=0

    temp_buffer_data_collector = []
    counter_data_collector_symbol_index = 0

    for i in range(len(list_cropped_symbols)):

        ############### FOR-DATA-COLLECTOR ############
        temp_buffer_data_collector = [
            str(counter_data_collector_pano_index) + "_" + str(counter_data_collector_symbol_index),
            object_id,
            pano_sector_index,
            object_color,
            list_cropped_symbols[i]['data_collector'],
            "OK"
        ]
        ###############################################

        counters_difference_symbols = extractor_difference.extractor_difference(
            list_cropped_symbols[i]['ref_frame'].copy(),
            list_cropped_symbols[i]['sample_frame'].copy(),
            method=2,  #method=2 is the inlay process on post-subtraction#
            sobel_scale=pano_sobel_scale,
            sharp_scale=pano_sharp_scale,
            draw_diff=False,
            draw_diff_circle=False,
            buffer_percentage=90,
            filter_lower_ratio=10,
            filter_upper_ratio=15,
            rso_ratio=pano_rso_ratio,
            window=False,
            counter=counter,
            activate_debug_images=flag_activate_debug_images,
            kernel = pano_kernel,
            dilate_iteration=2,
            erode_iteration=1,
            pixel_width=rpw_ratio
        )[4]
        counter += 1
        counter_data_collector_symbol_index += 1

        #list_temp_save_image = [list_cropped_symbols[i]['ref_frame'], list_cropped_symbols[i]['sample_frame']]
        #filename = [str(i)+"_1ref", str(i)+"_2sample"]
        #save_image(list_temp_save_image, path="temp_files/symbols", filename=filename, format="png")

        if counters_difference_symbols:

            ############### FOR-DATA-COLLECTOR ############
            temp_buffer_data_collector[5] = "NOK"
            ###############################################

            for cent in counters_difference_symbols:
                x, y = coordinate_Scaling(
                    cent[0], cent[1],
                    list_cropped_symbols[i]['sample_frame'].shape[1],
                    list_cropped_symbols[i]['sample_frame'].shape[0],
                    frame_sample.shape[1],
                    frame_sample.shape[0],
                    crop_x=list_cropped_symbols[i]['crop'][0],
                    crop_y=list_cropped_symbols[i]['crop'][1],
                    task='CROP'
                )

                list_counters_diffs_of_symbols.append((x,y))

        ############### FOR-DATA-COLLECTOR ############
        # if flag_test_mode is False:
        #    self.add_buffer_dataset_collector(temp_buffer_data_collector)
        ###############################################

    list_counters_diffs_of_symbols = np.array(list_counters_diffs_of_symbols)

    list_counters_diffs_of_symbols = list_counters_diffs_of_symbols.reshape(-1,2)
    list_counters_diffs_of_symbols[:,0], list_counters_diffs_of_symbols[:,1] = coordinate_Scaling(
        x=list_counters_diffs_of_symbols[:,0], y=list_counters_diffs_of_symbols[:,1],
        old_w=secondary_crop_w, old_h=secondary_crop_h,
        new_w=primary_crop_w, new_h=primary_crop_h,
        crop_x=crop_param_secondary[2], crop_y=crop_param_secondary[0],
        task='CROP',
        is_dual=False
    )

    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####