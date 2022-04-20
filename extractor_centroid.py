#cython: language_level=3, boundscheck=False
import cv2
import numpy as np
import math
import importlib
import libs

import time

from image_tools import show_image, save_image
from image_manipulation import remove_Small_Object, contour_Centroids, contour_Extreme_Points, detect_Pallette, sharpening


def preProcess(src_frame, is_label=True, is_bigger_text=False, object_color='white', bbox_invert=True, kernel=20, control_scratch=False, gpu_obj=None, is_inlay=False, counter=0):
    
    if object_color == 'white':
        if bbox_invert:
            src_frame = cv2.bitwise_not(src_frame)
        #src_frame = src_frame
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


    image_pack = [src_frame, closing, fill, fill_inv, combine]
    tittle_pack = [str(counter) + "_1_src_frame", str(counter) + "_2_closing", str(counter) + "_3_fill", str(counter) + "_4_fill_inv", str(counter) + "_5_combine"]
    save_image(image_pack, path="temp_files/extractor_centroid/preProcess", filename=tittle_pack, format="png")
    #show_image(image_pack, title='EC', open_order=1)
    
    return closing, fill_inv, combine


def zone_above_inlay(src_frame, object_color='white', counter=0, kernel=30, threshold=[40, 255, cv2.THRESH_BINARY]):
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
    rso = remove_Small_Object(closing.copy(), ratio=7)[0]

    image_pack = [gray, invert, th, th_invert, combine, closing, rso]
    tittle_pack = ["1_gray" + str(counter), "2_invert" + str(counter), "3_th" + str(counter), "4_th_invert" + str(counter), "5_combine" + str(counter), "6_closing" + str(counter), "7_rso" + str(counter)]
    save_image(image_pack, path="temp_files/extractor_centroid/zone_above_inlay", filename=tittle_pack, format="png")
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

    display_image_symbols, bb_cnt, _, centroids_of_symbols, _ = coord(rso)
    
    
    return display_image_symbols, bb_cnt, np.array(centroids_of_symbols)


def coord(src_frame):
    contours, _ = cv2.findContours(src_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    """diss = cv2.cvtColor(src_frame, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours):
        for k in c:
            cv2.circle(diss, (int(k[0][0]), int(k[0][1])) , 2, (0,255,0), -1)
    show_image(diss, title='COORDS', open_order=1)"""
    
    ###CENTROID###
    contours_poly = [None]*len(contours)
    bb_cnt = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bb_cnt[i] = cv2.boundingRect(contours_poly[i])
        
    #centroids_of_symbols = contour_Centroids(contours, get_bbox_centroid=False)
    centroids_of_symbols = contour_Centroids(bb_cnt, get_bbox_centroid=True)
        
    ###EXTREME###
    coords_ext = list()
    bb_ext = list()
    for i in range(len(contours)):
        temp = contour_Extreme_Points(contours[i], get_centroid=True, is_width_height=True)
        bb_ext.append(tuple(temp[:len(temp)-1]))
        coords_ext.append(temp[-1])
    coords_ext = np.array(coords_ext)
    
    #import pdb; pdb.set_trace()
    
    ###NORMOLIZATION EXTREME TO CENTROID###
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

        
        """norm_bb_cnt[-1].append(bb_cnt[i][0]) 
        norm_bb_cnt[-1].append(bb_ext[i][1])
        norm_bb_cnt[-1].append(bb_cnt[i][2])
        norm_bb_cnt[-1].append(bb_ext[i][3])"""

    bb_cnt = norm_bb_cnt

    
    ###DISPLAY###
    display_image_symbols = cv2.cvtColor(src_frame.copy(), cv2.COLOR_GRAY2BGR)
    
    for i, value in enumerate(centroids_of_symbols):
        cv2.circle(display_image_symbols, (value[0], value[1]) , 5, (0,255,0), -1)
        cv2.rectangle(display_image_symbols, (int(bb_cnt[i][0]), int(bb_cnt[i][1]), int(bb_cnt[i][2]), int(bb_cnt[i][3])), (24,132,255), 1)
    
    for i, value in enumerate(coords_ext):
        cv2.circle(display_image_symbols, (int(value[0]), int(value[1])) , 5, (0,0,255), -1)
        cv2.rectangle(display_image_symbols, (int(bb_ext[i][0]), int(bb_ext[i][1]), int(bb_ext[i][2]), int(bb_ext[i][3])), (255,132,24), 1)
    #show_image(display_image_symbols)
    return display_image_symbols, bb_cnt, bb_ext, centroids_of_symbols, coords_ext
