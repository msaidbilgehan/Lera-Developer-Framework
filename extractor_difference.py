#cython: language_level=3, boundscheck=False
# python37 .\tools\background_subtraction.py -r c:\Users\said.bilgehan\Workspace\Python\lazer-baski-sonrasi-panolarin-goruntu-kontrolu\images\#1#\#HOMO#\reference-1765050009-SIDE_RIGHT.png -i c:\Users\said.bilgehan\Workspace\Python\lazer-baski-sonrasi-panolarin-goruntu-kontrolu\images\#1#\#HOMO#\samplehomo-1765050009-SIDE_RIGHT.png -w
"""
    usage: background_subtraction.py [-h] -r REFERENCE -i INPUT [-o OUTPUT]
                                 [-v [VERBOSE]] [-tw] [-w]

    Program for create mask of difference between reference and input image.

    optional arguments:
    -h, --help            show this help message and exit
    -r REFERENCE, --reference REFERENCE
                            path to reference image
    -i INPUT, --input INPUT
                            path to input image
    -o OUTPUT, --output OUTPUT
                            path to output image (mask)
    -v [VERBOSE], --verbose [VERBOSE]
                            detailed output
    -tw, --test-windows   detailed image windows at every step
    -w, --window          detailed output
"""


from pickle import TRUE
import argparse
import time
import math
import cv2
import numpy as np
from skimage.filters import threshold_li, roberts
from skimage.metrics import mean_squared_error, structural_similarity
from skimage import img_as_ubyte
# import cupy as cp

from stdo import stdo
from tools import time_log, time_list, get_time  # Basic Tools
from image_tools import save_image
from image_manipulation import sharpening, remove_Small_Object, contour_Centroids, remove_Pixel_Width, get_Contours_and_Hull, draw_Hulls_and_Bbox, get_Largest_Contour_Object, diff_Image_Masking, remove_Pixel_With_Distance_Transform, get_Thickness_With_Distance_Transform, edge_Image_Masking
from math_tools import normalize_Comma_to_Dot_Float, point_In_Bbox_Match

arguments = None


arguments = list()
arguments_output = str()

# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
def argument_control(scenario=None, img_path="test.png"):
    global arguments
    description = (
        "Program for create mask of difference between reference and input image."
    )

    if scenario is None:
        argument_parser = argparse.ArgumentParser(
            description=description
        )  # Parse arguments

        argument_parser.add_argument(
            "-r",
            "--reference",
            required=True,
            type=str,
            help="path to reference image",
        )  # Image path argument

        argument_parser.add_argument(
            "-i",
            "--input",
            default="input.png",
            required=True,
            type=str,
            help="path to input image",
        )  # Image path argument

        argument_parser.add_argument(
            "-o",
            "--output",
            default="output.png",
            type=str,
            help="path to output image (mask)",
        )  # Image path argument

        argument_parser.add_argument(
            "-v",
            "--verbose",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=False,
            help="detailed output",
        )
        """
        argument_parser.add_argument(
            "-w",
            "--window",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=False,
            help="detailed output",
        )
        """
        argument_parser.add_argument(
            "-tw",
            "--test-windows",
            # default=argparse.SUPPRESS,
            action="store_true",
            help="detailed image windows at every step",
        )
        argument_parser.add_argument(
            "-w",
            "--window",
            # default=argparse.SUPPRESS,
            action="store_true",
            help="detailed output",
        )

        # Store arguments as dictionary in shared area
        arguments_to_string(arguments)
        return vars(argument_parser.parse_args())

    else:
        stdo(3, "Wrong scenario parameter: {}".format(scenario))
        return -1


def arguments_to_string(arguments) -> str:
    # https://docs.python.org/3/tutorial/datastructures.html
    # https://stackoverflow.com/questions/9453820/alternative-to-python-string-item-assignment

    # shared.arguments = arguments
    # shared.tempList["imageName"] = shared.arguments["image"].split("/")[-1]

    global arguments_output

    temp_arguments_output = "Given Arguments:\n"
    for argument in list(arguments):
        temp_arguments_output += "\t\t    |- {} : {}\n".format(
            argument, arguments[argument]
        )

    list_of_args = list(temp_arguments_output)
    list_of_args[temp_arguments_output.rfind("|")] = "'"

    for char in list_of_args:
        arguments_output += char

    return arguments_output


def argument_info():
    stdo(1, arguments_output)
    return 0


def str_to_bool(string_parameter):
    if isinstance(string_parameter, bool):
        return string_parameter
    if string_parameter.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif string_parameter.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def resize_to_same(src, dst):
    stdo(1, "Source Shape: {} | Destination Shape: {}".format(src.shape, dst.shape))
    if src.shape != dst.shape:
        if len(src.shape) == 3:
            dim = src.shape[:2]
        else:
            dim = src.shape[:: -1] # For reverse dimensions
        stdo(1, "Destination Shape Resized to : {}".format(dst.shape))
        return cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    else:
        stdo(1, "Source Shape and Destination Shape is same.")
        return dst


def preprocessing_5(frame, object_color='white', dict_color_symbol_extraction={}, pano_sector_index='M'):
    
    if ((frame.shape[1] / frame.shape[0]) > 3) or ((frame.shape[0] / frame.shape[1]) > 3): # Long-Thin Lines #
        
        for i in range(1):
            frame = sharpening(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # lower = dict_color_symbol_extraction[object_color]['BGR']['lower']
        # upper = dict_color_symbol_extraction[object_color]['BGR']['upper']
        
        if (object_color == 'white') or (object_color == 'gray - siyah sembol'):
            lower = np.array([150,190,90], dtype="uint8")
            upper = np.array([255,255,255], dtype="uint8")
        else:
            lower = np.array([150,190,90], dtype="uint8")
            upper = np.array([255,255,255], dtype="uint8")
        
        color_range = cv2.inRange(frame, lower, upper)
        th = color_range
        
    elif (200 > frame.shape[0] > 180 and 200 > frame.shape[1] > 170): # beko -> eo
        
        for i in range(1):
            frame = sharpening(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        th1 = cv2.bitwise_and(th_otsu, th_triangle)
        
        th_li = gray > threshold_li(gray)
        th_li = np.where(th_li, 255, 0).astype(np.uint8)
        th2 = cv2.bitwise_and(th_otsu, th_li)
        
        th3 = cv2.bitwise_and(th2, th1)
        th4 = cv2.bitwise_and(th1, th2)
        th = cv2.add(th3, th4)
    
    elif (frame.shape[0] > 200 and frame.shape[1] > 180): # beko -> bk
        
        for i in range(1):
            frame = sharpening(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        th = cv2.bitwise_and(th_otsu, th_triangle)
        
        # _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        # th_li = gray > threshold_li(gray)
        # th_li = np.where(th_li, 255, 0).astype(np.uint8)
        
        # th_alpha = cv2.bitwise_and(th_triangle, th_li)
        # th_beta = cv2.bitwise_or(th_triangle, th_li)
        # th = cv2.bitwise_xor(th_alpha, th_beta)

    else:
        for i in range(1):
            frame = sharpening(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        th = cv2.bitwise_and(th_otsu, th_triangle)
    
    # lower = np.array(dict_color_symbol_extraction[object_color]["BGR"]['lower'], dtype="uint8")
    # upper = np.array(dict_color_symbol_extraction[object_color]["BGR"]['upper'], dtype="uint8")
    # th = cv2.inRange(frame, lower, upper)
    
    if (object_color == 'white') or (object_color == 'gray - siyah sembol'): 
        th = cv2.bitwise_not(th)
        
    return th


def extractor_difference(
        image_reference, 
        image_sample, 
        method=1, 
        sobel_scale=1, 
        sharp_scale=3, 
        is_contour_number_for_area=False, 
        ratio=0, 
        buffer_percentage=95, 
        is_filter=False, 
        filter_lower_ratio=10, 
        filter_upper_ratio=250, 
        threshold_config=[-1, 3],
        draw_diff=False, 
        draw_diff_circle=False, 
        fill_color=[0, 0, 255], 
        rso_ratio=5,
        decision_ratio=10,
        window=False, 
        bbox_invert=False, 
        counter=0, 
        activate_debug_images=False,
        kernel = [2,2],
        dilate_iteration=1,
        erode_iteration=1,
        pixel_width=2,
        pixel_width_color=[0,0,255],
        pano_sector='SL',
        object_color='white', 
        dict_color_symbol_extraction={},
        image_processing_data_collector=None,
        data_symbol_coords={},
        symbol_start_coords=[],
        redecession_learning_data_collector=None,
    ):
    
    start_seq = time.time()
    flag_symbol_type = 'char'
    
    #### #### #### #### #### ####
    #### ## Pre-Processing ## ###
    #### #### #### #### #### ####
    start_preprocess = time.time()
    
    org_height, org_width, _ = image_reference.shape
    image_reference_pad = image_reference.copy()
    image_sample_pad = image_sample.copy()
    
    if (org_width / org_height > 3) or (org_height / org_width > 3): # Long-Thin Lines #
        
        if (object_color == 'white') or (object_color == 'gray - siyah sembol'): 
            color = [255,255,255]
        else:
            color = [0,0,0]
        
        # Calculate padding sizes
        if (org_width > org_height): # Reference is taller than wider
            pad_top, pad_bottom = 20, 20
            pad_left, pad_right = 5, 5
        else:
            pad_top, pad_bottom = 5, 5
            pad_left, pad_right = 10, 10
        image_reference_pad = cv2.copyMakeBorder(image_reference.copy(), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, value=color) # top, bottom, left, right,
        image_sample_pad = cv2.copyMakeBorder(image_sample.copy(), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None, value=color) # top, bottom, left, right,
        
        # padded_image_ref = np.zeros((org_width + pad_top + pad_bottom, org_height + pad_left + pad_right, 3), dtype=image_reference.dtype)
        # padded_image_ref[:,:,:] = color
        # padded_image_ref[pad_top:-pad_bottom, pad_left:-pad_right] = image_reference.copy()
        # image_reference = padded_image_ref.copy()

        # padded_image_sample = np.zeros((org_width + pad_top + pad_bottom, org_height + pad_left + pad_right, 3), dtype=image_sample.dtype)
        # padded_image_sample[:,:,:] = color
        # padded_image_sample[pad_top:-pad_bottom, pad_left:-pad_right] = image_sample
        # image_sample = padded_image_sample.copy()
        
        flag_symbol_type = 'line'

    elif (org_width <= 40) and (org_height <= 40): # small char #
        flag_symbol_type = 'small_char'
    
    image_reference_preprocessed = preprocessing_5(image_reference_pad, object_color=object_color, dict_color_symbol_extraction=dict_color_symbol_extraction, pano_sector_index=pano_sector)
    image_sample_preprocessed = preprocessing_5(image_sample_pad, object_color=object_color, dict_color_symbol_extraction=dict_color_symbol_extraction, pano_sector_index=pano_sector)
    
    stop_preprocess = time.time() - start_preprocess
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### ## Largest Contour # ###
    #### #### #### #### #### ####
    start_largest_contour = time.time()
    
    if (org_width / org_height > 3) or (org_height / org_width > 3): # Long-Thin Lines #
        image_reference_preprocessed_largest_contour = image_reference_preprocessed
        image_sample_preprocessed_largest_contour = image_sample_preprocessed
    else:
        image_reference_preprocessed_largest_contour = get_Largest_Contour_Object(image_reference_preprocessed)
        image_sample_preprocessed_largest_contour = get_Largest_Contour_Object(image_sample_preprocessed)
    
    stop_largest_contour = time.time() - start_largest_contour
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### #### Hull #### #### ####
    #### #### #### #### #### ####
    start_hull = time.time()
    
    ref_contours, _ = get_Contours_and_Hull(image_reference_preprocessed_largest_contour.copy())
    sample_contours, sample_hulls = get_Contours_and_Hull(image_sample_preprocessed_largest_contour.copy())
    
    dict_sample_symbol_contours = dict()
    # dict_sample_symbol_contours.append(dict())
    # dict_sample_symbol_contours[counter] --> counter = symbol_id
    dict_sample_symbol_contours["pano_sector_index"] = pano_sector
    dict_sample_symbol_contours["symbol_id"] = counter
    
    # area_hull = cv2.contourArea(sample_hulls[0])
    # rect_hull = cv2.minAreaRect(sample_hulls[0])
    # w_hull = rect_hull[1][0]
    # h_hull = rect_hull[1][1]
    # dict_sample_symbol_contours["hull_w"] = round(w_hull, 2)
    # dict_sample_symbol_contours["hull_h"] = round(h_hull, 2)
    # dict_sample_symbol_contours["hull_area"] = round(area_hull, 2)
    # dict_sample_symbol_contours["hull_contours"] = sample_contours
    
    stop_hull = time.time() - start_hull
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### Warp Perspective ## ####
    #### #### #### #### #### ####
    start_warp_perspective = time.time()
    
    if (
            (org_width / org_height > 3) or (org_height / org_width > 3) # Long-Thin Lines #
        ) or (
            (205 > org_width > 190) and (200 > org_height > 180) # beko -> o #
        ):

        img_bbox_ref, ref_hulls, ref_img_box = draw_Hulls_and_Bbox(image_reference_preprocessed_largest_contour.copy(), ref_contours, flag_activate_debug_images=activate_debug_images)
        img_bbox_sample, sample_hulls, sample_img_bbox = draw_Hulls_and_Bbox(image_sample_preprocessed_largest_contour.copy(), sample_contours, flag_activate_debug_images=activate_debug_images)
        
        if ref_img_box is not None and sample_img_bbox is not None:
            M = cv2.getPerspectiveTransform(np.float32(sample_img_bbox), np.float32(ref_img_box))
            warped_sample = cv2.warpPerspective(image_sample_preprocessed_largest_contour.copy(), M, (image_sample_preprocessed_largest_contour.shape[1], image_sample_preprocessed_largest_contour.shape[0]))
            image_sample_pad_warped = cv2.warpPerspective(image_sample_pad.copy(), M, (image_sample_pad.shape[1], image_sample_pad.shape[0]))
        else:
            warped_sample = image_sample_preprocessed_largest_contour.copy()
            image_sample_pad_warped = image_sample_pad.copy()
        image_sample_preprocessed_largest_contour = warped_sample
    else: # beko -> e #
        img_bbox_ref = image_reference_preprocessed_largest_contour.copy()
        img_bbox_sample = image_sample_preprocessed_largest_contour.copy()
        image_sample_pad_warped = image_sample_pad.copy()
        
    stop_warp_perspective = time.time() - start_warp_perspective
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### ## Edge Detection # ####
    #### #### #### #### #### ####
    start_edge_detection = time.time()
    
    roberts_ref = roberts(image_reference_preprocessed_largest_contour)
    roberts_ref = img_as_ubyte(roberts_ref)
    _, roberts_ref = cv2.threshold(roberts_ref, 0, 255, cv2.THRESH_OTSU)
    contours_roberts_ref, _ = cv2.findContours(roberts_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contours_roberts_ref = len(contours_roberts_ref)
    
    roberts_sample = roberts(image_sample_preprocessed_largest_contour)
    roberts_sample = img_as_ubyte(roberts_sample)
    _, roberts_sample = cv2.threshold(roberts_sample, 0, 255, cv2.THRESH_OTSU)
    contours_roberts_sample, _ = cv2.findContours(roberts_sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contours_roberts_sample = len(contours_roberts_sample)
    
    stop_edge_detection = time.time() - start_edge_detection
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### ### Subtracting ### ####
    #### #### #### #### #### ####
    start_subtract = time.time()
    
    diff = cv2.absdiff(image_reference_preprocessed_largest_contour, image_sample_preprocessed_largest_contour)
    mask = edge_Image_Masking(diff_image=diff, edge_image_sample=roberts_sample)
    
    stop_subtract = time.time() - start_subtract
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### #### ####
    ## Detect Inside-Outside Faults ##
    #### #### #### #### #### #### ####
    start_elim = time.time()
    
    if activate_debug_images:
        draw_org_diff, list_inside, list_outside = diff_Image_Masking(org_image=image_sample_pad_warped, diff_image=diff, edge_image_sample=roberts_sample, edge_image_ref=roberts_ref)
    
    # image_sample_preprocessed_largest_contour_not = cv2.bitwise_not(image_sample_preprocessed_largest_contour)
    # combine_diff_outside_real_fault = cv2.bitwise_and(diff, image_sample_preprocessed_largest_contour_not)
    
    image_sample_preprocessed_largest_contour_not = cv2.bitwise_not(image_sample_preprocessed_largest_contour)
    diff_outside = cv2.bitwise_and(mask, image_sample_preprocessed_largest_contour_not)
    _, diff_outside = cv2.threshold(diff_outside, 0, 255, cv2.THRESH_OTSU)
    contours_outside, _ = cv2.findContours(diff_outside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    diff_inside = cv2.subtract(cv2.bitwise_and(mask, image_sample_preprocessed_largest_contour), roberts_sample)
    _, diff_inside = cv2.threshold(diff_inside, 0, 255, cv2.THRESH_OTSU)
    contours_inside, _ = cv2.findContours(diff_inside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    stop_elim = time.time() - start_elim
    #### #### #### #### #### #### ####
    #### #### #### #### #### #### ####
    #### #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### Distance Transform #####
    #### #### #### #### #### ####
    start_dt = time.time()
    
    if (org_width / org_height > 6) or (org_height / org_width > 6): # Long-Thin Lines #
        rpw_method = 8
    else:
        rpw_method = 6
        
        
    temp_flag_symbol_type = point_In_Bbox_Match(point=symbol_start_coords, bboxes=data_symbol_coords[pano_sector])
    if temp_flag_symbol_type is not None:
        flag_symbol_type = temp_flag_symbol_type
        pixel_width = data_symbol_coords[pano_sector][flag_symbol_type]['pano_rpw_ratio']
    else:
        pixel_width = data_symbol_coords[pano_sector][flag_symbol_type]['pano_rpw_ratio']

    pixel_width = normalize_Comma_to_Dot_Float(float(pixel_width))
    rpw_ratio_px = pixel_width * 20.244 # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025   
     
    
    if activate_debug_images:
        ropw_color_outside, _ = remove_Pixel_Width(diff_outside.copy(), contours_outside, pixel_width=rpw_ratio_px, pixel_width_color=pixel_width_color, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls, is_color=True, title='outside')
        ropw_color_inside, _ = remove_Pixel_Width(diff_inside.copy(), contours_inside, pixel_width=rpw_ratio_px, pixel_width_color=pixel_width_color, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls, is_color=True, title='inside')
    
    rpw_outside, dict_rpw_contours = remove_Pixel_Width(diff_outside.copy(), contours_outside, pixel_width=rpw_ratio_px, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls, title='outside')
    contours_outside, _ = cv2.findContours(rpw_outside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if flag_symbol_type == 'line':
        rpw_ratio_px_for_outside = rpw_ratio_px
    else:
        rpw_ratio_px_for_outside = 0.1 * 20.244 # 0.1mm * 20.244px -> 2.0244px
    
    if flag_symbol_type == 'line':
        rpw_outside_filtered, _ = remove_Pixel_With_Distance_Transform(rpw_outside.copy(), contours_outside, thin_threshold=rpw_ratio_px_for_outside)
    else:
        if len_contours_roberts_ref == len_contours_roberts_sample:
            rpw_outside_filtered, _ = remove_Pixel_With_Distance_Transform(rpw_outside.copy(), contours_outside, thin_threshold=rpw_ratio_px_for_outside)
        else:
            rpw_outside_filtered = rpw_outside.copy()
            # rpw_outside_filtered, _ = remove_Pixel_With_Distance_Transform(rpw_outside.copy(), contours_outside, thin_threshold=rpw_ratio_px_for_outside)
    
    rpw_inside, dict_rpw_contours = remove_Pixel_Width(diff_inside.copy(), contours_inside, pixel_width=rpw_ratio_px, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls, title='inside')
    contours_inside, _ = cv2.findContours(rpw_inside, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rpw_inside_filtered, list_max_thickness = remove_Pixel_With_Distance_Transform(rpw_inside.copy(), contours_inside, thin_threshold=rpw_ratio_px)
    # _, _, _, _, not_removed_buffer_for_inside = remove_Small_Object(
    #     rpw_inside_filtered.copy(),
    #     ratio=1
    # )
    
    combine_both = cv2.add(rpw_outside_filtered, rpw_inside_filtered)
    
    # rso_ratio = normalize_Comma_to_Dot_Float(float(rso_ratio))
    # rso_ratio_px = rso_ratio * 20.244 # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
    # combine_both_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
    #     combine_both.copy(),
    #     ratio=int(rso_ratio_px*rso_ratio_px)
    # )
    combine_both_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
        combine_both.copy(),
        ratio=1
    )
    
    if (org_width / org_height > 3) or (org_height / org_width > 3): # Long-Thin Lines #
        contours = [cnt - [pad_left, pad_top] for cnt in contours]
        not_removed_buffer = [cnt - [pad_left, pad_top] for cnt in not_removed_buffer]
        combine_both = combine_both[pad_top:-pad_bottom, pad_left:-pad_right]
        combine_both_rso = combine_both_rso[pad_top:-pad_bottom, pad_left:-pad_right]
        
    image_sample_masked_gray_removed_small_objects = combine_both
    centroid_contour = contour_Centroids(not_removed_buffer)
    
    stop_dt = time.time() - start_dt
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    
    #### #### #### #### #### ####
    #### #### Redecision ### ####
    #### #### #### #### #### ####
    start_redecision = time.time()

    rpw_elements = dict_sample_symbol_contours
    list_thickness = []
    
    flag_decision = 'OK'

    if centroid_contour:
        flag_decision = 'NOK'
        
        if not_removed_buffer:
            
            list_thickness = get_Thickness_With_Distance_Transform(combine_both_rso, not_removed_buffer)
            
            ########### FOR-REDECISION-COLLECTOR ##########
            temp_buffer_data_collector = [
                pano_sector + "_" + str(counter),
                "",
                "",
                "",
                "",
                object_color,
                image_sample_pad_warped,
                diff,
                roberts_sample,
                roberts_ref,
                centroid_contour,
                list_thickness,
            ]
            image_processing_data_collector.buffer_add(temp_buffer_data_collector)
            ###############################################
    
    temp_buffer_redess_data_collector = [
        object_color,
        image_sample_preprocessed_largest_contour,
        roberts_sample,
        flag_decision,
        counter,
        symbol_start_coords,
    ]
    redecession_learning_data_collector.buffer_add(temp_buffer_redess_data_collector)
    
    stop_redecision = time.time() - start_redecision
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    #### #### #### #### #### ####
    
    stop_seq = time.time() - start_seq
    
    if activate_debug_images:
        
        # stdo(1, "[{}][{}][{}] T:{:.3f} | pp:{:.3f} | lc:{:.3f} | hull:{:.3f} | wp:{:.3f} | ed:{:.3f} | sub:{:.3f} | elim:{:.3f} | dt:{:.3f} | r:{:.3f} - Matched Area:{} | Inside Thickness:{}".format(
        #         "ED",
        #         counter,
        #         flag_symbol_type,
        #         stop_seq,
        #         stop_preprocess,
        #         stop_largest_contour,
        #         stop_hull,
        #         stop_warp_perspective,
        #         stop_edge_detection,
        #         stop_subtract,
        #         stop_elim,
        #         stop_dt,
        #         stop_redecision,
        #         all_contour_area,
        #         list_max_thickness
        #     )
        # )
        stdo(1, "[{}][{}][{}] T:{:.3f} | Matched Area:{} | Inside Thickness:{}".format(
                "ED",
                counter,
                flag_symbol_type,
                stop_seq,
                all_contour_area,
                list_max_thickness
            )
        )
        
        list_temp_save_image = [
            image_reference_pad, 
            image_sample_pad,
            image_reference_preprocessed_largest_contour, 
            image_sample_preprocessed_largest_contour, 
            img_bbox_ref, 
            img_bbox_sample,
            diff,
            mask, 
            draw_org_diff, 
            diff_outside,
            ropw_color_outside,
            rpw_outside,
            rpw_outside_filtered,
            diff_inside,
            ropw_color_inside, 
            rpw_inside,
            rpw_inside_filtered,
            combine_both_rso
        ]
        filename = [
            str(counter)+"_1_ref_pad", 
            str(counter)+"_2_sample_pad",        
            str(counter)+"_3_ref_pre", 
            str(counter)+"_4_sample_pre", 
            str(counter)+"_5_ref_bbox", 
            str(counter)+"_6_sample_bbox", 
            str(counter)+"_7_diff", 
            str(counter)+"_8_mask", 
            str(counter)+"_9_draw_org_diff",
            str(counter)+"_10_diff_outside",
            str(counter)+"_11_ropw_color_outside",
            str(counter)+"_12_rpw_outside",
            str(counter)+"_13_rpw_outside_filtered",
            str(counter)+"_14_diff_inside",
            str(counter)+"_15_ropw_color_inside",
            str(counter)+"_16_rpw_inside",
            str(counter)+"_17_rpw_inside_filtered",
            str(counter)+"_18_combine_both_rso",
        ]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-"+str(method), filename=filename, format="png")
            
    
    return image_sample_masked_gray_removed_small_objects, image_sample, contours, not_removed_buffer, centroid_contour, diff, rpw_elements, dict_sample_symbol_contours, list_thickness, flag_symbol_type