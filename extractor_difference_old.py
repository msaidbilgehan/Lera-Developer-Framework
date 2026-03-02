# cython: language_level=3, boundscheck=False
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

# from pickle import TRUE
import argparse
import time
import math
import cv2
import numpy as np
from skimage.filters import threshold_li
from skimage.metrics import mean_squared_error, structural_similarity
# import cupy as cp

from stdo import stdo
from tools import time_log, time_list, get_time  # Basic Tools
from image_tools import open_image, show_image, save_image
from image_manipulation import sharpening, sobel_gradient, erosion, threshold, remove_Small_Object, look_Up_Table, sobel_gradient, draw_Circle, contour_Centroids, contour_Areas, transparent_Draw, pruning, adjust_brightness, adjust_Contrast, gamma_Correction, remove_Pixel_Width, get_Contours_and_Hull, draw_Hulls_and_Bbox


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


def sword_of_justice(src_frame, counter):
    kernel = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    filt1 = cv2.filter2D(src_frame, -1, kernel, cv2.BORDER_DEFAULT)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    filt2 = cv2.filter2D(src_frame, -1, kernel, cv2.BORDER_DEFAULT)

    combine = filt1 + filt2

    sub = src_frame - combine

    image_pack = [src_frame, filt1, filt2, combine, sub]
    tittle_pack = [str(counter)+"_"+"1src_frame",str(counter)+"_"+ "2filt1", str(counter)+"_"+"3filt2", str(counter)+"_"+"4combine", str(counter)+"_"+"5sub"]
    save_image(image_pack, path="temp_files/extractor_difference/sword", filename=tittle_pack, format="png")
    return sub


def preprocessing_4(frame, threshold_config):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    gray_inv = cv2.bitwise_not(gray)
    _, th = cv2.threshold(gray_inv, *threshold_config)# 40, 255, cv2.THRESH_BINARY)

    #show_pack = [gray, gray_inv, th]
    #show_image(show_pack, open_order=1, window=True)

    return th

def preprocessing_black(frame, threshold_config=210, counter=0):
    gamma = 2
    lookUpTable = np.empty((1,256), np.uint8)
    for j in range(256):
        lookUpTable[0,j] = np.clip(pow(j / 255.0, float(gamma)) * 255.0, 0, 255)
    gc = cv2.LUT(frame, lookUpTable)
    #sharp = sharpening(frame)

    gray = cv2.cvtColor(gc, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, *threshold_config) #Also working with 210 threshold coefficient#

    #show_pack = [frame, sharp, gray, th]
    #show_image(show_pack, title='P', open_order=1, window=True)

    image_pack = [frame, gc, gray, th]
    tittle_pack = [str(counter)+"_"+"1frame",str(counter)+"_"+ "2gamma", str(counter)+"_"+"3gray", str(counter)+"_"+"4th"]
    save_image(image_pack, path="temp_files/extractor_difference/preprocessing_black", filename=tittle_pack, format="png")

    return th


def preprocessing_3(frame, threshold_config, sharp_scale=3, sobel_scale=0.05, counter=0, pano_title='p'):
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # sharp = sharpening(gray)

    # SHARP
    if sharp_scale != 0:
        image_sharpened = frame

        for i in range(sharp_scale):
            image_sharpened = sharpening(image_sharpened)
    else:
        image_sharpened = frame.copy()

    # SOBEL
    if sobel_scale != 0:
        image_sharpened_sobel = sobel_gradient(image_sharpened, sobel_scale)
    else:
        image_sharpened_sobel = image_sharpened.copy()


    th_not = cv2.bitwise_not(image_sharpened_sobel)
    _, th = cv2.threshold(th_not, *threshold_config) #135, 255, cv2.THRESH_BINARY)
    # th = cv2.adaptiveThreshold(th_not, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 7)

    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
    # closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_opening)
    closing = th
    # rso = remove_Small_Object(closing.copy(), ratio=1000)[0]

    # show_pack = [frame, th, th_not, closing] #, rso]
    # show_image(show_pack, open_order=1)

    list_temp_save_image = [frame, image_sharpened, image_sharpened_sobel, th]
    filename = ["1_image", "2_image_sharpened", "3_image_sharpened_sobel", "4_th"]
    filename = [
        pano_title+"_"+str(counter)+"_1_image",
        pano_title+"_"+str(counter)+"_2_image_sharpened",
        pano_title+"_"+str(counter)+"_3_image_sharpened_sobel",
        pano_title+"_"+str(counter)+"_4_image_sharpened_sobel_threshold",
    ]
    save_image(list_temp_save_image, path="temp_files/extractor_difference/preprocessing_3", filename=filename, format="png")

    return closing

def preprocessing_2(frame, threshold_config, counter, bbox_invert=False):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    #sharp = sharpening(gray)
    #_, th = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY_INV)
    #th_not = cv2.bitwise_not(th)

    th = threshold(gray.copy(), configs=threshold_config)
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_opening)

    if bbox_invert:
        closing = cv2.bitwise_not(closing)

    rso = remove_Small_Object(th.copy(), ratio=10)[0]

    if bbox_invert:
        rso = cv2.bitwise_not(rso)

    """
    kernel = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )
    filt1 = cv2.filter2D(rso, -1, kernel, cv2.BORDER_DEFAULT)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    filt2 = cv2.filter2D(rso, -1, kernel, cv2.BORDER_DEFAULT)

    combine = filt1 + filt2

    sub = rso - combine

    #show_pack = [gray, th, closing, rso, filt1, filt2, combine, sub]
    #show_image(show_pack, open_order=1, window=True)
    # import pdb; pdb.set_trace()
    return gray, th, closing, rso, filt1, filt2, combine, sub

    #  """

    sub = rso


    #image_pack = [gray, th, closing, rso, sub]
    #tittle_pack = [str(counter)+"_"+"1gray",str(counter)+"_"+ "2th", str(counter)+"_"+"3closing", str(counter)+"_"+"4rso", str(counter)+"_"+"8sub"]
    #save_image(image_pack, path="temp_files/extractor_difference/preprocessing2", filename=tittle_pack, format="png")

    return gray, th, closing, rso, sub

def preprocessing_5(frame, object_color='white', dict_color_symbol_extraction={}):

    # for i in range(1): # 20.07.2025
    #     sharpened = sharpening(frame)
    # gamma_corrected = exposure.adjust_gamma(sharpened.copy(), gamma=2, gain=1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if (
        ((frame.shape[1] / frame.shape[0]) >= 4) or ((frame.shape[0] / frame.shape[1]) >= 4)
    ):
        for i in range(1):
            frame = sharpening(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        th = cv2.bitwise_and(th_otsu, th_triangle)

    if (
        ((frame.shape[1] / frame.shape[0]) >= 10) or ((frame.shape[0] / frame.shape[1]) >= 10)
    ):
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # th_li = gray > threshold_li(gray)
        # th_li = np.where(th_li, 255, 0).astype(np.uint8)
        # th = cv2.bitwise_and(th_otsu, th_li)
        # _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        # _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
        # th = cv2.bitwise_and(th_otsu, th_triangle)

        # lower = dict_color_symbol_extraction[object_color]['BGR']['lower']
        # upper = dict_color_symbol_extraction[object_color]['BGR']['upper']
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

    elif (frame.shape[0] > 200 and frame.shape[1] > 180):  # beko -> bk

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

    # elif (frame.shape[0] > 200 and frame.shape[1] > 180):
    #     # _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
    #     # th_li = gray > threshold_li(gray)
    #     # th_li = np.where(th_li, 255, 0).astype(np.uint8)
    #     # th = cv2.bitwise_and(th_triangle, th_li)
    #     _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    #     th_li = gray > threshold_li(gray)
    #     th_li = np.where(th_li, 255, 0).astype(np.uint8)
    #     th = cv2.bitwise_and(th_otsu, th_li)

    # elif (frame.shape[0] > 100 and frame.shape[1] > 100):
    #     _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    #     _, th_triangle = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
    #     th = cv2.bitwise_and(th_otsu, th_triangle)

    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        th_li = gray > threshold_li(gray)
        th_li = np.where(th_li, 255, 0).astype(np.uint8)
        th = cv2.bitwise_and(th_otsu, th_li)

    # lower = np.array(dict_color_symbol_extraction[object_color]["BGR"]['lower'], dtype="uint8")
    # upper = np.array(dict_color_symbol_extraction[object_color]["BGR"]['upper'], dtype="uint8")
    # th = cv2.inRange(frame, lower, upper)

    if (object_color == 'white') or (object_color == 'gray - siyah sembol'):
        th = cv2.bitwise_not(th)

    return th


def preprocessing(image, method=1, is_sharp=True, is_sobel=True, is_threshold=True, window=False, sharp_scale=3, open_order=1, sobel_scale=0.05, threshold_config=[-1, -1], is_middle_object=False, middle_object_roi=[], middle_object_roi_2=[], pano_sector='', counter=0, activate_debug_images=False):
    # GRAYSCALE
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rrggbb = None
    # SHARP
    if is_sharp and sharp_scale != 0:
        image_sharpened = image
        """
        # FOR Strecth
        image_sharpened = gamma_correction(image_sharpened, gamma=2)
        image_sharpened = adjust_contrast(image_sharpened, contrast_factor=1.5)
        """

        for i in range(sharp_scale):
            image_sharpened = sharpening(image_sharpened)

        # show_image([image, image_sharpened, image_sharpened_c, image_sharpened_b, image_sharpened_g], ["image", "image_sharpened", "image_sharpened_c", "image_sharpened_b", "image_sharpened_g"], open_order=1)
        # show_image([image, image_sharpened], ["image", "image_sharpened"], open_order=1)

    else:
        image_sharpened = image.copy()

    # SOBEL
    if is_sobel and sobel_scale != 0:
        image_sharpened_sobel = sobel_gradient(image_sharpened, sobel_scale)
    else:
        image_sharpened_sobel = image_sharpened.copy()

    # THRESHOLD
    if is_threshold:
        image_sharpened_sobel_threshold = threshold(image_sharpened_sobel.copy(), configs=threshold_config)
        # FOR Strecth
        # image_sharpened_sobel_threshold = cv2.adaptiveThreshold(image_sharpened_sobel.copy(), threshold_config[0], cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_config[2], 251, 33)
    else:
        image_sharpened_sobel_threshold = image_sharpened_sobel.copy()

    if is_middle_object:
        mask_cekmece = image_sharpened_sobel_threshold.copy()

        # print("MID-ROI:", mask_cekmece.shape, middle_object_roi)

        """if threshold_config[2] == 'cv2.THRESH_BINARY_INV':
            color = 255
        else:
            color = 0"""

        # print("Ext-difference/preprocessing-middle_object_roi:", middle_object_roi)
        mask_cekmece[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 0
        if middle_object_roi_2:
            mask_cekmece[middle_object_roi_2[0]:middle_object_roi_2[1], middle_object_roi_2[2]:middle_object_roi_2[3]] = 0

        image_sharpened_sobel_threshold = mask_cekmece

        # show_image([image, image_sharpened_sobel_threshold, mask_cekmece], title='P', open_order=2)

        if activate_debug_images:
            rrggbb = cv2.cvtColor(image_sharpened_sobel_threshold.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(rrggbb, (middle_object_roi[2], middle_object_roi[0]), (middle_object_roi[3], middle_object_roi[1]), (0,255,0), 3)

    # MORPH
    if method == 3:
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
        openinig = cv2.morphologyEx(image_sharpened_sobel_threshold, cv2.MORPH_OPEN, kernel_opening)
        image_sharpened_sobel_threshold = remove_Small_Object(openinig.copy(), ratio=10)[0]
    if method == 4:


        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
        # image_sharpened_sobel_threshold = cv2.bitwise_not(image_sharpened_sobel_threshold)
        closing = cv2.morphologyEx(image_sharpened_sobel_threshold, cv2.MORPH_CLOSE, kernel_opening)
        image_sharpened_sobel_threshold = remove_Small_Object(closing.copy(), ratio=1000)[0]

        # th = cv2.bitwise_not(th)
        kernel = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )
        filt1 = cv2.filter2D(image_sharpened_sobel_threshold, -1, kernel, cv2.BORDER_DEFAULT)

        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        filt2 = cv2.filter2D(image_sharpened_sobel_threshold, -1, kernel, cv2.BORDER_DEFAULT)

        combine = filt1 + filt2
        sub = image_sharpened_sobel_threshold - combine

        image_sharpened_sobel_threshold = sub

        # image_sharpened_sobel_threshold &= image_sharpened_sobel_threshold_2

        # erode = cv2.bitwise_not(erode)
        # contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if False:
            if is_middle_object:
                show_pack = [image, image_sharpened, mask_cekmece, closing, image_sharpened_sobel_threshold, filt1, filt2, combine, sub]
            else:
                show_pack = [image, image_sharpened, closing, image_sharpened_sobel_threshold, filt1, filt2, combine, sub]
            show_image(show_pack, title='P', open_order=2)

    """
    image_reference_preprocessed[image_reference_preprocessed < 110] = [0]
    image_test_preprocessed[image_test_preprocessed < 110] = [0]
    """

    """
    show_image(
        # (image_reference, image_test, image_test_rgb),
        (
            image,
            image_sharpened,
            image_sharpened_sobel,
            image_sharpened_sobel_threshold,
        ),
        title=["image", "sharpened", "sobel", "threshold"],
        option="plot",
        cmap="gray",
        window=window,
        open_order=open_order,
    )
    """

    if method == 1 and is_middle_object:
        list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold, rrggbb]
        filename = ["1_image", "2_image_sharpened", "3_image_sharpened_sobel", "4_image_sharpened_sobel_threshold", "5_rrggbb_middle_object_roi"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/preprocessing", filename=filename, format="png")

    if activate_debug_images and (pano_sector != ''):
        list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold]
        filename = ["1_image", "2_image_sharpened", "3_image_sharpened_sobel", "4_image_sharpened_sobel_threshold"]
        filename = [
            pano_sector+"_" + "1_image",
            pano_sector+"_" + "2_image_sharpened",
            pano_sector+"_" + "3_image_sharpened_sobel",
            pano_sector+"_" + "4_image_sharpened_sobel_threshold",
        ]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/pano_sector/preprocessing", filename=filename, format="png")

    # list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold]
    # filename = [str(counter)+"_1_image", str(counter)+"_2_image_sharpened", str(counter)+"_3_image_sharpened_sobel", str(counter)+"_4_image_sharpened_sobel_threshold"]
    # save_image(list_temp_save_image, path="temp_files/extractor_difference/preprocessing", filename=filename, format="png")

    return image_sharpened_sobel_threshold, image_sharpened, image_sharpened_sobel


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
    kernel=[2, 2],
    dilate_iteration=1,
    erode_iteration=1,
    pixel_width=2,
    pixel_width_color=[0, 0, 255],
    pano_sector='UL',
    object_color='white',
    dict_color_symbol_extraction={}
):

    image_subtracted_prc = -1
    centroid_contour = -1

    image_reference_gray = None
    image_sample_gray = None

    start_seq = time.time()

    # GRAYSCALE CONVERSION
    if method != 11:
        if len(image_reference.shape) == 3:
            if image_reference.shape[-1] == 3:
                image_reference_gray = cv2.cvtColor(image_reference.copy(), cv2.COLOR_RGB2GRAY)
        else:
            image_reference_gray = image_reference.copy()
        if len(image_sample.shape) == 3:
            if image_reference.shape[-1] == 3:
                image_sample_gray = cv2.cvtColor(image_sample.copy(), cv2.COLOR_RGB2GRAY)
        else:
            image_sample_gray = image_sample.copy()

    start_preprocess = time.time()
    # PREPROCESSING
    if method == 1:
        image_reference_preprocessed, _, _ = preprocessing(image_reference_gray.copy(), method=method, sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window, threshold_config=threshold_config, counter=counter)
        image_sample_preprocessed, _, _ = preprocessing(image_sample_gray.copy(), method=method, sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window, threshold_config=threshold_config, counter=counter+20)

    if method == 5:
        image_reference_preprocessed = preprocessing_3(image_reference_gray, threshold_config=threshold_config_reference)
        image_sample_preprocessed = preprocessing_3(image_sample_gray, threshold_config=threshold_config_sample)

    if method == 4:
        """
        image_reference_preprocessed = preprocessing(image_reference_gray.copy(), method=method, sharp_scale=sharp_scale, is_sobel=False, is_sharp=False, window=window, threshold_config=threshold_config)
        image_sample_preprocessed = preprocessing(image_sample_gray.copy(), method=method, sharp_scale=sharp_scale, is_sobel=False, is_sharp=False, window=window, threshold_config=threshold_config)
        """
        # show_image( [ image_reference_preprocessed, image_sample_preprocessed ], open_order = 1)

        image_reference_preprocessed = preprocessing_2(image_reference, threshold_config=threshold_config_reference, bbox_invert=bbox_invert, counter=counter)[-1]
        image_sample_preprocessed = preprocessing_2(image_sample, threshold_config=threshold_config_sample, bbox_invert=bbox_invert, counter=counter+20)[-1]

        # #############################
        # Save Temp Images For Debug #
        # #############################
        # list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed]
        # filename = ["image_reference" + str(counter), "image_sample" + str(counter), "image_reference_preprocessed" + str(counter), "image_sample_preprocessed" + str(counter)]
        # save_image(list_temp_save_image, path="temp_files/extractor_difference/method-4", filename=filename, format="png")

        """
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-1ref.png", image_reference)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-2sample.png", image_sample)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-3ref_pro.png", image_reference_preprocessed)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-4sample_pro.png", image_sample_preprocessed)
        """

    if method == 9 or method == 10:
        image_reference_preprocessed = preprocessing_3(image_reference_gray, threshold_config=threshold_config, sharp_scale=sharp_scale, sobel_scale=sobel_scale, counter=counter, pano_title='p')
        image_sample_preprocessed = preprocessing_3(image_sample_gray, threshold_config=threshold_config, sharp_scale=sharp_scale, sobel_scale=sobel_scale, counter=counter, pano_title='s')

    if method == 11:

        image_reference_preprocessed = preprocessing_5(image_reference, object_color=object_color, dict_color_symbol_extraction=dict_color_symbol_extraction)
        image_sample_preprocessed = preprocessing_5(image_sample, object_color=object_color, dict_color_symbol_extraction=dict_color_symbol_extraction)

        ##########

        ref_contours, _ = get_Contours_and_Hull(image_reference_preprocessed.copy())
        sample_contours, sample_hulls = get_Contours_and_Hull(image_sample_preprocessed.copy())

        dict_sample_symbol_contours = dict()
        # dict_sample_symbol_contours.append(dict())
        # dict_sample_symbol_contours[counter] --> counter = symbol_id
        dict_sample_symbol_contours["pano_sector_index"] = pano_sector
        dict_sample_symbol_contours["symbol_id"] = counter

        area_hull = cv2.contourArea(sample_hulls[0])
        rect_hull = cv2.minAreaRect(sample_hulls[0])
        w_hull = rect_hull[1][0]
        h_hull = rect_hull[1][1]
        dict_sample_symbol_contours["hull_w"] = round(w_hull, 2)
        dict_sample_symbol_contours["hull_h"] = round(h_hull, 2)
        dict_sample_symbol_contours["hull_area"] = round(area_hull, 2)
        dict_sample_symbol_contours["hull_contours"] = sample_contours

        if (
                (image_reference.shape[1] / image_reference.shape[0] > 3) or (image_reference.shape[0] // image_reference.shape[1] > 3)
            ) or (
                (205 > image_reference_preprocessed.shape[1] > 190) and (200 > image_reference_preprocessed.shape[0] > 180) # beko -> o #
            ):

            # image_reference_preprocessed = cv2.copyMakeBorder(image_reference_preprocessed, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # image_sample_preprocessed = cv2.copyMakeBorder(image_sample_preprocessed, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # ref_contours, ref_hulls = get_Contours_and_Hull(image_reference_preprocessed.copy())
            # sample_contours, sample_hulls = get_Contours_and_Hull(image_sample_preprocessed.copy())


            img_bbox_ref, ref_hulls, ref_img_box = draw_Hulls_and_Bbox(image_reference_preprocessed.copy(), ref_contours)
            img_bbox_sample, sample_hulls, sample_img_bbox = draw_Hulls_and_Bbox(image_sample_preprocessed.copy(), sample_contours)

            if ref_img_box is not None and sample_img_bbox is not None:
                M = cv2.getPerspectiveTransform(np.float32(sample_img_bbox), np.float32(ref_img_box))
                warped_sample = cv2.warpPerspective(image_sample_preprocessed.copy(), M, (image_sample_preprocessed.shape[1], image_sample_preprocessed.shape[0]))
                warped_sample = threshold(warped_sample, configs=[100, 255, cv2.THRESH_BINARY])
            else:
                warped_sample = image_sample_preprocessed.copy()

            # img_bbox_ref = img_bbox_ref[40:-40, 40:-40]
            # img_bbox_sample = img_bbox_sample[40:-40, 40:-40]
            # warped_sample = warped_sample[40:-40, 40:-40]
            # image_reference_preprocessed = image_reference_preprocessed[40:-40, 40:-40]

            # combine_img = cv2.addWeighted(warped_sample, 0.6, image_reference_preprocessed , 1, 0)
            # combine_img_ref = cv2.addWeighted(image_sample_preprocessed, 0.6, image_reference_preprocessed , 1, 0)
            image_sample_preprocessed = warped_sample
        else: # beko -> e #
            img_bbox_ref = image_reference_preprocessed.copy()
            img_bbox_sample = image_sample_preprocessed.copy()
        ##########
    # stop_preprocess = time.time() - start_preprocess

    start_diff = time.time()
    # SUBTRACTING
    if method == 1:
        image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid_contour = contour_Centroids(contours)

    if method == 5:
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)

        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        #dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_CLOSE, kernel, iterations = 1)
        #erode = cv2.erode(dilate, kernel, iterations = 1) # 2


        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=ratio#70,
        )

        ropw_color = cv2.cvtColor(image_subtracted_prc_rso.copy(), cv2.COLOR_GRAY2BGR)
        ropw_color = remove_One_Pixel_Width(ropw_color.copy(), contours, counter, is_color=True)
        ropw = remove_One_Pixel_Width(image_subtracted_prc_rso.copy(), contours, counter)

        dilate = image_subtracted_prc
        all_contour_area = np.array(all_contour_area)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
        if all_contour_area[all_contour_area > decision_ratio].any():
            dilate = cv2.dilate(image_subtracted_prc, kernel, iterations=dilate_iteration)
            erode = cv2.erode(dilate, kernel, iterations=erode_iteration)

            # print(counter, ": ", all_contour_area[all_contour_area > 5])

        else:
            erode = cv2.erode(image_subtracted_prc, kernel, iterations=5)

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            ropw.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=ratio#70,
        )

        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso

        centroid_contour = contour_Centroids(not_removed_buffer)

        stdo(1, "{}:{}".format(counter, all_contour_area))

        # image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso

        # centroid_contour = contour_Centroids(not_removed_buffer)

        # show_pack = [image_subtracted_prc, image_subtracted_prc_rso]
        # show_image(show_pack, open_order=1, window=window)


        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, ropw, ropw_color, dilate, erode, image_subtracted_prc_rso]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc", str(counter)+"_6ropw", str(counter)+"_6.1ropw_color", str(counter)+"_7dilate", str(counter)+"_8erode", str(counter)+"_9rso"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-5", filename=filename, format="png")

        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 4:
        # image_subtracted_prc_ref = image_reference_preprocessed - image_sample_preprocessed
        # image_subtracted_prc_sample = image_sample_preprocessed - image_reference_preprocessed
        # image_subtracted_prc_ref_sample = image_subtracted_prc_ref | image_subtracted_prc_sample
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        # image_subtracted_prc_total = image_subtracted_prc_ref | image_subtracted_prc_sample

        # show_image(image_subtracted_prc, open_order=1)

        # show_image([image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc], open_order=1)

        # pure = sword_of_justice(image_subtracted_prc_total, counter)

        """image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc_sample.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio #100, #70
        )"""

        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1)  # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_CLOSE, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)  # 2

        image_subtracted_prc_rso, contours, _, _, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio  # 100, #70
        )

        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations=1)

        image_sample_masked_gray_removed_small_objects = res

        centroid_contour = contour_Centroids(not_removed_buffer)

        ##############################
        # Save Temp Images For Debug #
        ##############################
        # """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, dilate, erode, image_subtracted_prc_rso, res]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_total", str(counter)+"_6dilate", str(counter)+"_7erode" , str(counter)+"_8rso",  str(counter)+"_9res"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-4", filename=filename, format="png")
        # """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 6:
        image_reference_preprocessed = preprocessing_black(image_reference, threshold_config=threshold_config_reference, counter=counter+1)
        image_sample_preprocessed = preprocessing_black(image_sample, threshold_config=threshold_config_sample, counter=counter+1)

        # #########################################################3
        # #########################################################3
        """
        image_sample_preprocessed_prunned = pruning(image_sample_preprocessed, method=2)
        image_reference_preprocessed_prunned = pruning(image_reference_preprocessed, method=2)
        show_pack = [
            image_reference_preprocessed, image_reference_preprocessed_prunned,
            image_sample_preprocessed, image_sample_preprocessed_prunned
        ]
        show_image( show_pack , open_order=2)

        image_reference_preprocessed = image_reference_preprocessed_prunned.copy()
        image_sample_preprocessed = image_sample_preprocessed_prunned.copy()
        """
        # #########################################################3
        # #########################################################3

        image_subtracted_prc_negative = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_positive = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)

        image_subtracted_prc = image_subtracted_prc_positive | image_subtracted_prc_negative
        # image_subtracted_prc_or = image_subtracted_prc.copy()

        # show_image( image_subtracted_prc, title="before rso", open_order=1)

        if type(rso_ratio) != int:
            rso_ratio1 = rso_ratio[0]
            rso_ratio2 = rso_ratio[1]
        else:
            rso_ratio1 = rso_ratio
            rso_ratio2 = 2

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio1  # 5,
        )
        # __after_rso = image_subtracted_prc.copy()

        # image_subtracted_prc = cv2.subtract(image_subtracted_prc_rso_negative, image_subtracted_prc_rso_positive)
        # image_subtracted_prc_rso_temp = image_subtracted_prc_rso.copy()

        # #########################################################3
        # #########################################################3
        # #########################################################3

        if image_subtracted_prc_rso.any() == 1:
            kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
            # kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12) )
            image_subtracted_prc_rso = cv2.morphologyEx(image_subtracted_prc_rso, cv2.MORPH_ERODE, kernel_opening)
            image_subtracted_prc_rso = cv2.morphologyEx(image_subtracted_prc_rso, cv2.MORPH_DILATE, kernel_opening)
            # show_image( [__after_rso, image_subtracted_prc], title=["after rso", "after MORPH_CLOSE"] , open_order=2)

        image_subtracted_prc_rso_2, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc_rso.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio2,
        )

        # #########################################################3
        # #########################################################3
        # #########################################################3
        # #########################################################3

        # show_image([image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_positive, image_subtracted_prc_rso_positive, image_subtracted_prc_negative, image_subtracted_prc_rso_negative, image_subtracted_prc, image_subtracted_prc_rso], title='S', open_order=2)

        # #############################
        # Save Temp Images For Debug #
        # #############################
        # list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_positive, image_subtracted_prc_negative, image_subtracted_prc_or, image_subtracted_prc_rso, image_subtracted_prc_rso_2]
        # filename = ["image_reference" + str(counter), "image_sample" + str(counter), "image_reference_preprocessed" + str(counter), "image_sample_preprocessed" + str(counter), "image_subtracted_prc_positive" + str(counter), "image_subtracted_prc_negative" + str(counter), "image_subtracted_prc_or" + str(counter), "image_subtracted_prc_rso" + str(counter), "image_subtracted_prc_rso_2" + str(counter)]
        # save_image(list_temp_save_image, path="temp_files/extractor_difference/method-6", filename=filename, format="png")

        """
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-1ref.png", image_reference)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-2sample.png", image_sample)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-3ref_pro.png", image_reference_preprocessed)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-4sample_pro.png", image_sample_preprocessed)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-5prc_pos.png", image_subtracted_prc_positive)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-6prc_neg.png", image_subtracted_prc_negative)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-7prc_or.png", image_subtracted_prc_or)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-8prc_rso.png", image_subtracted_prc_rso)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-9prc_rso2.png", image_subtracted_prc_rso_2)
        """

        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso

        centroid_contour = contour_Centroids(not_removed_buffer)

        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_positive, image_subtracted_prc_negative, image_subtracted_prc_or, image_subtracted_prc_rso, image_subtracted_prc_rso_2]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_pos", str(counter)+"_6prc_neg",  str(counter)+"_7prc_or", str(counter)+"_8prc_rso", str(counter)+"_9prc_rso_2"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-6/", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 7:
        image_reference_preprocessed = preprocessing_4(image_reference, threshold_config_reference)
        image_sample_preprocessed = preprocessing_4(image_sample, threshold_config_sample)

        #image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_CLOSE, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)


        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio
        )
        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations=1)

        image_sample_masked_gray_removed_small_objects = res
        # contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid_contour = contour_Centroids(not_removed_buffer)

        # show_pack = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, image_subtracted_prc_rso]
        # show_image(show_pack, open_order=2, window=True)
        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, dilate, erode, image_subtracted_prc_rso, res]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_total", str(counter)+"_6dilate", str(counter)+"_7erode" , str(counter)+"_8rso",  str(counter)+"_9res"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-7", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 8:
        # image_reference_preprocessed = preprocessing_4(image_reference, threshold_config)
        # image_sample_preprocessed = preprocessing_4(image_sample, threshold_config)

        # image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2) )
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_OPEN, kernel, iterations = 1)
        erode = cv2.erode(dilate, kernel, iterations = 1)

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio
        )
        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations = 1)

        image_sample_masked_gray_removed_small_objects = res
        # contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid_contour = contour_Centroids(not_removed_buffer)

        # show_pack = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, image_subtracted_prc_rso]
        # show_image(show_pack, open_order=2, window=True)
        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, dilate, erode, image_subtracted_prc_rso, res]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_total", str(counter)+"_6dilate", str(counter)+"_7erode" , str(counter)+"_8rso",  str(counter)+"_9res"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-8", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 9:
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        image_subtracted_prc_rso, contours, _, _, _ = remove_Small_Object(
            image_subtracted_prc.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=1  # 70,
        )

        ropw_color = cv2.cvtColor(image_subtracted_prc_rso.copy(), cv2.COLOR_GRAY2BGR)
        ropw_color, _ = remove_Pixel_Width(ropw_color.copy(), contours, pixel_width=pixel_width, pixel_width_color=pixel_width_color, counter=counter, is_color=True)
        ropw, _ = remove_Pixel_Width(image_subtracted_prc_rso.copy(), contours, pixel_width=pixel_width, counter=counter)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
        dilate = cv2.dilate(ropw, kernel, iterations=dilate_iteration)
        # erode = cv2.erode(dilate, kernel, iterations=erode_iteration)

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            dilate.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=rso_ratio,  # 70,
            counter=counter
        )
        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
        centroid_contour = contour_Centroids(not_removed_buffer)

        stdo(1, "[{}]: Matched Area: {}".format(counter, all_contour_area))
        # stdo(1, "[{}]: Centroid Contour: {}".format(counter, centroid_contour))

        if activate_debug_images:
            list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, ropw_color, ropw, image_subtracted_prc_rso]
            filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc", str(counter)+"_6ropw_color", str(counter)+"_7ropw", str(counter)+"_8rso"]
            save_image(list_temp_save_image, path="temp_files/extractor_difference/method-"+str(method), filename=filename, format="png")

            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 10:
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        image_subtracted_prc_rso, contours, _, _, _ = remove_Small_Object(
            image_subtracted_prc.copy(),
            # is_contour_number_for_area=True,
            # is_filter=is_filter,
            ratio=1#70,
        )

        median_blured = cv2.medianBlur(image_subtracted_prc_rso, 3, 3)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
        dilate = cv2.dilate(median_blured, kernel, iterations=dilate_iteration)
        #erode = cv2.erode(dilate, kernel, iterations=erode_iteration)

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            dilate.copy(),
            #is_contour_number_for_area=True,
            #is_filter=is_filter,
            ratio=rso_ratio,#70,
            counter=counter
        )
        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
        centroid_contour = contour_Centroids(not_removed_buffer)

        stdo(1, "[{}]: Matched Area: {}".format(counter, all_contour_area))
        #stdo(1, "{}:{}".format(counter, not_removed_buffer))

        if activate_debug_images:
            list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, median_blured, dilate, image_subtracted_prc_rso]
            filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc", str(counter)+"_6median_blured", str(counter)+"_7dilate", str(counter)+"_8rso"]
            save_image(list_temp_save_image, path="temp_files/extractor_difference/method-"+str(method), filename=filename, format="png")

            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 11:

        start_subtract = time.time()

        if (
            (205 > image_reference_preprocessed.shape[1] > 190) and (200 > image_reference_preprocessed.shape[0] > 180) # beko -> o #
        ) or (
            (image_reference_preprocessed.shape[0] > 200 and image_reference_preprocessed.shape[1] > 180) # beko -> bk #
        ):
            contours, _ = cv2.findContours(image_reference_preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image_reference_preprocessed)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            image_reference_preprocessed = cv2.bitwise_and(image_reference_preprocessed, mask)

            contours, _ = cv2.findContours(image_sample_preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image_sample_preprocessed)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            image_sample_preprocessed = cv2.bitwise_and(image_sample_preprocessed, mask)

        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)  # Close 20.07.2025
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)  # Close 20.07.2025
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)  # Close 20.07.2025
        # image_subtracted_prc = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)  # 20.07.2025

        # ref_gpu = cp.asarray(image_reference_preprocessed)
        # sample_gpu = cp.asarray(image_sample_preprocessed)
        # diff1 = cp.clip(ref_gpu.astype(cp.int16) - sample_gpu.astype(cp.int16), 0, 255)
        # diff2 = cp.clip(sample_gpu.astype(cp.int16) - ref_gpu.astype(cp.int16), 0, 255)
        # image_subtracted_prc = cp.clip(diff1 + diff2, 0, 255).astype(cp.uint8)

        image_subtracted_prc_rso, contours, _, _, _ = remove_Small_Object(
            image_subtracted_prc.copy(),
            ratio=1
        )

        if (
            (205 > image_reference_preprocessed.shape[1] > 190) and (200 > image_reference_preprocessed.shape[0] > 180) # beko -> o #
        ) or (
            (image_reference_preprocessed.shape[0] > 200 and image_reference_preprocessed.shape[1] > 180) # beko -> bk #
        ):
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,1))
            opening = cv2.morphologyEx(image_subtracted_prc_rso.copy(), cv2.MORPH_OPEN, kernel)
            _, opening = cv2.threshold(opening, 0, 255, cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_subtracted_prc_rso = opening

        # stop_subtract = time.time() - start_subtract

        start_rpw = time.time()
        rpw_ratio_px = pixel_width * 20.244 # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025

        if (200 > image_reference_preprocessed.shape[0] > 180) and (200 > image_reference_preprocessed.shape[1] > 170): # beko -> eo
            rpw_method = 7
        else:
            rpw_method = 6

        if activate_debug_images:
            ropw_color, _ = remove_Pixel_Width(image_subtracted_prc_rso.copy(), contours, pixel_width=rpw_ratio_px, pixel_width_color=pixel_width_color, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls, is_color=True)
        ropw, dict_rpw_contours = remove_Pixel_Width(image_subtracted_prc_rso.copy(), contours, pixel_width=rpw_ratio_px, method=rpw_method, counter=counter, pano_sector_index=pano_sector, symbol_hull_list=sample_hulls)
        stop_rpw = time.time() - start_rpw

        start_dilate = time.time()
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
        dilate = cv2.dilate(ropw, kernel, iterations=dilate_iteration) # Open 20.07.2025
        stop_dilate = time.time() - start_dilate

        start_rso = time.time()
        rso_ratio_px = rso_ratio * 20.244  # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            dilate.copy(), # 20.07.2025 # ropw.copy
            ratio=int(rso_ratio_px*rso_ratio_px),
            counter=counter
        )
        # stop_rso = time.time() - start_rso

        start_cc = time.time()
        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
        centroid_contour = contour_Centroids(not_removed_buffer)
        # stop_cc = time.time() - start_cc

        start_mse = time.time()
        #########
        if (image_reference.shape[1] / image_reference.shape[0] > 5.5) or ((image_reference.shape[0] // image_reference.shape[1] > 5.5)):

            if centroid_contour:

                ref_gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
                sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY).astype(np.float64)

                mse_noise = mean_squared_error(ref_gray, sample_gray)
                if activate_debug_images:
                    stdo(1, "[{}]: mse_noise: {}".format(counter, mse_noise))

                if mse_noise < 400:
                    contours = []
                    not_removed_buffer = []
                    centroid_contour = []
        #########
        # stop_mse = time.time() - start_mse

        start_secondary_decision = time.time()
        ##########
        # if (centroid_contour) or ((image_reference.shape[1] / image_reference.shape[0] > 5.5) or (image_reference.shape[0] // image_reference.shape[1] > 5.5)):

        #     ref_count = np.sum(image_reference_preprocessed == 255)
        #     sample_count = np.sum(image_sample_preprocessed == 255)
        #     count_diff = abs(ref_count-sample_count)

        #     ref_gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
        #     sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY).astype(np.float64)

        #     me_noise = math.sqrt(mean_squared_error(ref_gray, sample_gray))
        #     hypotenuse_coeff = math.sqrt(math.pow(image_reference.shape[1],2) + math.pow(image_reference.shape[0],2))
        #     rpw_hypo_coeff = hypotenuse_coeff * (rpw_ratio_px * 2)
        #     rpw_count_diff = count_diff - (hypotenuse_coeff * (rpw_ratio_px * 2))
        #     last_diff = rpw_hypo_coeff - count_diff

        #     ssim = structural_similarity(ref_gray, sample_gray, data_range=sample_gray.max()-sample_gray.min())
        #     rpw_ssim = ssim + pixel_width

        #     me_rpw_count_diff = rpw_count_diff - me_noise

        #     if activate_debug_images:
        #         stdo(1, "[{}]: | hypotenuse_coeff:{:.2f} | count_diff:{:.2f},rpw_count_diff:{:.2f} - me_noise: {:.2f} = me_rpw_count_diff:{:.2f} | ssim:{:.2f},rpw_ssim:{:.2f}".format(
        #             counter, hypotenuse_coeff, count_diff, rpw_count_diff, me_noise, me_rpw_count_diff, ssim, rpw_ssim,
        #         ))

        #     if ((me_rpw_count_diff < 20) and (rpw_ssim > 0.88)):

        #         if not (image_reference.shape[1] / image_reference.shape[0] > 5.5) or not (image_reference.shape[0] // image_reference.shape[1] > 5.5):
        #             contours = []
        #             not_removed_buffer = []
        #             centroid_contour = []

        filtered = image_subtracted_prc_rso.copy()

        if (centroid_contour):
            ref_gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY).astype(np.float64)
            ssim = structural_similarity(ref_gray, sample_gray, data_range=sample_gray.max()-sample_gray.min())

            ref_count = np.sum(image_reference_preprocessed == 255)
            sample_count = np.sum(image_sample_preprocessed == 255)
            count_diff = abs(ref_count-sample_count)

            hypotenuse_coeff = math.sqrt(math.pow(image_reference.shape[1],2) + math.pow(image_reference.shape[0],2)) * 2

            if (ssim > 0.98): #and not (image_reference.shape[0] > 200 and image_reference.shape[1] > 180): #and (count_diff < hypotenuse_coeff):

                if (image_reference.shape[1] / image_reference.shape[0] > 4) or ((image_reference.shape[0] / image_reference.shape[1] > 4)):
                    pass

                else:
                    contours = []
                    not_removed_buffer = []
                    centroid_contour = []


            if activate_debug_images:
                stdo(1, "[{}]: ssim:{:.2f} | count_diff:{} | hypotenuse_coeff*2:{:.1f} | Matched Area: {}".format(counter, ssim, count_diff, hypotenuse_coeff, all_contour_area))

            filtered = image_subtracted_prc_rso.copy()

        else:

            if (image_reference.shape[1] / image_reference.shape[0] > 10) or ((image_reference.shape[0] / image_reference.shape[1] > 10)):

                dist_transform = cv2.distanceTransform(image_subtracted_prc.copy(), cv2.DIST_L2, 3)

                # 3. Kalınlığı < 2 piksel olan yerleri bul
                thin_mask = (dist_transform * 2) <= 2  # bool: True = silinecek piksel

                # 4. Bu yerleri 0 yap (siyah)
                filtered = image_subtracted_prc.copy()
                filtered[thin_mask] = 0

                dilate = cv2.dilate(filtered, kernel, iterations=dilate_iteration)

                image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
                    dilate.copy(),
                    ratio=10,
                    counter=counter
                )

                image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
                centroid_contour = contour_Centroids(not_removed_buffer)

                ref_gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
                sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY).astype(np.float64)
                ssim = structural_similarity(ref_gray, sample_gray, data_range=sample_gray.max()-sample_gray.min())

                if (ssim > 0.98):
                    contours = []
                    not_removed_buffer = []
                    centroid_contour = []

                if activate_debug_images:
                    stdo(1, "[{}]: distanceTransform-lines - ssim:{}".format(counter, ssim))

            # elif (image_reference.shape[0] > 170 and image_reference.shape[1] > 160):

            #     dist_transform = cv2.distanceTransform(image_subtracted_prc.copy(), cv2.DIST_L2, 3)

            #     # 3. Kalınlığı < 2 piksel olan yerleri bul
            #     thin_mask = (dist_transform * 2) <= 2  # bool: True = silinecek piksel

            #     # 4. Bu yerleri 0 yap (siyah)
            #     filtered = image_subtracted_prc.copy()
            #     filtered[thin_mask] = 0

            #     dilate = cv2.dilate(filtered, kernel, iterations=dilate_iteration)

            #     image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            #         dilate.copy(), # 20.07.2025 # ropw.copy
            #         ratio=1,
            #         counter=counter
            #     )

            #     image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
            #     centroid_contour = contour_Centroids(not_removed_buffer)

            #     ref_gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
            #     sample_gray = cv2.cvtColor(image_sample, cv2.COLOR_BGR2GRAY).astype(np.float64)
            #     ssim = structural_similarity(ref_gray, sample_gray, data_range=sample_gray.max()-sample_gray.min())

            #     if (ssim > 0.98):
            #         contours = []
            #         not_removed_buffer = []
            #         centroid_contour = []

            #     if activate_debug_images:
            #         stdo(1, "[{}]: distanceTransform-beko_word - ssim:{}".format(counter, ssim))

        draw_color = None
        rpw_elements = None
        if centroid_contour:
            draw_color = image_sample.copy()
            if (object_color == 'white') or (object_color == 'gray - siyah sembol'):
                color = [255,255,255]
            else:
                color = [0,0,0]
            draw_color = cv2.copyMakeBorder(draw_color, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)

            for cnt in dict_rpw_contours:
                list_cnt = cnt["actual_faults"]
                if list_cnt:
                    list_cnt = [(x + 10, y + 10) for (x, y) in list_cnt]
                    contour_np = np.array(list_cnt, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.drawContours(draw_color, [contour_np], -1, color=[0,0,255], thickness=cv2.FILLED)
            rpw_elements = dict_rpw_contours

        ##########
        stop_secondary_decision = time.time() - start_secondary_decision


        # if activate_debug_images:
        #     stdo(1, "[{}][{}] T:{:.5f} | sub: {:.5f} | rpw: {:.5f} | dil:{:.5f} | rso: {:.5f} | cc: {:.5f} | mse: {:.5f} | sd:{:.5f}".format(
        #             "Diff",
        #             counter,
        #             stop_subtract+stop_rpw+stop_dilate+stop_rso+stop_cc+stop_mse,
        #             stop_subtract,
        #             stop_rpw,
        #             stop_dilate,
        #             stop_rso,
        #             stop_cc,
        #             stop_mse,
        #             stop_secondary_decision
        #         )
        #     )


        if activate_debug_images:

            stdo(1, "[{}]: Matched Area: {}".format(counter, all_contour_area))

            list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, img_bbox_ref, img_bbox_sample, image_subtracted_prc, ropw_color, ropw, image_subtracted_prc_rso, filtered]
            filename = [
                str(counter)+"_1ref",
                str(counter)+"_2sample",
                str(counter)+"_3ref_pre",
                str(counter)+"_4sample_pre",
                str(counter)+"_5ref_bbox",
                str(counter)+"_6sample_bbox",
                str(counter)+"_7prc",
                str(counter)+"_8ropw_color",
                str(counter)+"_9ropw",
                str(counter)+"_10rso",
                str(counter)+"_11filtered"

            ]
            save_image(list_temp_save_image, path="temp_files/extractor_difference/method-"+str(method), filename=filename, format="png")

            # save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="jpg")
            # save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="jpg")
            # save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="jpg")
            # save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="jpg")
            # save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="jpg")
            # save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="jpg")

    stop_diff = time.time() - start_diff

    stop_seq = time.time() - start_seq
    # if activate_debug_images:
    #     stdo(1, "[{}][{}] T:{:.5f} | pre-P: {:.5f} | Diff: {:.5f}".format(
    #             "ED",
    #             counter,
    #             stop_seq,
    #             stop_preprocess,
    #             stop_diff,
    #         )
    #     )

    return image_sample_masked_gray_removed_small_objects, image_sample, contours, not_removed_buffer, centroid_contour, image_subtracted_prc, draw_color, rpw_elements, dict_sample_symbol_contours


def excessive_1(src):
    src_gray = cv2.cvtColor(src.copy(), cv2.COLOR_RGB2GRAY)
    src_gray_gaus = cv2.GaussianBlur(src_gray.copy(), (3, 3), 0)
    src_gray_gaus_lap = cv2.Laplacian(src_gray_gaus - src_gray, cv2.CV_64F, ksize=5, delta=0.5)

    show_image(
        (src, src_gray_gaus, src_gray_gaus_lap),
        title=["src", "src_gray_gaus", "src_gray_gaus_lap"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=1,
    )

    exit()
    # import pdb


def excessive_2(src, smpl, sharp_scale=10, is_smooth=False, smooth_scale=0, is_sharp=False):
    from image_manipulation import median_blur

    # SHARPING
    src_sharpened = src.copy()
    smpl_sharpened = smpl.copy()

    for _ in range(sharp_scale):
        src_sharpened = sharpening(src_sharpened)
        smpl_sharpened = sharpening(smpl_sharpened)
        if is_smooth:
            configs = []
            src_sharpened = median_blur(src_sharpened, configs=configs)
            smpl_sharpened = median_blur(smpl_sharpened, configs=configs)

    # SMOOTHING
    src_smoothed = src_sharpened.copy()
    smpl_smoothed = smpl_sharpened.copy()
    configs = []
    for _ in range(smooth_scale):
        src_smoothed = median_blur(src_smoothed, configs=configs)
        smpl_smoothed = median_blur(smpl_smoothed, configs=configs)
        if is_sharp:
            src_smoothed = sharpening(src_smoothed)
            smpl_smoothed = sharpening(smpl_smoothed)

    # src_gray = cv2.cvtColor(src_smoothed.copy(), cv2.COLOR_RGB2GRAY)
    # smpl_gray = cv2.cvtColor(smpl_smoothed.copy(), cv2.COLOR_RGB2GRAY)

    # src_gray_gaus = cv2.GaussianBlur(src.copy(), (3, 3), 0)
    # src_gray_gaus_lap = cv2.Laplacian(src_gray_gaus - src_gray, cv2.CV_64F, ksize=5, delta=0.5)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(18,18))
    src_gray_cl = clahe.apply(src_smoothed)
    smpl_gray_cl = clahe.apply(smpl_smoothed)

    show_image(
        (src_sharpened, smpl_sharpened, src_gray_cl, smpl_gray_cl),
        #title=["src_sharpened", "smpl_sharpened", "src_sharpened_gray_cl", "smpl_sharpened_gray_cl"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=2,
    )

    return src_gray_cl, smpl_gray_cl
    # import pdb


def excessive_3_sobel_gradient(src, smpl, sharp_scale=10, is_smooth=False, smooth_scale=0, is_sharp=False):
    from image_manipulation import median_blur

    # SHARPING
    src_sharpened = src.copy()
    smpl_sharpened = smpl.copy()

    import pdb; pdb.set_trace();
    for _ in range(sharp_scale):
        src_sharpened = sharpening(src_sharpened, kernel_size=(5, 5), alpha=1.5, beta=-0.7, gamma=0, over_run = 0)
        smpl_sharpened = sharpening(smpl_sharpened, kernel_size=(5, 5), alpha=1.5, beta=-0.7, gamma=0, over_run = 0)
        if is_smooth:
            configs = []
            src_sharpened = median_blur(src_sharpened, configs=configs)
            smpl_sharpened = median_blur(smpl_sharpened, configs=configs)

    # SMOOTHING
    src_smoothed = src_sharpened.copy()
    smpl_smoothed = smpl_sharpened.copy()
    configs = []
    for _ in range(smooth_scale):
        src_smoothed = median_blur(src_smoothed, configs=configs)
        smpl_smoothed = median_blur(smpl_smoothed, configs=configs)
        if is_sharp:
            src_smoothed = sharpening(src_smoothed)
            smpl_smoothed = sharpening(smpl_smoothed)

    """
    # TEST FOR SOBEL
    ksize = 15
    sobelx_64F = cv2.Sobel(smpl.copy(), cv2.CV_64F, 1, 0, ksize=ksize)
    sobely_64F = cv2.Sobel(smpl.copy(), cv2.CV_64F, 0, 1, ksize=ksize)
    sobelx_8U = cv2.Sobel(smpl.copy(), cv2.CV_8U, 1, 0, ksize=ksize)
    sobely_8U = cv2.Sobel(smpl.copy(), cv2.CV_8U, 0, 1, ksize=ksize)
    show_image((sobelx_64F, sobely_64F, sobelx_8U, sobely_8U), open_order=1, window=True)
    show_image(
        (
            sobelx_64F * sobelx_8U,
            sobely_64F * sobely_8U,
            sobelx_64F + sobely_64F + sobelx_8U + sobely_8U,
            (sobelx_64F + sobely_64F) - (sobelx_8U + sobely_8U),
            (sobelx_64F + sobely_64F) * (sobelx_8U + sobely_8U),
            (sobelx_8U + sobely_8U) ** (sobelx_64F + sobely_64F)
        ),
        open_order=1,
        window=True
    )
    # import pdb; pdb.set_trace();
    # ksize = 1; sobely = cv2.Sobel(smpl.copy(), cv2.CV_64F, 0, 1, ksize=ksize); sobelx = cv2.Sobel(smpl.copy(), cv2.CV_64F, 1, 0, ksize=ksize);
    # ksize = 1; sobely = cv2.Sobel(smpl.copy(), cv2.CV_64F, 0, 1, ksize=ksize); sobelx = cv2.Sobel(smpl.copy(), cv2.CV_64F, 1, 0, ksize=ksize);
    # show_image((sobelx, sobely, sobelx+sobely, sobelx-sobely, sobelx*sobely), open_order=1, window=True)
    """

    # src_gray = cv2.cvtColor(src_smoothed.copy(), cv2.COLOR_RGB2GRAY)
    # smpl_gray = cv2.cvtColor(smpl_smoothed.copy(), cv2.COLOR_RGB2GRAY)

    # src_gray_gaus = cv2.GaussianBlur(src.copy(), (3, 3), 0)
    # src_gray_gaus_lap = cv2.Laplacian(src_gray_gaus - src_gray, cv2.CV_64F, ksize=5, delta=0.5)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=18.0, tileGridSize=(18,18))
    src_gray_cl = clahe.apply(src_smoothed)
    smpl_gray_cl = clahe.apply(smpl_smoothed)

    show_image(
        (src_sharpened, smpl_sharpened, src_gray_cl, smpl_gray_cl),
        #title=["src_sharpened", "smpl_sharpened", "src_sharpened_gray_cl", "smpl_sharpened_gray_cl"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=2,
    )

    return src_gray_cl, smpl_gray_cl
    # import pdb


def excessive_4_preprocessing(image, is_sharp=True, is_sobel=True, is_threshold=True, window=False, sharp_scale=2, open_order=1, sobel_scale=0.05, threshold_config=[-1, -1]):
    #BLUR
    from image_manipulation import median_blur
    blur = median_blur(image, configs=[3])

    # SHARP
    if is_sharp:
        image_sharpened = blur
        for _ in range(sharp_scale):
            image_sharpened = sharpening(image_sharpened)
    else:
        image_sharpened = image.copy()

    # SOBEL
    if is_sobel:
        image_sharpened_sobel = sobel_gradient(image_sharpened, sobel_scale)
    else:
        image_sharpened_sobel = image_sharpened.copy()

    # THRESHOLD
    if is_threshold:
        image_sharpened_sobel_threshold = threshold(
            image_sharpened_sobel.copy(), configs=[-1,3])
    else:
        image_sharpened_sobel_threshold = image_sharpened_sobel.copy()


    # MORPH
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    openinig = cv2.morphologyEx(image_sharpened_sobel_threshold, cv2.MORPH_OPEN, kernel_opening)

    #kernel_erode = np.ones((5,5), np.uint8)
    #erosion = cv2.erode(openinig, kernel_erode, iterations=1)

    #kernel_opening2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    #openinig2 = cv2.morphologyEx(openinig, cv2.MORPH_OPEN, kernel_opening2)
    rso = remove_Small_Object(openinig.copy(), ratio=10)[0]

    # FILLED
    """filled = closing.copy()
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in (contours):
        cv2.fillPoly(filled, pts=[c], color=(255,255,255))
    #combine = cv2.bitwise_and(closing, filled)
    """
    #fill = image_sharpened_sobel.copy()
    #h, w = image.shape[:2]
    #mask = np.zeros((h+2, w+2), np.uint8)
    #cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_MASK_ONLY)
    #fill_not = cv2.bitwise_not(fill)
    #combine = image_sharpened_sobel | fill

    return image, blur, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold, openinig, rso

def excessive_4(src, smpl):
    ref = excessive_4_preprocessing(src.copy(), sharp_scale=2, sobel_scale=1, window=True)
    sample = excessive_4_preprocessing(smpl.copy(),  sharp_scale=2, sobel_scale=1, window=True)

    show_pack = [*ref, *sample]
    show_image(show_pack, open_order=2, window=True)
    return ref[-1], sample[-1]


def main():
    global arguments

    arguments = argument_control()
    argument_info()

    if arguments["test_windows"]:
        window = True
    else:
        window = False

    # To log the time passed, initialize the time logger
    time_log(id="background_substraction")
    stdo(1, "Program Started.")

    #########
    # START #
    #########
    # OPEN IMAGES

    ###################
    image_reference = open_image(arguments["reference"], option = "cv2-rgb", is_numpy=True)
    image_test = open_image(arguments["input"], option = "cv2-rgb", is_numpy=True)

    image_reference_gray = cv2.cvtColor(image_reference.copy(), cv2.COLOR_RGB2GRAY)
    image_sample_gray= cv2.cvtColor(image_test.copy(), cv2.COLOR_RGB2GRAY)


    ref, sample = excessive_4(image_reference_gray, image_sample_gray)


    # SUBTRACTING
    image_subtracted_prc = ref - sample

    # return remove_Small_Object(image_subtracted_prc.copy(), ratio=ratio, buffer_percentage=buffer_percentage, is_filter=is_filter, filter_lower_ratio=filter_lower_ratio)

    # MASKING
    image_sample_masked_prc = cv2.bitwise_or(image_sample_gray, image_sample_gray, mask=image_subtracted_prc)

    image_sample_masked_gray_removed_small_objects, \
    contours, \
    all_contours_area, \
    will_be_removed_buffer, \
    not_removed_buffer = remove_Small_Object(
        image_subtracted_prc.copy(),
        ratio=0,
        buffer_percentage=40,
        is_filter=True,
        is_contour_number_for_area=True,
        filter_lower_ratio=40,
        filter_upper_ratio=50
    )

    #print("will_be_removed_buffer", will_be_removed_buffer)
    #print("not_removed_buffer", not_removed_buffer)
    #print("contours", contours)

    #from image_manipulation import erosion
    #kernel = [[0,0,0],
    #          [0,1,0],
    #          [1,1,0]]
    #erode = erosion(image_sample_masked_prc, kernel)
    #erode = erosion(erode, kernel)

    """th = threshold(image_sample_masked_prc.copy(), configs=[1,255])
    th_inv = cv2.bitwise_not(th)
    #masked_img = cv2.bitwise_and(th, th, mask=horizontal_inv)
    #masked_img_inv = cv2.bitwise_not(masked_img)
    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])
    erode = cv2.filter2D(th_inv, -1, kernel, cv2.BORDER_DEFAULT)
    erode = cv2.bitwise_not(erode)"""
    #image_sample_masked_gray_removed_small_objects, counters, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(image_sample_masked_prc.copy(), ratio=1, buffer_percentage=95, is_filter=False, filter_lower_ratio=10)


    draw_diff = True
    draw_diff_circle = True
    fill_color = [0,0,255]
    if draw_diff:
        image_test_draw = image_test.copy()
        image_test_draw[image_sample_masked_gray_removed_small_objects != 0] = fill_color

        #image_test_draw[erode != 0] = fill_color
    else:
        image_test_draw = None

    """
    for _, cnt in enumerate(contours):
        if cv2.contourArea(cnt) in not_removed_buffer:
            cv2.drawContours(image_test, [cnt], 0, [255, 0, 0], -1)
    """

    """print("not_removed_buffer:", not_removed_buffer)
    if draw_diff_circle:
        for contour in contours:
            center_point = contour_Centroids([contour])[0]

            fill_color = (10, 10, 255)
            radius = 35
            thickness = 3


            draw_Circle(
                image_test,
                center_point,
                radius=radius,
                color=fill_color,
                thickness=thickness
            )


            if contour_Areas([contour])[0] != 0.0:
                if contour_Areas([contour])[0] in not_removed_buffer:

                    #cv2.circle(
                    draw_Circle(
                        image_test,
                        tuple(center_point),
                        radius=radius,
                        color=fill_color,
                        thickness=thickness
                    )"""

    show_pack = [image_reference, image_test, image_sample_masked_prc, image_test_draw]
    show_image(show_pack, open_order=2, window=True)
    return
    ###################
    ###################
    ###################


    image_reference = open_image(arguments["reference"], option = "cv2-rgb", is_numpy=True)[1000:-1000, 1300:-1400]
    image_test = open_image(arguments["input"], option = "cv2-rgb", is_numpy=True)[1000:-1000, 1300:-1400]
    image_reference = open_image(arguments["reference"], option = "cv2-rgb", is_numpy=True)
    image_test = open_image(arguments["input"], option = "cv2-rgb", is_numpy=True)
    # print("image_reference", len(image_reference))
    # print("image_test", len(image_test))

    #image_reference = image_reference[1000:-1000, 1300:-1400]
    #image_test = image_test[1000:-1000, 1300:-1400]
    #print("image_reference", len(image_reference))
    #print("image_test", len(image_test))

    # Masking
    stdo(1, "Program Started.")
    image_reference_gray = cv2.cvtColor(image_reference.copy(), cv2.COLOR_RGB2GRAY)
    image_test_gray = cv2.cvtColor(image_test.copy(), cv2.COLOR_RGB2GRAY)
    # image_subtracted_gray = cv2.cvtColor(image_subtracted, cv2.COLOR_RGB2GRAY)

    # PREPROCESS
    stdo(1, "Program Started.")
    down_table = [0, 130, 130, 130, 130]
    up_table = [150, 150, 150, 150, 255]
    smpl_sharpened_LUT = look_Up_Table(image_test, down_table=down_table, up_table=up_table)

    src_sobel_gradient = sobel_gradient(image_test, 1)

    """
    ### ### ### ### ### ### ###
    ###  Fourier Transform  ###
    ### ### ### ### ### ### ###
    import cv2 as cv
    import numpy as np

    f = np.fft.fft2(image_test_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows, cols = image_test_gray.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    show_image(
        (image_test_gray, img_back, magnitude_spectrum),
        title=["Input Image", "HPF - JET", "Result in MAG"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=1,
    )

    dft = cv.dft(np.float32(image_test_gray),flags = cv.DFT_COMPLEX_OUTPUT)
    print(dft.shape)
    dft[:, -5:5] = 0
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log( cv.magnitude( dft_shift[:,:,0], dft_shift[:,:,1]) )
    #magnitude_spectrum[:, -5:5] = 0

    show_image(
        (image_test_gray, magnitude_spectrum),
        title=["Input Image", "Magnitude Spectrum"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=1,
    ); exit();
    """

    """
    show_image(
        (image_test, smpl_sharpened_LUT, src_sobel_gradient),
        #title=["src_sharpened", "smpl_sharpened", "src_sharpened_gray_cl", "smpl_sharpened_gray_cl"],
        option="plot",
        cmap="gray",
        window=True,
        open_order=2,
    ); exit();
    """
    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###

    #excessive_1(image_test)
    #image_reference_gray, image_test_gray = excessive_2(image_reference_gray, image_test_gray, sharp_scale=0, is_smooth=False, smooth_scale=0, is_sharp=False)

    # dst = cv2.equalizeHist(image_test_gray.copy()); show_image([image_test_gray.copy(), dst], open_order=1, window=True)
    # import pdb; pdb.set_trace();
    """
    image_reference_gray, image_test_gray = excessive_3_sobel_gradient(
        image_reference_gray,
        image_test_gray,
        sharp_scale=0,
        is_smooth=False,
        smooth_scale=0,
        is_sharp=False
    )
    """


    # image_test_gray = resize_to_same(image_reference_gray, image_test_gray)

    sobel_scale = 1
    image_reference_preprocessed,_,_ = preprocessing(image_reference_gray.copy(), sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window)
    image_test_preprocessed,_,_ = preprocessing(image_test_gray.copy(), sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window, threshold_config=[-1, 3])

    # SUBTRACTION
    image_subtracted_prc = image_test_preprocessed - image_reference_preprocessed

    # extractor_difference(image_reference, image_test, sobel_scale=1, ratio=0, buffer_percentage=95, is_filter=True, filter_lower_ratio=10, threshold_config=[-1, 3])

    #########
    #  END  #
    #########

    # To get the time passed, end the time logger and print the passed time
    time_log(id="background_substraction")
    stdo(
        1,
        "Time passed: {:.3f} ms".format(time_list["background_substraction"]["passed"]),
    )

    # Alpha Blending
    # image_test_alphablended = cv2.addWeighted(image_test, 0.5, image_subtracted, 0.5, 0)

    image_test_masked_prc = cv2.bitwise_or(image_test_gray, image_test_gray, mask=image_subtracted_prc)

    show_image(
        (
            image_test_gray,                image_test_gray,
            image_test_masked_prc
        ),
        title=[
            "image_test_gray",                "image_test_gray",
            "image_test_masked_prc"
            ],
        option="plot",
        cmap="gray",
        window=arguments["window"],
        open_order=2,
    )


    kernel = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
    kernel = [-1, -3, -1]


    image_test_masked_rso = image_test.copy()

    # 10 filter_lower_ratio for others
    # 150 filter_lower_ratio for inlay
    image_test_masked_gray_removed_small_objects_1, contours, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
        image_subtracted_prc.copy(), ratio=0, buffer_percentage=95, is_filter=True, filter_lower_ratio=10
    )
    image_test_masked_gray_removed_small_objects, contours, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
        image_test_masked_gray_removed_small_objects_1.copy(), ratio=0, buffer_percentage=95, is_filter=True, filter_lower_ratio=10
    )



    ### #### ###
    ### DRAW ###
    ### #### ###

    # import pdb; pdb.set_trace()
    image_test_masked_rso[image_test_masked_gray_removed_small_objects != 0] = [255, 0, 0]

    center_points = contour_Centroids(contours)
    for center_point in center_points:
        draw_Circle(
            image_test_masked_rso,
            tuple(center_point),
            radius=1,
            color=(255, 0, 0),
            thickness=-1
        )

    # import pdb; pdb.set_trace()
    # image_test_masked_gray_removed_small_objects = resize_to_same(image_test_masked_rso, image_test_masked_gray_removed_small_objects)
    image_test_masked_rso[image_test_masked_gray_removed_small_objects != 0] = [180, 10, 0]
    ### ### ###

    # image_test_masked_gray_erosion = erosion(image_subtracted_prc.copy(), kernel)
    # image_test_masked_erosion = image_test.copy()
    # image_test_masked_erosion[image_test_masked_gray_erosion != 0] = [180, 10, 0]

    #import pdb; pdb.set_trace();
    """stdo(
        1,
        "contours: {} | will_be_removed_buffer: {} | not_removed_buffer: {}".format(
            len(contours), len(will_be_removed_buffer), len(not_removed_buffer)
        ),
    )"""

    # from image_tools import open_image, show_image, save_image; show_image((debug_src, src, src - debug_src), title=["ttt", "ttt2", "ttt3"], option="plot", cmap="gray", window=True, open_order=1)
    show_image(
        (
            image_reference,                image_test,
            image_reference_preprocessed,   image_test_preprocessed,
            image_subtracted_prc,           image_test_masked_prc,
            image_test_masked_gray_removed_small_objects_1,
            image_test_masked_gray_removed_small_objects, image_test_masked_rso
        ),
        title=[
            "image_reference",              "image_test",
            "image_reference_preprocessed", "image_test_preprocessed",
            "image_subtracted_prc",         "image_test_masked_prc",
            "image_test_masked_gray_removed_small_objects_1",
            "image_test_masked_gray_erosion","image_test_masked_gray_removed_small_objects",
            "image_test_masked_erosion",    "image_test_masked_rso"],
        option="plot",
        cmap="gray",
        window=arguments["window"],
        open_order=2,
    )

    """show_image(
        (
            image_reference,                image_test,
            image_reference_preprocessed,   image_test_preprocessed,
            image_subtracted_prc,           image_test_masked_prc,
            image_test_masked_gray_erosion, image_test_masked_gray_removed_small_objects,
            image_test_masked_erosion,      image_test_masked_rso
        ),
        title=[
            "image_reference",              "image_test",
            "image_reference_preprocessed", "image_test_preprocessed",
            "image_subtracted_prc",         "image_test_masked_prc",
            "image_test_masked_gray_erosion","image_test_masked_gray_removed_small_objects",
            "image_test_masked_erosion",    "image_test_masked_rso"],
        option="plot",
        cmap="gray",
        window=arguments["window"],
        open_order=2,
    )"""

    save_image(image_test, arguments["output"])




# Program Body Trigger
if __name__ == "__main__":
    main()
