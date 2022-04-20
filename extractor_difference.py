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


from stdo import stdo
import argparse
import time
import cv2
import numpy as np
from tools import time_log, time_list, get_time  # Basic Tools
from image_tools import open_image, show_image, save_image
from image_manipulation import sharpening, sobel_gradient, erosion, threshold, remove_Small_Object, look_Up_Table, sobel_gradient, draw_Circle, contour_Centroids, contour_Areas, transparent_Draw, pruning, adjust_brightness, adjust_contrast, gamma_correction

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

def preprocessing_3(frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #sharp = sharpening(gray)
    th_not = cv2.bitwise_not(frame)
    _, th = cv2.threshold(th_not, 135, 255, cv2.THRESH_BINARY)

    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3) )
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_opening) 
    

    # rso = remove_Small_Object(closing.copy(), ratio=1000)[0]

    #show_pack = [frame, th, th_not, closing] #, rso]
    #show_image(show_pack, open_order=1)
    
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


def preprocessing(image, method=1, is_sharp=True, is_sobel=True, is_threshold=True, window=False, sharp_scale=3, open_order=1, sobel_scale=0.05, threshold_config=[-1, -1], is_middle_object=False, middle_object_roi=[], middle_object_roi_2=[], counter=0):
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

        #print("MID-ROI:", mask_cekmece.shape, middle_object_roi)
        
        """if threshold_config[2] == 'cv2.THRESH_BINARY_INV':
            color = 255
        else:
            color = 0"""

        #print("Ext-difference/preprocessing-middle_object_roi:", middle_object_roi)
        mask_cekmece[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 0
        if middle_object_roi_2:
            mask_cekmece[middle_object_roi_2[0]:middle_object_roi_2[1], middle_object_roi_2[2]:middle_object_roi_2[3]] = 0

        image_sharpened_sobel_threshold = mask_cekmece

        #show_image([image, image_sharpened_sobel_threshold, mask_cekmece], title='P', open_order=2)
        
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

    if method == 1 and is_middle_object:
        list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold, rrggbb]
        filename = ["1_image", "2_image_sharpened", "3_image_sharpened_sobel", "4_image_sharpened_sobel_threshold", "5_rrggbb_middle_object_roi"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/preprocessing", filename=filename, format="png")
    
    #list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold]
    #filename = [str(counter)+"_1_image", str(counter)+"_2_image_sharpened", str(counter)+"_3_image_sharpened_sobel", str(counter)+"_4_image_sharpened_sobel_threshold"]
    #save_image(list_temp_save_image, path="temp_files/extractor_difference/preprocessing", filename=filename, format="png")

    return image_sharpened_sobel_threshold, image_sharpened, image_sharpened_sobel


def extractor_difference(image_reference, image_sample, method=1, sobel_scale=1, sharp_scale=3, is_contour_number_for_area=True, ratio=0, buffer_percentage=95, is_filter=True, filter_lower_ratio=10, filter_upper_ratio=250, threshold_config=[-1, 3], draw_diff=True, draw_diff_circle=False, fill_color=[0, 0, 255], rso_ratio=5, window=False, bbox_invert=False, counter=0, activate_debug_images=False):
    image_subtracted_prc = -1
    centroid_contour = -1
    
    image_reference_gray = None
    image_sample_gray = None

    start_diff = time.time()
    # GRAYSCALE CONVERSION

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


    # PREPROCESSING
    image_reference_preprocessed,_,_ = preprocessing(image_reference_gray.copy(), method=method, sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window, threshold_config=threshold_config, counter=counter)
    image_sample_preprocessed,_,_ = preprocessing(image_sample_gray.copy(), method=method, sharp_scale=sharp_scale, sobel_scale=sobel_scale, window=window, threshold_config=threshold_config, counter=counter+20)
    
    if method == 5:
        image_reference_preprocessed = preprocessing_3(image_reference_gray)
        image_sample_preprocessed = preprocessing_3(image_sample_gray)
        
    if method == 4:
        """image_reference_preprocessed = preprocessing(image_reference_gray.copy(), method=method, sharp_scale=sharp_scale, is_sobel=False, is_sharp=False, window=window, threshold_config=threshold_config)
        image_sample_preprocessed = preprocessing(image_sample_gray.copy(), method=method, sharp_scale=sharp_scale, is_sobel=False, is_sharp=False, window=window, threshold_config=threshold_config)"""
        # show_image( [ image_reference_preprocessed, image_sample_preprocessed ], open_order = 1)
        
        
        image_reference_preprocessed = preprocessing_2(image_reference, threshold_config=threshold_config, bbox_invert=bbox_invert, counter=counter)[-1]
        image_sample_preprocessed = preprocessing_2(image_sample, threshold_config=threshold_config, bbox_invert=bbox_invert, counter=counter+20)[-1]


        ##############################
        # Save Temp Images For Debug #
        ##############################
        #list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed]
        #filename = ["image_reference" + str(counter), "image_sample" + str(counter), "image_reference_preprocessed" + str(counter), "image_sample_preprocessed" + str(counter)]
        #save_image(list_temp_save_image, path="temp_files/extractor_difference/method-4", filename=filename, format="png")

        """
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-1ref.png", image_reference)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-2sample.png", image_sample)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-3ref_pro.png", image_reference_preprocessed)
        cv2.imwrite("/home/alpplas/Desktop/subs/"+ str(counter) +"-4sample_pro.png", image_sample_preprocessed)
        """
        
        
    # SUBTRACTING
    image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
    contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroid_contour = contour_Centroids(contours)
    
    
    if method == 5:
        image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        
        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio#70,
        )
        
        image_sample_masked_gray_removed_small_objects = image_subtracted_prc_rso
        
        centroid_contour = contour_Centroids(not_removed_buffer)
        
        #show_pack = [image_subtracted_prc, image_subtracted_prc_rso]
        #show_image(show_pack, open_order=1, window=window)

        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, image_subtracted_prc_rso]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc", str(counter)+"_6rso"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-5", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")

    if method == 4:
        #image_subtracted_prc_ref = image_reference_preprocessed - image_sample_preprocessed
        #image_subtracted_prc_sample = image_sample_preprocessed - image_reference_preprocessed
        #image_subtracted_prc_ref_sample = image_subtracted_prc_ref | image_subtracted_prc_sample
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        #image_subtracted_prc_total = image_subtracted_prc_ref | image_subtracted_prc_sample

        #show_image(image_subtracted_prc, open_order=1)
        
        #show_image([image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc], open_order=1)
        
        #pure = sword_of_justice(image_subtracted_prc_total, counter)


        """image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc_sample.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio #100, #70
        )"""


        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2) )
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_CLOSE, kernel, iterations = 1)
        erode = cv2.erode(dilate, kernel, iterations = 1) # 2

        image_subtracted_prc_rso, contours, _, _, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio #100, #70
        )

        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations = 1)


        
        image_sample_masked_gray_removed_small_objects = res
        
        centroid_contour = contour_Centroids(not_removed_buffer)
        
        ##############################
        # Save Temp Images For Debug #
        ##############################
        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, dilate, erode, image_subtracted_prc_rso, res]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_total", str(counter)+"_6dilate", str(counter)+"_7erode" , str(counter)+"_8rso",  str(counter)+"_9res"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/method-4", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([image_subtracted_prc_rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")
        
    if method == 6:
        image_reference_preprocessed = preprocessing_black(image_reference, threshold_config=threshold_config, counter=counter+1)
        image_sample_preprocessed = preprocessing_black(image_sample, threshold_config=threshold_config, counter=counter+1)
        
        ##########################################################3
        ##########################################################3
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
        ##########################################################3
        ##########################################################3

        image_subtracted_prc_negative = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_positive  = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)

        image_subtracted_prc = image_subtracted_prc_positive | image_subtracted_prc_negative
        image_subtracted_prc_or = image_subtracted_prc.copy()

        # show_image( image_subtracted_prc, title="before rso", open_order=1) 

        if type(rso_ratio) != int:
            rso_ratio1 = rso_ratio[0]
            rso_ratio2 = rso_ratio[1]
        else:
            rso_ratio1 = rso_ratio
            rso_ratio2 = 2

        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            image_subtracted_prc.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio1 #5,
        )
        # __after_rso = image_subtracted_prc.copy()

        # image_subtracted_prc = cv2.subtract(image_subtracted_prc_rso_negative, image_subtracted_prc_rso_positive)
        image_subtracted_prc_rso_temp = image_subtracted_prc_rso.copy()

        ##########################################################3
        ##########################################################3
        ##########################################################3
        
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
        
        ##########################################################3
        ##########################################################3
        ##########################################################3
        ##########################################################3
        
        #show_image([image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_positive, image_subtracted_prc_rso_positive, image_subtracted_prc_negative, image_subtracted_prc_rso_negative, image_subtracted_prc, image_subtracted_prc_rso], title='S', open_order=2)
        
        
        ##############################
        # Save Temp Images For Debug #
        ##############################
        #list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_positive, image_subtracted_prc_negative, image_subtracted_prc_or, image_subtracted_prc_rso, image_subtracted_prc_rso_2]
        #filename = ["image_reference" + str(counter), "image_sample" + str(counter), "image_reference_preprocessed" + str(counter), "image_sample_preprocessed" + str(counter), "image_subtracted_prc_positive" + str(counter), "image_subtracted_prc_negative" + str(counter), "image_subtracted_prc_or" + str(counter), "image_subtracted_prc_rso" + str(counter), "image_subtracted_prc_rso_2" + str(counter)]
        #save_image(list_temp_save_image, path="temp_files/extractor_difference/method-6", filename=filename, format="png")
        
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
        image_reference_preprocessed = preprocessing_4(image_reference, threshold_config)
        image_sample_preprocessed = preprocessing_4(image_sample, threshold_config)
        
        #image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2) )
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_CLOSE, kernel, iterations = 1)
        erode = cv2.erode(dilate, kernel, iterations = 1)


        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio
        )
        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations = 1)

        
        image_sample_masked_gray_removed_small_objects = res
        #contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid_contour = contour_Centroids(not_removed_buffer)
        
        #show_pack = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, image_subtracted_prc_rso]
        #show_image(show_pack, open_order=2, window=True)
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
        #image_reference_preprocessed = preprocessing_4(image_reference, threshold_config)
        #image_sample_preprocessed = preprocessing_4(image_sample, threshold_config)
        
        #image_subtracted_prc = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_sample = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_ref = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        image_subtracted_prc = cv2.add(image_subtracted_prc_ref, image_subtracted_prc_sample)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2) )
        # dilate = cv2.dilate(image_subtracted_prc, kernel, iterations = 1) # 1
        dilate = cv2.morphologyEx(image_subtracted_prc, cv2.MORPH_OPEN, kernel, iterations = 1)
        erode = cv2.erode(dilate, kernel, iterations = 1)


        image_subtracted_prc_rso, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
            erode.copy(),
            #is_contour_number_for_area=True,
            # is_filter=is_filter, 
            ratio=rso_ratio
        )
        res = cv2.dilate(image_subtracted_prc_rso, kernel, iterations = 1)

        
        image_sample_masked_gray_removed_small_objects = res
        #contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroid_contour = contour_Centroids(not_removed_buffer)
        
        #show_pack = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc, image_subtracted_prc_rso]
        #show_image(show_pack, open_order=2, window=True)
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
    
    # MASKING
    # image_sample_masked_prc = cv2.bitwise_or(image_sample_gray, image_sample_gray, mask=image_subtracted_prc)
    stop_diff = time.time() - start_diff

    start_rso = time.time()
    # src, contours, buffer, removed_buffer, not_removed_buffer
    """
    image_sample_masked_gray_removed_small_objects, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
        image_subtracted_prc.copy(),
        is_contour_number_for_area=is_contour_number_for_area,
        ratio=ratio, 
        buffer_percentage=buffer_percentage, 
        is_filter=is_filter, 
        filter_lower_ratio=filter_lower_ratio, 
        filter_upper_ratio=filter_upper_ratio
    )
    """
    
    ####EXTRA####
    image_sample_masked_gray_removed_small_objects, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
        image_subtracted_prc.copy(),
        ratio=5,
    )
    """
    contours, _ = cv2.findContours(image_subtracted_prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_sample_masked_gray_removed_small_objects = image_subtracted_prc.copy()
    for _, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 5:
            cv2.drawContours(image_sample_masked_gray_removed_small_objects, [cnt], 0, (0,0,0), -1)
    """
    stop_rso = time.time() - start_rso
    ##############
    
    
    # import pdb; pdb.set_trace()
    
    
    start_draw = time.time()
    if method == 2:
        """th = threshold(image_subtracted_prc.copy(), configs=[1,255])
        th = cv2.bitwise_not(th)
        kernel = np.array([ [1,1,1], 
                            [1,1,1], 
                            [1,1,1]])
        erode = cv2.filter2D(th, -1, kernel, cv2.BORDER_DEFAULT)
        erode = cv2.bitwise_not(erode)"""
        
        image_subtracted_prc_negative = cv2.subtract(image_sample_preprocessed, image_reference_preprocessed)
        image_subtracted_prc_positive  = cv2.subtract(image_reference_preprocessed, image_sample_preprocessed)
        
        image_subtracted_prc = cv2.add(image_subtracted_prc_positive, image_subtracted_prc_negative)

        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2) )
        erode = cv2.erode(image_subtracted_prc, kernel, iterations = 2)

        rso = remove_Small_Object(erode.copy(), is_chosen_max_area=True, ratio=1)[0] #50
        
        res = cv2.dilate(rso, kernel, iterations = 1)
        #res = cv2.erode(res, kernel, iterations = 1)
        
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10) )
        #dilation = cv2.dilate(rso, kernel, iterations = 5)
        
        contours, _ = cv2.findContours(rso, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print("METHOD-2_CNTS:", contours)
        centroid_contour = contour_Centroids(contours)

        """
        list_temp_save_image = [image_reference, image_sample, image_reference_preprocessed, image_sample_preprocessed, image_subtracted_prc_negative, image_subtracted_prc_positive, image_subtracted_prc, erode, rso, res]
        filename = [str(counter)+"_1ref", str(counter)+"_2sample", str(counter)+"_3ref_pre", str(counter)+"_4sample_pre", str(counter)+"_5prc_negative", str(counter)+"_6prc_positive", str(counter)+"_7prc_add", str(counter)+"_8erode", str(counter)+"_9rso", str(counter)+"_10res"]
        save_image(list_temp_save_image, path="temp_files/extractor_difference/", filename=filename, format="png")
        """
        if activate_debug_images:
            save_image([image_reference], path="Operator_Debug/pano_symbols/reference_symbols", filename=[str(counter)+"_1_ref"], format="png")
            save_image([image_reference_preprocessed], path="Operator_Debug/pano_symbols/reference_preprocessed", filename=[str(counter)+"_2_ref_prep"], format="png")
            save_image([image_sample], path="Operator_Debug/pano_symbols/sample_symbols", filename=[str(counter)+"_3_sample"], format="png")
            save_image([image_sample_preprocessed], path="Operator_Debug/pano_symbols/sample_preprocessed", filename=[str(counter)+"_4_sample_prep"], format="png")
            save_image([image_subtracted_prc], path="Operator_Debug/pano_symbols/extractor_symbols", filename=[str(counter)+"_5_subtracted_add"], format="png")
            save_image([rso], path="Operator_Debug/pano_symbols/rso_symbols", filename=[str(counter)+"_6_rso"], format="png")
        
        
    if draw_diff:
        image_sample[image_sample_masked_gray_removed_small_objects != 0] = fill_color
    
    if draw_diff_circle:
        center_points = contour_Centroids(contours)
        
        alpha = 0.7
        beta = 1
        frame_display = image_sample.copy()
        roi = frame_display.copy()
        for center_point in center_points:
            draw_Circle(
                roi,
                tuple(center_point),
                radius=35,
                color=fill_color,
                thickness=-1
            )
        image_sample = cv2.addWeighted(frame_display, alpha, roi, beta, 0.0)
            
    stop_draw = time.time() - start_draw
    stop_total = time.time() - start_diff
    
    """
    stdo(1, "SUBSTRACTION TIMES -- DIFF: {:.3f} | RSO: {:.3f} | DRAW: {:.3f} | TOTAL: {:.3f}".format
                (
                    stop_diff,
                    stop_rso,
                    stop_draw,
                    stop_total
                )
            )

    print("TODO: remove - not_removed_buffer, image_subtracted_prc ON THE RETURN")
    """
    #TODO: remove - not_removed_buffer, image_subtracted_prc
    return image_sample_masked_gray_removed_small_objects, image_sample, contours, not_removed_buffer, centroid_contour, image_subtracted_prc


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
