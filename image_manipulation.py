"""
Written for Python 3.7.x
Tested with Python 3.7.7 on Windows 10

usage: image_manipulation.py [-h] [-iip INPUT_IMAGE_PATH]
                             [-oip OUTPUT_IMAGE_PATH]
                             [-p PREPROCESS [PREPROCESS ...]]
                             [-c [CONFIGS [CONFIGS ...]]]

optional arguments:
  -h, --help            show this help message and exit
  -iip INPUT_IMAGE_PATH, --input-image-path INPUT_IMAGE_PATH
                        path to input image
  -oip OUTPUT_IMAGE_PATH, --output-image-path OUTPUT_IMAGE_PATH
                        path to output image
  -p PREPROCESS [PREPROCESS ...], --preprocess PREPROCESS [PREPROCESS ...]
                        type of preprocessing to be done
  -c [CONFIGS [CONFIGS ...]], --configs [CONFIGS [CONFIGS ...]]
                        configs of preprocessing to be done (such as '256 256'
                        for resize (without quotes))

Example Usage:
        python image_manipulation.py -iip .\2k_1.png -p "grayscale" "gaussian_blur" "threshold" "median_blur" "resize" -c 5 5 0 -1 -1 1 -1 -1

        image_manipulation.py           -> Program Main Run File
        -i i2.png                       -> Image Path Parameter
        -cam no                         -> Camera Usage Parameter
        -p "grayscale" "gaussian_blur" "threshold" "median_blur" "resize"   -> Preprocesses as Parameter
        -c 5 5 0 -1 -1 1 -1 -1          -> Configs of Preprocesses as Parameter

Snippets:
    python image_manipulation.py -iip .\2k_1.png -p "grayscale" "gaussian_blur" "threshold" "median_blur" "resize" -c 5 5 0 -1 -1 1 -1 -1
    python image_manipulation.py -iip .\2k_1.png -p "grayscale" "gaussian_blur" "threshold" "resize" -c 1 1 2 -1 -1 -1 -1
    python image_manipulation.py -iip .\2k_1.png -p "grayscale" "gaussian_blur" "threshold" -c 1 1 2 -1 -1

"""

import time

import hashlib
import inspect
from tools import stdo
import cv2
# import circleTools
import numpy as np
import itertools


from skimage import exposure
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import ransac, LineModelND


from inspect import currentframe, getframeinfo
from image_tools import is_numpy_image


from image_tools import show_image, save_image
from math_tools import coordinate_Scaling, filter_Close_Points, filter_Close_Points_2, filter_Closest_Bottom_Points, get_Angles_Of_Horizontal_Lines, determine_Line_Position, compute_Contour_Thickness, compute_Contour_Thickness_2, line_Intersection

# from fourier_transform import fft_blur, fft2


def detect_Pallette(src, method=2):
    palette = ""
    if method == 1:
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        data = np.reshape(hsv, (-1,3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _,_,centers = cv2.kmeans(data,1,None,criteria,10,flags)
        
        b = centers[0].astype(np.int32)[0]
        g = centers[0].astype(np.int32)[1]
        r = centers[0].astype(np.int32)[2]

        if 0 < b < 50 and 0 < g < 50 and 170 < r < 255:
            palette = "white"
        if 20 < b < 100 and 100 < g < 170 and 50 < r < 150:    
            palette = "black"
        if 20 < b < 100 and 170 < g < 255 and 50 < r < 170:
            palette = "gray"
    
    if method == 2:
        
        #src must be respectively - undistorted, cropped and resized
        
        gamma = 2
        lookUpTable = np.empty((1,256), np.uint8)
        for j in range(256):
            lookUpTable[0,j] = np.clip(pow(j / 255.0, float(gamma)) * 255.0, 0, 255)
        gc = cv2.LUT(src, lookUpTable)

        bgr_planes = cv2.split(gc)
        histSize = 256
        histRange = (0, 256)
        accumulate = False
        b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

        cv2.normalize(b_hist, b_hist, alpha=0, beta=src.shape[0], norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=src.shape[0], norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=src.shape[0], norm_type=cv2.NORM_MINMAX)

        ratio_b = sum(b_hist) / 3
        ratio_g = sum(g_hist) / 3
        ratio_r = sum(r_hist) / 3

        print("Pallet Color:", ratio_b, ratio_g, ratio_r)
        
        if 500 < ratio_b < 2500 and 500 < ratio_g < 2300 and 500 < ratio_r < 3500:
            palette = "white"
        elif 1000 < ratio_b < 1800 and 800 < ratio_g < 2000 and 600 < ratio_r < 3000:    
            palette = "black"
        elif 0 < ratio_b < 1000 and 2000 < ratio_g < 10000 and 3000 < ratio_r < 15000:
            palette = "gray"

        """if 0 < ratio_b < 600 and 0 < ratio_g < 600 and 0 < ratio_r < 600:
            palette = "white"
        elif 1000 < ratio_b < 1800 and 800 < ratio_g < 2000 and 600 < ratio_r < 3000:    
            palette = "black"
        elif 0 < ratio_b < 1000 and 2000 < ratio_g < 10000 and 3000 < ratio_r < 15000:
            palette = "gray"""
    
    return palette


def fill_in(src):
    fill = src
    mask = np.zeros((fill.shape[0] + 2, fill.shape[1] + 2), np.uint8)
    cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
    inlay_symbols = cv2.bitwise_not(fill)

    return fill, mask, inlay_symbols


def mixture_transparent(image, overlay, transparency_factor=0.4):
    # x, y, w, h = 10, 10, 10, 10  # Rectangle parameters
    # cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)  # A filled rectangle

    # Following line overlays transparent rectangle over the image
    return cv2.addWeighted(
        overlay,
        transparency_factor,
        image,
        1 - transparency_factor,
        0
    )


def contour_Extreme_Points(list_contour, get_centroid=False, is_width_height=False):
    leftmost = tuple(list_contour[list_contour[:,:,0].argmin()][0])
    rightmost = tuple(list_contour[list_contour[:,:,0].argmax()][0])
    topmost = tuple(list_contour[list_contour[:,:,1].argmin()][0])
    bottommost = tuple(list_contour[list_contour[:,:,1].argmax()][0])
    centroid = (-1, -1)
    
    # stdo(1, "contour_Extreme_Points-rightmost: {} |||| {}".format(list_contour, rightmost))

    if get_centroid:
        centroid = (int(round((leftmost[0] + rightmost[0] + topmost[0] + bottommost[0]) / 4)), int(round((leftmost[1] + rightmost[1] + topmost[1] + bottommost[1]) / 4)))

    if is_width_height:
        start_x = leftmost[0]
        start_y = topmost[1]
        w = abs(leftmost[0] - rightmost[0])
        h = abs(topmost[1] - bottommost[1])
        return_data = [start_x, start_y, w, h, centroid]
        
    else:
        return_data = [leftmost, topmost, rightmost, bottommost, centroid]

    return return_data


def contour_Moments(list_contour):
    moments = list()
    for contour in list_contour:
        moments.append(cv2.moments(contour))
    return moments


def contour_Areas(list_contour, is_chosen_max=True, is_contour_number_for_area=False):
    contour_areas = list()
    
    for contour in list_contour:
        if is_chosen_max:
            current_area_cv2 = cv2.contourArea(contour)
            current_area_len = len(contour)
            if current_area_cv2 > current_area_len:
                contour_areas.append(current_area_cv2)
            else:
                contour_areas.append(current_area_len)

        else:
            if is_contour_number_for_area:
                contour_areas.append(len(contour))
            else:
                contour_areas.append(cv2.contourArea(contour))

    return contour_areas


def contour_Centroids(list_contour, get_bbox_centroid=False, is_single=False):
    contour_centroids = list()
    if get_bbox_centroid:
        if is_single:
            startx = list_contour[0]
            starty = list_contour[1]
            endx = startx + list_contour[2]
            endy = starty + list_contour[3]

            centroids_x = int((startx + endx) / 2)
            centroids_y = int((starty + endy) / 2)

            contour_centroids.append([centroids_x, centroids_y])
        else:
            for contour in list_contour:
                startx = contour[0]
                starty = contour[1]
                endx = startx + contour[2]
                endy = starty + contour[3]

                centroids_x = int((startx + endx) / 2)
                centroids_y = int((starty + endy) / 2)

                contour_centroids.append([centroids_x, centroids_y])
    else:
        for contour in list_contour:
            moment = cv2.moments(contour)
            if moment["m00"] != 0:
                centroids_x = int(moment['m10']/moment['m00'])
                centroids_y = int(moment['m01']/moment['m00'])
            else:
                centroids_x, centroids_y = -1, -1
                #continue
            #if centroids_x == -1 and centroids_y == -1:
                
            contour_centroids.append([centroids_x, centroids_y])

    return contour_centroids


def contour_Information(list_contour):
    # https://docs.opencv.org/4.1.2/dd/d49/tutorial_py_contour_features.html

    contour_information = dict()

    contour_information["areas"] = list() 
    contour_information["moments"] = list() 
    contour_information["centroids"] = list() 
    contour_information["perimeter"] = list() 
    contour_information["convexity"] = list() 

    for contour in list_contour:
        contour_information["areas"].append(cv2.contourArea(contour))
        contour_information["moments"].append(cv2.moments(contour))

        centroids_x = int(contour_information["moments"][-1]['m10']/contour_information["moments"][-1]['m00'])
        centroids_y = int(contour_information["moments"][-1]['m01']/contour_information["moments"][-1]['m00'])
        contour_information["centroids"].append([centroids_x, centroids_y])

        contour_information["perimeter"].append(cv2.arcLength(contour,True))

        """ # https://docs.opencv.org/4.1.2/dd/d49/tutorial_py_contour_features.html#Contour_Approximation
        epsilon_arch_len = 0.1
        epsilon = epsilon_arch_len * cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        """

        contour_information["convexity"].append(cv2.isContourConvex(contour))

    return contour_information


def adjust_brightness(img, brightness_factor):
    # https://www.programcreek.com/python/example/89460/cv2.LUT
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if len(img.shape) == 2:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            return cv2.LUT(img, table)[:,:,np.newaxis]
        else:
            return cv2.LUT(img, table) 
    else:
        return cv2.LUT(img, table) 


def adjust_Contrast(img, contrast_factor=1.5, method='Adjust', alpha=1, beta=0):
    # https://www.programcreek.com/python/example/89460/cv2.LUT
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    
    if method == 'Adjust':
        if not is_numpy_image(img):
            raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
        table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(contrast_factor)
        if len(img.shape) == 3:
            if img.shape[2]==1:
                return cv2.LUT(img, table)[:,:,np.newaxis]
            else:
                return cv2.LUT(img, table)
        else:
            return cv2.LUT(img,table)
        
    elif method == 'Stretching':
        #new_image = np.zeros(img.shape, img.dtype)
        
        """
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
        """
        
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img
        


def gamma_Correction(img, gamma = 2):
    lookUpTable = np.empty((1,256), np.uint8)
    for j in range(256):
        lookUpTable[0,j] = np.clip(pow(j / 255.0, float(gamma)) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)


def look_Up_Table(src, down_table=[], up_table=[], is_gray_scale = True):
    # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#lut
    """
        ### ### ### 
        ### LUT ###
        ### ### ### 
        Performs a look-up table transform of an array.

        C++: void LUT(InputArray src, InputArray lut, OutputArray dst, int interpolation=0 )
        Python: cv2.LUT(src, lut[, dst[, interpolation]]) → dst
        C: void cvLUT(const CvArr* src, CvArr* dst, const CvArr* lut)
        Python: cv.LUT(src, dst, lut) → None
        Parameters:	
            src – input array of 8-bit elements.
            lut – look-up table of 256 elements; in case of multi-channel input array, the table should either have a single channel (in this case the same table is used for all channels) or the same number of channels as in the input array.
            dst – output array of the same size and number of channels as src, and the same depth as lut.
        The function LUT fills the output array with values from the look-up table. Indices of the entries are taken from the input array. 
    """
    if is_gray_scale:
        #Sample usage of tables#
        #down_table = [0, 64, 128, 190, 255]
        #up_table = [0, 16, 128, 240, 255]

        if len(down_table) != 5 and len(up_table) != 5:
            text = 'All sended table parameters length must be equal 5'
            raise Exception(text)
        else:
            xee = np.arange(256)
            table = np.interp(xee, down_table, up_table).astype('uint8')
            lut = cv2.LUT(src, table)
            return lut, table
    else:
        stdo(3, "Not implemented. Check the comments in function... (REF: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#lut)")

def remove_Small_Object(
        src, 
        is_chosen_max_area=True, 
        is_contour_number_for_area=True, 
        ratio=0, 
        buffer_percentage=30, 
        is_filter=True, 
        filter_lower_ratio=7, 
        filter_upper_ratio=253, 
        aspect='lower', 
        elim_coord=[],
        counter=0
    ):
    
    if buffer_percentage > 100:
        buffer_percentage = 100
    if buffer_percentage < 0:
        buffer_percentage = 0

    if filter_lower_ratio > filter_upper_ratio:
        stdo(3, "Lower Filter Ratio cant be bigger than Lower Upper Ratio.")
        return -1, -1, -1, -1, -1

    # contours, hierarchy
    contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blind_coords = list() # For Process
    not_removed_buffer = list() # For Process
    will_be_removed_buffer = list() # For Process
    removed_buffer = list() # For return
    
    list_index = list()
    list_area = list()
    
    start_buffer = time.time()
    for i, cnt in enumerate(contours):
        current_area = contour_Areas([cnt], is_chosen_max=is_chosen_max_area, is_contour_number_for_area=is_contour_number_for_area)[0]
        list_index.append(i)
        list_area.append(current_area)
    
    list_area_sorted, index_list_area_sorted = sorted(list_area), sorted(range(len(list_area)), key=lambda k: list_area[k])
    
    for i, current_area in enumerate(list_area_sorted):
        if is_filter:
            if current_area < filter_lower_ratio:
                will_be_removed_buffer.append(index_list_area_sorted[i])
                continue
            if current_area > filter_upper_ratio:
                not_removed_buffer.append(index_list_area_sorted[i])
                continue
        blind_coords.append(index_list_area_sorted[i])
    stop_buffer = time.time() - start_buffer

    if ratio == 0:
        start_draw = time.time()
        blind_coords_length = len(blind_coords)
        blind_coords_percentage = round((blind_coords_length * buffer_percentage) / 100)

        
        #stdo(4, "! buffer {}".format(buffer))
        #stdo(4, "! buffer[:{}] {}".format(buffer_len, buffer[:buffer_len]))
        will_be_removed_buffer += blind_coords[:blind_coords_percentage]
        not_removed_buffer += blind_coords[blind_coords_percentage:]
        removed_buffer = will_be_removed_buffer[:]
        #stdo(4, "! removed_buffer {}".format(removed_buffer))
        #stdo(4, "! not_removed_buffer {}".format(not_removed_buffer))
        
    
        test_draw = src.copy()
        
        if blind_coords_percentage != 0:
            for i in will_be_removed_buffer:
                cv2.drawContours(src, contours[i], -1, (0, 0, 0), -1)
                """cv2.putText(test_draw, str(list_area[i]), 
                            (contours[i][0][0][0],  contours[i][0][0][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255,255,255), 1, cv2.LINE_AA)"""
                cv2.imwrite("C:/Users/mahmut.yasak/Desktop/test_draw.png", test_draw)
            
        stop_draw = time.time() - start_draw
        
    
    elif ratio == -1:
        removed_buffer = list()
        not_removed_buffer = contours

    else:
        start_draw = time.time()
        
        if elim_coord:
            #import pdb; pdb.set_trace()
            xmin, xmax, ymin, ymax = elim_coord
            for i, cnt in enumerate(contours):
                for cnt_sub in cnt[0]:
                    if (cnt_sub[0] < xmin) or (cnt_sub[0] > xmax) or (cnt_sub[1] < ymin) or (cnt_sub[1] > ymax):
                        cv2.drawContours(src, [cnt], 0, [0, 0, 0], -1)
                        continue
            
        else:
            removed_buffer = list()
            not_removed_buffer = list()
            
            for index, cnt in enumerate(contours):
                if aspect == 'lower':
                    #print(counter, ": ", cv2.contourArea(cnt), "|", ratio)
                    if cv2.contourArea(cnt) < ratio:
                        removed_buffer.append(cnt)
                        cv2.drawContours(src, [cnt], 0, [0, 0, 0], -1)
                    else:
                        not_removed_buffer.append(cnt)
                elif aspect == 'upper':
                    if cv2.contourArea(cnt) > ratio:
                        removed_buffer.append(cnt)
                        cv2.drawContours(src, [cnt], 0, (0,0,0), -1)
                    else:
                        not_removed_buffer.append(cnt)
        stop_draw = time.time() - start_draw
        
    """
    stdo(1, "RSO TIMES -- BUFFER: {:.3f} | DRAW: {:.3f}".format(
            stop_buffer,
            stop_draw
        )
    )
    """
    
    return src, contours, list_area, removed_buffer, not_removed_buffer

def remove_Small_Object_old(src, is_chosen_max_area=True, is_contour_number_for_area=True, ratio=0, buffer_percentage=30, is_filter=True, filter_lower_ratio=7, filter_upper_ratio=253):
    if buffer_percentage > 100:
        buffer_percentage = 100
    if buffer_percentage < 0:
        buffer_percentage = 0

    if filter_lower_ratio > filter_upper_ratio:
        stdo(3, "Lower Filter Ratio cant be bigger than Lower Upper Ratio.")
        return -1, -1, -1, -1, -1

    # contours, hierarchy
    contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    buffer = list() # For Process
    not_removed_buffer = list() # For Process
    will_be_removed_buffer = list() # For Process
    removed_buffer = list() # For return
    
    #stdo(4, contours)
    #stdo(4, "contours: {}".format(contour_Areas(contours, is_chosen_max=is_chosen_max_area, is_contour_number_for_area=is_contour_number_for_area)))
    #import pdb; pdb.set_trace()
    start_buffer = time.time()
    for i, cnt in enumerate(contours):
        current_area = contour_Areas([cnt], is_chosen_max=is_chosen_max_area, is_contour_number_for_area=is_contour_number_for_area)[0]
        if is_filter:
            if current_area < filter_lower_ratio:
                will_be_removed_buffer.append(current_area)
                continue
            if current_area > filter_upper_ratio:
                not_removed_buffer.append(current_area)
                continue
        buffer.append(current_area)
    stop_buffer = time.time() - start_buffer

    if ratio == 0:
        start_elim = time.time()
        objects_contour = len(buffer)
        buffer_len = round((objects_contour * buffer_percentage) / 100)

        #index_sorted_buffer = sorted(range(len(buffer)), key=lambda k: buffer[k])
        buffer.sort()
        
        #stdo(4, "! buffer {}".format(buffer))
        #stdo(4, "! buffer[:{}] {}".format(buffer_len, buffer[:buffer_len]))
        will_be_removed_buffer += buffer[:buffer_len]
        not_removed_buffer += buffer[buffer_len:]
        removed_buffer = will_be_removed_buffer[:]
        #stdo(4, "! removed_buffer {}".format(removed_buffer))
        #stdo(4, "! not_removed_buffer {}".format(not_removed_buffer))
        stop_elim = time.time() - start_elim
        
        start_draw = time.time()
        if buffer_len != 0:
            
            for _, cnt in enumerate(contours):
                for buffer_contour, area in enumerate(will_be_removed_buffer):
                    current_area = contour_Areas([cnt], is_contour_number_for_area=is_contour_number_for_area)[0]
                    if area == current_area:
                        cv2.drawContours(src, [cnt], -1, (0, 0, 0), -1)
                        will_be_removed_buffer.pop(buffer_contour)
                        objects_contour -= 1
                        break
        stop_draw = time.time() - start_draw
        
    
    else:
        start_elim = time.time()
        removed_buffer = list()
        not_removed_buffer = list()
        for _, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < ratio:
                removed_buffer.append(cnt)
                cv2.drawContours(src, [cnt], 0, [0, 0, 0], -1)
            else:
                not_removed_buffer.append(cnt)
        stop_elim = time.time() - start_elim
        
    stdo(1, "RSO TIMES -- BUFFER: {:.3f} | ELİM: {:.3f} | DRAW: {:.3f}".format
                (
                    stop_buffer,
                    stop_elim,
                    stop_draw
                )
            )
    
    return src, contours, buffer, removed_buffer, not_removed_buffer
    
def draw_Text(image, text=[], center_point=(0,0), fontscale=1, color=(0,255,0), thickness=2, plain=False):
    
    if plain:
        text_format = "{}".format(text)
        
    else:
        count_index = 0
        for _, text_ in enumerate(text):
            count_index += 1
        
        if count_index == 1:
            text_format = "{}".format(*text)
        else:
            text_format = "{}:{}".format(*text)
        
    cv2.putText(
        image,
        text_format, 
        (center_point[0], center_point[1]-20), 
        cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA
    )
    return image

def draw_Line(image, start_point, end_point, color=(255, 255, 255), thickness=-1):
    return cv2.line(image, start_point, end_point, color, thickness)

def draw_Circle(image, center_point, radius=1, color=(255, 255, 255), thickness=-1):
    
    # stdo(1, "draw_Circle: center_point:{}".format(center_point))
    
    return cv2.circle(image, center_point, radius, color, thickness)

def draw_Rectangle(image, start_point, end_point, color=(255, 255, 255), thickness=-1): 
    image = cv2.rectangle(image, (int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1])), color, thickness)
    return image

def draw_Arrow(image, start_point, end_point, color=(255, 255, 255), thickness=-1):
    image = cv2.arrowedLine(image, start_point, end_point, color, thickness) 
    return image

def transparent_Draw(src_frame, alpha=0.5, beta=1, radius=5, pts=[], fill_color=(0,0,255)):
    frame_display = src_frame.copy()
    roi = frame_display.copy()
    
    #import pdb; pdb.set_trace()

    if len(pts) != 0:
        for center_point in pts:
            #if not center_point.all():
            #    continue
            
            draw_Circle(
                roi,
                tuple(center_point),
                radius=radius,
                color=fill_color,
                thickness=-1
            )
        return cv2.addWeighted(frame_display, alpha, roi, beta, 0.0)
    
    else:
        return -1

def api_Draw_Complementarily(
        image,
        drawing_function_list=[
            draw_Rectangle, 
            draw_Circle
        ],
        drawing_function_param_list=[
            [
                (0, 0),
                (10, 10),
                (0,255,0),
                5
            ],
            [
                (0, 0),
                (10, 10),
                (0,255,0),
                5
            ]
        ],
    ):
    for index, draw_function in enumerate(drawing_function_list):
        draw_function(image, *drawing_function_param_list[index])

"""
def fast_fourier_transform(img, configs):
    if configs == "":
        return fft2(img, "")
"""

def pruning(src, method=1, is_horizontal=True, is_vertical=True, over_run=1):

    if method == 1:
        kernel = np.array([
            [0,  1,  0], 
            [1, -1,  1], 
            [0,  1,  0], 
        ], np.uint8)
        kernel = np.array([
            [0,  -1,  0], 
            [-1, 1,  -1], 
            [0,  -1,  0], 
        ], np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
    elif method == 2:
        if is_horizontal:
            kernel = np.array([
                [0, -1, 0], 
                [0, -1, 1], 
                [0, -1, 0], 
            ], np.uint8)
            src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
            kernel = np.array([
                [0, -1, 0], 
                [1, -1, 0], 
                [0, -1, 0], 
            ], np.uint8)
            src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)

        if is_vertical:
            kernel = np.array([
                [0, 1, 0], 
                [-1, -1, -1], 
                [0, 0, 0], 
            ], np.uint8)
            src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
            kernel = np.array([
                [0, 0, 0], 
                [-1, -1, -1], 
                [0, 1, 0], 
            ], np.uint8)
            src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)

    """
    if is_horizontal:
        kernel = np.array([
            [0, 1, 0], 
            [0, 1, 1], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.erode(src, kernel, iterations=over_run)
        kernel = np.array([
            [0, 1, 0], 
            [1, 1, 0], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.erode(src, kernel, iterations=over_run)

    if is_vertical:
        kernel = np.array([
            [0, 1, 0], 
            [1, 1, 1], 
            [0, 0, 0], 
        ], np.uint8)
        src = cv2.erode(src, kernel, iterations=over_run)
        kernel = np.array([
            [0, 0, 0], 
            [1, 1, 1], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.erode(src, kernel, iterations=over_run)
    """

    return src

def thinning(src, is_horizontal=True, is_vertical=True, over_run=1):

    if is_horizontal:
        kernel = np.array([
            [0, 1, 0], 
            [0, 1, 1], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
        kernel = np.array([
            [0, 1, 0], 
            [1, 1, 0], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)

    if is_vertical:
        kernel = np.array([
            [0, 1, 0], 
            [1, 1, 1], 
            [0, 0, 0], 
        ], np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
        kernel = np.array([
            [0, 0, 0], 
            [1, 1, 1], 
            [0, 1, 0], 
        ], np.uint8)
        src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)

    return src

def erosion(img, kernel):
    erosion_of_img = None
    if kernel == []:
        kernel = np.ones((5, 5), np.uint8)
    # elif configs == [-1, -1, -1]:
    #     kernel = np.ones((3, 3), np.uint8)
    # elif configs == [-1, -2, -1]:
    #     kernel = np.ones((5, 5), np.uint8)
    # elif configs == [-1, -3, -1]:
    #     kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # else:
    #     kernel = np.array(configs, np.uint8)

    erosion_of_img = cv2.erode(img, kernel, iterations=1)

    return erosion_of_img

def dilation(img, kernel):
    dilate_of_img = None
    if kernel == []:
        kernel = np.ones((5, 5), np.uint8)
    # elif configs == [-1, -1, -1]:
    #     kernel = np.ones((3, 3), np.uint8)
    # elif configs == [-1, -2, -1]:
    #     kernel = np.ones((5, 5), np.uint8)
    # elif configs == [-1, -3, -1]:
    #     kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # else:
    #     # configs = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    #     # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    #     kernel = np.array(configs, np.uint8)

    dilate_of_img = cv2.dilate(img, kernel, iterations=1)

    return dilate_of_img

def opening(img, configs):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    stdo(3, "This function still at development")

def closing(img, configs):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    stdo(3, "This function still at development")

def crop(img, configs):
    # x and y start points of cropping, h and w, height and width that cropped area
    # recommended 400x400 for ORC
    #try:
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape

    if configs == [
        -1,
        -1,
        -1,
    ]:  # Custom Config for taking image's center point with fixed crop length
        cropLength = 170
        maxHeight = w // 2 + cropLength
        minHeight = w // 2 - cropLength

        maxWidth = h // 2 + cropLength
        minWidth = h // 2 - cropLength

        stdo(
            1,
            "Cropped Image Size: {0}x{1}".format(
                str(maxHeight - minHeight), str(maxWidth - minWidth)
            ),
        )
        return img[minHeight:maxHeight, minWidth:maxWidth].copy()

    elif configs == [
        -1,
        -1,
        -2,
    ]:  # Custom Config for taking image's center point with fixed crop length
        if w > h:
            cropLength = round(h / 3)
        else:
            cropLength = round(w / 3)

        maxHeight = w // 2 + cropLength
        minHeight = w // 2 - cropLength

        maxWidth = h // 2 + cropLength
        minWidth = h // 2 - cropLength

        stdo(
            1,
            "Cropped Image Size: {0}x{1}".format(
                str(maxHeight - minHeight), str(maxWidth - minWidth)
            ),
        )
        return img[minHeight:maxHeight, minWidth:maxWidth].copy()

    elif configs == [
        -1,
        -1,
        configs[2],
    ]:  # Custom Config for taking image's center point with given crop length
        cropLength = configs[2]
        maxHeight = w // 2 + cropLength
        minHeight = w // 2 - cropLength

        maxWidth = h // 2 + cropLength
        minWidth = h // 2 - cropLength

        stdo(
            1,
            "Cropped Image Size: {0}x{1}".format(
                str(maxHeight - minHeight), str(maxWidth - minWidth)
            ),
        )
        return img[minHeight:maxHeight, minWidth:maxWidth].copy()

    elif configs != "" and configs != [""] and configs != []:

        startY, startX, cropLength = configs

        maxHeight = startX + cropLength
        minHeight = startX - cropLength

        maxWidth = startY + cropLength
        minWidth = startY - cropLength

        """
            If maximum height bigger than picture's height
                then add the part that goes over the edge length to min height and sync max height to picture's height,
            If maximum width bigger than pictures width
                then add the part that goes over the edge length to min width and sync max width to picture's width,

            If minimum height smaller than picture's height
                then add the part that goes over the edge length to max height and sync max height to zero,
            If minimum width smaller than picture's width
                then add the part that goes over the edge length to max width and sync max width to zero,

        import pdb; pdb.set_trace()
        if h < maxHeight:
            minHeight = minHeight - maxHeight - h
            maxHeight = h

        if w < maxWidth:
            minWidth = minWidth - maxWidth - w
            maxWidth = w

        if minHeight < 0:
            maxHeight = (-minHeight) + maxHeight
            minHeight = 0

        if minWidth < 0:
            maxWidth = (-minWidth) + maxWidth
            minWidth = 0
        import pdb; pdb.set_trace()
        """

        if h < maxHeight:
            maxHeight = h

        if w < maxWidth:
            maxWidth = w

        if minHeight < 0:
            minHeight = 0

        if minWidth < 0:
            minWidth = 0

        """     # For Debug
        stdo(1,
        "\n\tmaxHeight = " + str(maxHeight) +
        "\n\tminHeight = " + str(minHeight) +
        "\n\tmaxWidth = " + str(maxWidth) +
        "\n\tminWidth = " + str(minWidth) )
        import pdb; pdb.set_trace()
            # img[startY - cropLength: startY + cropLength, startX - cropLength: startX + cropLength]
        """

        if maxHeight > h or maxWidth > w:
            stdo(
                2,
                "Crop Height or Width parameter bigger than image shape. Returning image itself without process.",
            )
            return img
        elif h == startX + cropLength and w == startY + cropLength:
            stdo(
                2,
                "Crop Height and Width parameter equal to image shape. Returning image itself without process.",
            )
            return img
        else:
            stdo(
                1,
                "Cropped Image Size: {0}x{1}".format(
                    str(maxHeight - minHeight), str(maxWidth - minWidth)
                ),
            )
            return img[minHeight:maxHeight, minWidth:maxWidth].copy()

    else:
        stdo(2, "Crop operation not applied, returning image itself.")
        return img

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'crop' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def resize(img, configs):
    # For some advices: https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
    ##try:
    if configs == [-1, -1]:
        return cv2.resize(
            img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )  # resize the image
    elif configs == [-1, 0]:
        return cv2.resize(
            img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )  # resize the image
    elif configs == [-1, 1]:
        return cv2.resize(
            img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
        )  # resize the image
    elif configs != "" and configs != [""] and configs != []:

        if configs[0] < img.shape[0] or configs[1] < img.shape[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        return cv2.resize(
            img, (configs[0], configs[1]), interpolation
        )  # resize the image
    else:
        return cv2.resize(img, (620, 480))  # resize the image

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'resize' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
    """

# Padding
def expand(img, configs):
    # Expand will fill all empty spaces with white(1), black(0) or given grey(2) pixels
    #try:
    # Expand the image, and fill up the color to empty spaces
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    # Background Color
    if configs[0] == 0:
        color = 0  # Black
    elif configs[0] == 1:
        color = 255  # White
    elif configs[0] == 2:
        color = 123  # Grey
    else: 
        color = configs[0]  # Custom Color

    # Lock Aspect Ratio
    if configs[1] == 0:  # NO Lock Aspect Ratio
        size = (
            configs[2],
            configs[2],
        )  # Make a square image as background with given size
    elif configs[1] == 1:  # Lock Aspect Ratio
        size = (
            height * configs[2],
            width * configs[2],
        )  # Lock aspect ratio and multiple with given number

    elif configs[1] == 2:  # Lock Aspect Ratio
        size = (
            height + configs[2][0],
            width + configs[2][1],
        )  # Lock aspect ratio and multiple with given number
    else:
        size = (
            configs[2],
            configs[2],
        )  # Make a square image as background with given size

    # print("shape h-w:", height, width)
    # print("size:", size)
    backgroundImg = np.zeros(size, np.uint8)  # default color will be black - RGB 0
    backgroundImg[:] = color
    bImageHight, bImageWidth = backgroundImg.shape[:2]  # [:2] for remove rgb
    # print("bImage", bImageHight, bImageWidth)

    startPoints = ((bImageWidth - width) // 2, (bImageHight - height) // 2)
    #endPoints = ((bImageWidth + width) // 2, (bImageHight + height) // 2)
    endPoints = ((startPoints[1] + height), (startPoints[0] + width))

    # print("startPoints:", startPoints)
    # print("endPoints:", endPoints)
    backgroundImg[
        startPoints[0] : endPoints[0], startPoints[1] : endPoints[1]
    ] = img

    """
    offset = ((b_Img_h_Width - width) // 2, (b_Img_h_Hight - height) // 2)
    offset = ((bImageWidth - width) // 2, (bImageHight - height) // 2)
    backgroundImg.paste(img, offset)
    """

    return backgroundImg

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'expand' image manipulation -> "
            "An error occurred while doing 'expand' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def grayscale_Conversion(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convert_RGB(img, configs):
    # https://www.programcreek.com/python/example/89231/skimage.color.gray2rgb
    # from skimage.color import gray2rgb
    # TODO: Removed SKIMAGE dependency
    #try:
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        c = 1
        h, w = img.shape
    """
    except Exception:
        # Removed SKIMAGE Dependency
        # from skimage.color import gray2rgb
        h, w = img.shape
        stdo(
            3,
            "An error occurred while doing 'convert_RGB' image manipulation -> Grayscale image arrived.",
        )
        # return gray2rgb(img)
    """
    if c == 1:
        return img
    if c == 3:
        return img
    elif c == 4:
        alpha = img[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255
        img = img[:, :, :3]
        canvas = np.ones((h, w, 3), dtype=np.float32) * 255
        composed = alpha * img + (1 - alpha) * canvas
        return composed.astype(np.uint8)
    else:
        stdo(
            3,
            "An error occurred while doing 'convert_RGB' image manipulation -> Invalid image channel size",
        )

def invert_color(img, configs):
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.bitwise_not(img)  # invert image color
    else:
        return cv2.bitwise_not(img)  # invert image color

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'invert_color' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def bilateral_blur(img, configs):
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.bilateralFilter(
            img, int(configs[0]), int(configs[1]), int(configs[2])
        )  # Blur to reduce noise
    else:
        return cv2.bilateralFilter(img, 11, 17, 17)  # Blur to reduce noise
    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'bilateral_blur' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def median_blur(img, configs=[]):
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.medianBlur(img, int(configs[0]))  # Blur to reduce noise
    else:
        return cv2.medianBlur(img, 3)  # Blur to reduce noise
    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'median_blur' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def gaussian_blur(img, configs):
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.GaussianBlur(
            img, (int(configs[0]), int(configs[1])), int(configs[2])
        )  # Blur to reduce gaussian noise
    else:
        return cv2.GaussianBlur(img, (5, 5), 0)  # Blur to reduce noise
    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'gaussian_blur' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        #try:
            recoveryConfigs = [0, 0, 0]
            for i in range(len(configs)):
                if len(configs) != i + 1:
                    recoveryConfigs[i] = configs[i] + 1
                else:
                    recoveryConfigs[i] = configs[i]
            stdo(
                2,
                "Trying different 'gaussian_blur' image manipulation parameters: Current{} -> Recovery{}".format(
                    configs, recoveryConfigs
                ),
            )
            return cv2.GaussianBlur(
                img,
                (int(recoveryConfigs[0]), int(recoveryConfigs[1])),
                int(recoveryConfigs[2]),
            )  # Blur to reduce gaussian noise
        except Exception as error:
            stdo(
                3,
                "An error occurred while doing 'gaussian_blur' image manipulation recovery action -> "
                + error.__str__(),
                getframeinfo(currentframe()),
            )
            return img
    """

def edge_Detection(image, method='Canny', configs=[100,200]):
    
    if method == 'Canny':
        if configs != "" and configs != [""] and configs != []:
            return cv2.Canny(image, int(configs[0]), int(configs[1]))
        else:
            return cv2.Canny(image, 100, 120)
        
    else:
        return image

def threshold(img, configs):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

    """
    Special parameters for threshold, use -1 as firs param, and than use -1, 0, 1, and so on, as second param...

    FOOTNOTE: threshold takes grayscale images as img parameter
    """
    #try:
    if configs == [-1, -1]:
        #   ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #   ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        #   ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        #   cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif configs == [-1, 0]:
        return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    elif configs == [-1, 1]:
        return cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
    elif configs == [-1, 2]:
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2
        )
    elif configs == [-1, 3]:
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif len(configs) == 3:
        if type(configs[2]) is not int:
            configs[2] = eval(configs[2])
        return cv2.threshold(img, int(configs[0]), int(configs[1]), configs[2])[1]

    elif configs != "" and configs != [""] and configs != []:
        return cv2.threshold(
            img, int(configs[0]), int(configs[1]), cv2.THRESH_BINARY)[1]
    else:
        return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'threshold' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def contours(img, configs):
    # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
    
    #try:
    """
    if configs != "" and configs != [""] and configs != []:
        return img
    else:
    """

    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # I tried threshold but edge_Detection better for more optimized output
    cEDImage = edge_Detection(img, [])

    """
    # import pdb; pdb.set_trace()
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    Upper codes failed because of fails with error "not enough values to unpack (expected 3, got 2)" error
    but fix is here: https://github.com/facebookresearch/maskrcnn-benchmark/issues/339
    """

    contours, hierarchy = cv2.findContours(
        cEDImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    # Draw Contours Docs: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'contours' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def label_connected(img, configs):
    # Label Connected
    #try:
    # https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    # https://stackoverflow.com/questions/51523765/how-to-use-opencv-connectedcomponents-to-get-the-images

    return cv2.connectedComponents(img)

    """
    except Exception as error:
        stdo(
            3,
            "An error occurred while doing 'label_connected' image manipulation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return 0
    """

def connected_components(img, configs):
    """
    At development state
    """
    # https://stackoverflow.com/questions/51523765/how-to-use-opencv-connectedcomponents-to-get-the-images
    # image = cv2.imread('image.png', cv2.IMREAD_UNCHANGED);
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # getting mask with connectComponents
    ret, labels = cv2.connectedComponents(binary)
    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        cv2.imshow("component", mask)
        cv2.waitKey(0)

    # getting ROIs with find_contours
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ROI = image[y : y + h, x : x + w]
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def deskew(img, configs=[0, 0]):
    """
    Should be applied after simplified one character image for the best result
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if configs == []:
        configs = [0, 0]
    SZ = img_gray.shape[0]
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    m = cv2.moments(img_gray)
    if abs(m["mu02"]) < 1e-2:  # If there is no need to De-skewing, return original image
        return img

    if configs == [0, 0]:  # Auto re-De-skewing mode off - One time De-skewing
        deskewCount = 1
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)

    elif configs == [1, 0]:  # Auto re-De-skewing mode on - Multi De-skewing
        deskewCount = 0
        while abs(m["mu02"]) >= 1e-2:
            deskewCount += 1
            skew = m["mu11"] / m["mu02"]
            M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mNew = cv2.moments(img_gray)
            if m == mNew:  # To prevent infinite looping
                break
            else:
                m = mNew

    elif (
        configs[0] == 1
    ):  # Custom number of re-re-De-skewing mode on - Multi De-skewing with given number
        deskewCount = configs[1]
        while deskewCount > 0:
            deskewCount -= 1
            skew = m["mu11"] / m["mu02"]
            M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mNew = cv2.moments(img_gray)
            if m == mNew:  # To prevent infinite looping
                break
            else:
                m = mNew

    else:
        stdo(
            1,
            """   '- Wrong configurations for De-skewing - configs: {}""".format(
                configs
            ),
        )
        return img

    stdo(1, """   '- {} number of De-skewing applied to image""".format(deskewCount))

    return img

def hog(img):
    bin_n = 16  # Number of bins

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = []
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist, bin_cells, mag_cells

def palette(img):
    """
    Return palette in descending order of frequency
    """
    arr = np.asarray(img)
    palette, index = np.unique(convert_as_np_void(arr).ravel(), return_inverse=True)
    palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
    count = np.bincount(index)
    order = np.argsort(count)
    return palette[order[::-1]]

def convert_as_np_void(arr):
    """View the array as dtype np.void (bytes)
    This collapses ND-arrays to 1D-arrays, so you can perform 1D operations on them.
    http://stackoverflow.com/a/16216866/190597 (Jaime)
    http://stackoverflow.com/a/16840350/190597 (Jaime)
    Warning:
    >>> convert_as_np_void([-0.]) == convert_as_np_void([0.])
    array([False], dtype=bool)
    """
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def sobel_gradient(image, scale):
    grad_x = cv2.Sobel(
        image,
        cv2.CV_16S,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )
    grad_y = cv2.Sobel(
        image,
        cv2.CV_16S,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return dst

def sharpening(image, kernel_size=(5, 5), alpha=1.5, beta=-0.7, gamma=0, iteration=0):
    image_blurred = cv2.GaussianBlur(image, kernel_size, 20)
    image_sharpened = cv2.addWeighted(image, alpha, image_blurred, beta, gamma, image_blurred)

    for _ in range(iteration):
        image_blurred = cv2.GaussianBlur(image_sharpened, kernel_size, 20)
        image_sharpened = cv2.addWeighted(image_sharpened, alpha, image_blurred, beta, gamma, image_blurred)

    return image_sharpened

def Linear_Fitting_on_Locally_Deflection_Algorithm(list_line_points, tolerance=3):
    """
    ------
    PARAMS
    ------
    list_line_points : A array like variable of line points for optimizing (removing noise) as line
    tolerance=3 : Tolerate range of filter in both positive and negative (like from -3 to +3)   

    -------
    METHODS
    -------
    1. Method Extraction of Coefficient with Repetition Values at Sequence and Optimizing Non Tolerance Values;
        Get repetition of values from some function like histogram(list_line_points)
            Repetition of values is "element_occurrence_list"
            Values is "element_list"
            sigma (Σ) is the sum of given values

            Formula:
                weighted_mean = Σ(element_list * element_occurrence_list) / Σ(element_occurrence_list)

            Now filter the elements at sequence one by one with tolerance:
                if difference of element and weighted_mean is not between (-tolerance) and tolerance:
                    Equalize the element to weighted_mean
    """

    if type(list_line_points) is not np.array:
        list_line_points = np.array(list_line_points, dtype="int16")

    line_points_len = len(list_line_points)
    
    #TODO: Fix the start with no tolerance point issue
    detected_change_points, detected_stable_points = sequence_Changes_Detection_1D(
        list_line_points, 
        tolerance, 
        store_last_tolerance=True, 
        is_return_reverse=True
    )

    line_points_mean = 0
    local_buffer_len = 0
    last_mean_diff = 0
    mean_diff = 0
    mean_diff_number = 0

    # for index in range(0, line_points_len - buffer_len, 1):
    index = 0
    sequence_start = 0
    sequence_end = 0
    # is_first_no_tolerance = True
    while index < line_points_len:
        #TODO: Fix the start with no tolerance point issue
        
        if detected_change_points[index] is True:
            sequence_start = index

            for index_sub_level_1 in range(sequence_start + 1, line_points_len):
                if detected_change_points[index_sub_level_1] is False or index_sub_level_1 == line_points_len - 1:
                    sequence_end = index_sub_level_1  # - 1
                    index = index_sub_level_1
                    break

            if sequence_end == line_points_len - 1:
                local_buffer_optimized, last_mean_diff = optimized_sampling_in_range(
                    value_start=list_line_points[sequence_start - 1],
                    value_end=list_line_points[sequence_start - 1] + (
                        (line_points_len-sequence_start) * (mean_diff / mean_diff_number)),
                    number_of_samples=line_points_len-sequence_start
                )
                list_line_points[sequence_start:] = local_buffer_optimized
            else:
                local_buffer_optimized, last_mean_diff = optimized_sampling_in_range(
                    value_start=list_line_points[sequence_start - 1],
                    value_end=list_line_points[sequence_end],
                    number_of_samples=sequence_end-sequence_start
                )
                list_line_points[sequence_start:sequence_end] = local_buffer_optimized

            mean_diff += last_mean_diff
            mean_diff_number += 1
            line_points_mean += sum(local_buffer_optimized)
            local_buffer_len += len(local_buffer_optimized)
            if sequence_end == line_points_len:
                break
        index += 1

    if local_buffer_len > 0:
        line_points_mean = line_points_mean / local_buffer_len
    else:
        local_buffer_len = -1
        
    return list_line_points, line_points_mean

def auto_Interpolation(list_line_points, method=4, loop_over=1, tolerance=3, is_fixed_optimization=False, point_filter=[0], buffer_len=5):
    """
    Recommended Method(s): 4, 3 (with fixed point optimization)
    ------
    PARAMS
    ------
    list_line_points : A array like variable of line points for optimizing (removing noise) as line
    method=1 : Method selector
    loop_over=1 : Number of Render Loop 
    tolerance=3 : Tolerate range of filter in both positive and negative (like from -3 to +3)   
    is_fixed_optimization=False : To optimize the non tolerance range points with fixed (True) or dynamic (False) value

    -------
    METHODS
    -------
    1. Method Extraction of Coefficient with Repetitional Values at Sequence and Optimizing Non Tolerance Values;
        Get repetition of values from some function like histogram(list_line_points)
            Repetition of values is "element_occurrence_list"
            Values is "element_list"
            sigma (Σ) is the sum of given values

            Formula:
                weighted_mean = Σ(element_list * element_occurrence_list) / Σ(element_occurrence_list)

            Now filter the elements at sequence one by one with tolerance:
                if difference of element and weighted_mean is not between (-tolerance) and tolerance:
                    if is_fixed_optimization: # Fixed Equalization
                        Equalize the element to weighted_mean
                    else: # Dynamic Equalization
                        Equalize the element to mean of sum of element and weighted_mean
    """
    if type(list_line_points) is not np.array:
        list_line_points = np.array(list_line_points, dtype="int16")

    line_points_len = len(list_line_points)

    for i in range(loop_over):
        if method == 1:
            unique, indices = np.unique(list_line_points, return_counts=True)
            dict_sorted_line_points_occurrence = dict(zip(unique, indices))

            weighted_mean_of_value = 0
            sum_of_value = 0
            for key, value in dict_sorted_line_points_occurrence.items():
                if point_filter is not None:
                    if key in point_filter:
                        continue
                weighted_mean_of_value += (key * value)
                sum_of_value += value
            weighted_mean_of_value = weighted_mean_of_value / sum_of_value

            for index in range(line_points_len):
                in_tolerance = (- tolerance) < (
                    list_line_points[index] - weighted_mean_of_value) < tolerance
                if not in_tolerance:
                    if is_fixed_optimization:
                        list_line_points[index] = weighted_mean_of_value
                    else:
                        list_line_points[index] = (
                            list_line_points[index] + weighted_mean_of_value) / 2
            return list_line_points, weighted_mean_of_value
        if method == 2:
            unique, indices = np.unique(list_line_points, return_counts=True)
            dict_sorted_line_points_occurrence = dict(zip(unique, indices))

            detected_change_points, detected_stable_points = sequence_Changes_Detection_1D(
                list_line_points, tolerance, store_last_tolerance=True, is_return_reverse=True)

            not_changed_points = list_line_points[detected_stable_points]
            line_points_mean = sum(not_changed_points) / \
                len(not_changed_points)

            weighted_mean_of_value = 0
            sum_of_value = 0
            index = 0
            for key, value in dict_sorted_line_points_occurrence.items():
                if point_filter is not None:
                    if key in point_filter:
                        continue
                if detected_stable_points[index]:
                    # We don't use it but stays for future use cases
                    weighted_mean_of_value += (key * value)
                    sum_of_value += value
                index += 1
            # We don't use it but stays for future use cases
            weighted_mean_of_value = weighted_mean_of_value / sum_of_value

            for index in range(line_points_len):
                if detected_change_points[index]:
                    if is_fixed_optimization:
                        # list_line_points[index] = weighted_mean_of_value # We don't use it but stays for future use cases
                        list_line_points[index] = line_points_mean
                    else:
                        # list_line_points[index] = (weighted_mean_of_value * sum_of_value + list_line_points[index]) / (sum_of_value) # We don't use it but stays for future use cases
                        list_line_points[index] = (
                            line_points_mean * sum_of_value + list_line_points[index]) / (sum_of_value)
            return list_line_points, line_points_mean
        if method == 3:
            # unique, indices = np.unique(list_line_points, return_counts=True)
            # dict_sorted_line_points_occurrence = dict(zip(unique, indices))

            # DO NOT TAKE LOCAL DETECTED CHANGE POINTS! Because local change points may depend on start points like;
            # For Tolerance 3, take global and local change points:
            # GLOBAL
            #   [10, 11, 12, 18, 14, 15, 17, 19, 20]
            #                *
            # LOCAL
            #               [18, 14, 15, 17, 19, 20]
            #                     *   *
            detected_change_points, detected_stable_points = sequence_Changes_Detection_1D(
                list_line_points, tolerance, store_last_tolerance=True, is_return_reverse=True)

            line_points_mean = 0

            # for index in range(0, line_points_len - buffer_len, 1):
            for index in range(0, line_points_len, 1):
                local_buffer = list()
                local_sum_of_value = 0
                local_buffer = list_line_points[index: index + buffer_len]

                not_changed_points = list_line_points[index: index +
                                                      buffer_len][detected_stable_points[index: index + buffer_len]]
                #print("|", index, "|", not_changed_points, "|")
                if len(not_changed_points) == 0:
                    local_line_points_mean = list_line_points[index]
                    #print("local_line_points_mean", local_line_points_mean, "| list_line_points[index]", list_line_points[index], "|")
                else:
                    local_line_points_mean = sum(
                        not_changed_points) / len(not_changed_points)  # buffer_len

                for buffer_index in range(len(local_buffer)):
                    if point_filter is not None:
                        if local_buffer[buffer_index] in point_filter:
                            continue
                    if detected_stable_points[index + buffer_index]:
                        local_sum_of_value += 1

                for buffer_index in range(len(local_buffer)):
                    if detected_change_points[index + buffer_index]:
                        if is_fixed_optimization:
                            list_line_points[index +
                                             buffer_index] = local_line_points_mean
                        else:
                            list_line_points[index + buffer_index] = (local_line_points_mean * (local_sum_of_value) + list_line_points[index + buffer_index]) / (
                                len(local_buffer)) + abs(list_line_points[index + buffer_index] - local_line_points_mean) / 3  # len(local_buffer)
                            #, local_line_points_mean, list_line_points[index + buffer_index - 1: index + buffer_index + 2]

                            # list_line_points[index + buffer_index] = (local_line_points_mean * local_sum_of_value + list_line_points[index + buffer_index]) / (len(local_buffer))
                        # not detected_change_points[index + buffer_index]
                        detected_change_points[index + buffer_index] = False

                line_points_mean += local_line_points_mean
            line_points_mean = line_points_mean / (line_points_len - 1)

            return list_line_points, line_points_mean
        elif method == 4:
            #TODO: Fix the start with no tolerance point issue
            detected_change_points, detected_stable_points = sequence_Changes_Detection_1D(
                list_line_points, tolerance, store_last_tolerance=True, is_return_reverse=True)

            line_points_mean = 0
            local_buffer_len = 0
            last_mean_diff = 0
            mean_diff = 0
            mean_diff_number = 0

            # for index in range(0, line_points_len - buffer_len, 1):
            index = 0
            sequence_start = 0
            sequence_end = 0
            # is_first_no_tolerance = True
            while index < line_points_len:
                #TODO: Fix the start with no tolerance point issue
                """
                import pdb; pdb.set_trace();
                if is_first_no_tolerance:
                    while detected_change_points[index] is True:
                        index += 1
                is_first_no_tolerance = False
                """

                if detected_change_points[index] is True:
                    sequence_start = index

                    # if sequence_start < line_points_len:
                    for index_sub_level_1 in range(sequence_start + 1, line_points_len):
                        if detected_change_points[index_sub_level_1] is False or index_sub_level_1 == line_points_len - 1:
                            sequence_end = index_sub_level_1  # - 1
                            index = index_sub_level_1
                            break

                    if sequence_end == line_points_len - 1:
                        """
                        if mean_diff_number == 0:
                            import pdb
                            pdb.set_trace()
                        """
                        local_buffer = list_line_points[sequence_start:]
                        local_buffer_optimized, last_mean_diff = optimized_sampling_in_range(
                            value_start=list_line_points[sequence_start - 1],
                            value_end=list_line_points[sequence_start - 1] + (
                                (line_points_len-sequence_start) * (mean_diff / mean_diff_number)),
                            number_of_samples=line_points_len-sequence_start
                        )
                        list_line_points[sequence_start:] = local_buffer_optimized
                    else:
                        local_buffer = list_line_points[sequence_start:sequence_end]
                        local_buffer_optimized, last_mean_diff = optimized_sampling_in_range(
                            value_start=list_line_points[sequence_start - 1],
                            value_end=list_line_points[sequence_end],
                            number_of_samples=sequence_end-sequence_start
                        )
                        list_line_points[sequence_start:sequence_end] = local_buffer_optimized

                    mean_diff += last_mean_diff
                    mean_diff_number += 1
                    line_points_mean += sum(local_buffer_optimized)
                    local_buffer_len += len(local_buffer_optimized)
                    if sequence_end == line_points_len:
                        break
                index += 1

            if local_buffer_len > 0:
                line_points_mean = line_points_mean / local_buffer_len
            else:
                local_buffer_len = -1
            print("!!! - ::: ", list_line_points, " | ", line_points_mean)
            return list_line_points, line_points_mean

    stdo(3, "Non implemented method: {}".format(method))
    return list_line_points, -1

def optimized_sampling_in_range(value_start, value_end, number_of_samples):
    list_sample = list()
    mean_diff = (value_end - value_start) / (number_of_samples + 1)
    for index_sample in range(1, number_of_samples + 1):
        list_sample.append(value_start + index_sample * mean_diff)
    return list_sample, mean_diff

def sequence_Changes_Detection_1D(array_like, tolerance, store_last_tolerance=True, is_return_reverse=False):
    detected_sequence_changes_1d = list()

    last_element = array_like[0]
    for element in array_like:
        is_in_tolerance = (-tolerance) < (last_element - element) < tolerance
        detected_sequence_changes_1d.append(not is_in_tolerance)

        if store_last_tolerance and is_in_tolerance:
            last_element = element
            continue

    if is_return_reverse:
        # np.logical_not(detected_change_points) # Numpy is slower
        return detected_sequence_changes_1d, [not elem for elem in detected_sequence_changes_1d]
    else:
        return detected_sequence_changes_1d

def color_Range_Mask(
        img, 
        color_palette_lower=(0, 0, 0), 
        color_palette_upper=(255, 255, 255), 
        is_HSV=False,
        get_Max=False 
    ):
    
    if is_HSV:
        color_transformed = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        color_transformed = img
        
    mask = cv2.inRange(color_transformed, color_palette_lower, color_palette_upper)
    
    """
    area = 0
    if contours:
        for contour in contours:
            area += cv2.contourArea(contour)
    
    result = True
    if area:
        cnt = max(contours, key = cv2.contourArea)
        
        # cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        max_matched_frame_coords = np.array([x, y, w, h])
    else:
        result = False
        max_matched_frame_coords = np.array([-1, -1, -1, -1])
    """
    
    if get_Max:
        contours = [contour for contour in cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0] if contour != []
        ]
        if contours:
            cnt = max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            max_matched_frame_coords = np.array([x, y, w, h])
        else:
            max_matched_frame_coords = np.array([0, 0, 0, 0])
    else:
        max_matched_frame_coords = np.array([0, 0, 0, 0])
        
    return mask, max_matched_frame_coords

def color_Range_Mask_Using_Palette(
        img, 
        dict_color_palette, 
        type_color='HSV', 
        ranged_color='green', 
        morph_kernel=[5,5], 
        rso_ratio=1000,
        area_threshold=5000, 
        pattern_id='0',
        configuration_id='N/A', 
        show_result=False, 
        show_specified_component=None
    ):
    
    start_time = time.time()
    
    if type_color == 'HSV':
        color_transformed = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        color_transformed = img
    
    lower = dict_color_palette[ranged_color][type_color]['lower']
    upper = dict_color_palette[ranged_color][type_color]['upper']
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(color_transformed, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel[0], morph_kernel[1]))
    morph_close = cv2.morphologyEx(morph_dilate.copy(), cv2.MORPH_CLOSE, kernel)
    
    rso = remove_Small_Object(morph_close.copy(), ratio=rso_ratio)[0]    
    contours, _ = cv2.findContours(rso, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # if (show_result == True) and (show_specified_component is not None) and (show_specified_component == int(pattern_id)):
    #    show_image([img,color_transformed,mask,morph,rso], ['ORG','CVT','MASKED','MORPH','RSO'], open_order=5, figsize=((15,7)))
    
    area = 0
    if contours:
        for contour in contours:
            area += cv2.contourArea(contour)
    else:
        area = 0
    
    if area >= area_threshold:    
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        max_matched_frame_coords = np.array([x, y, w, h])
    else:
        max_matched_frame_coords = np.array([-1, -1, -1, -1])
    
    result_dashboard = ""
    stop_time = (time.time() - start_time) * 1000    
    if show_result:
        # stdo(1, "[{}][{}]: Color_Area:{} | Color_Area-Threshold:{}".format(pattern_id, configuration_id, area, area_threshold))
        stop_time = (time.time() - start_time) * 1000
        
        temp_pattern_id = str(pattern_id)
        temp_configuration_id = str(configuration_id)
        temp_area = str(area)
        temp_area_threshold = str(area_threshold)
        temp_stop_time = str(np.round(stop_time, 2))
        
        # result_dashboard = "[{}][{}] Color_A:{} - Color_A-Thr:{} - T:{:.2f}ms".format(pattern_id, configuration_id, area, area_threshold, stop_time)
        result_dashboard = "[" + temp_pattern_id + "]" + "[" + temp_configuration_id + "]" + "\n-C_A:" + temp_area + "\n-C_A.Thr:" + temp_area_threshold + "\n-T:" + temp_stop_time + "ms"
    
        
        image_pack = [img, color_transformed, mask, morph_dilate, morph_close, rso]
        title_pack = [
            configuration_id + "_1_ORG_" + str(pattern_id), 
            configuration_id + "_2_CVT_" + str(pattern_id), 
            configuration_id + "_3_MASKED_" + str(pattern_id), 
            configuration_id + "_4_MORPH_D" + str(pattern_id),
            configuration_id + "_5_MORPH_C" + str(pattern_id),
            configuration_id + "_6_RSO_" + str(pattern_id),
        ]
        save_image(image_pack, path="temp_files/color_Range_Mask_Using_Palette/", filename=title_pack, format="png")
        
    return area, max_matched_frame_coords, result_dashboard

def image_Undistortion(image, undistortion_camera_matrix, undistortion_distortion_coefficients, undistortion_new_camera_matrix):
    dst = cv2.undistort(
        image,
        undistortion_camera_matrix, 
        undistortion_distortion_coefficients, 
        None, 
        undistortion_new_camera_matrix
    )
    return dst

def coordinate_Undistortion(coords, undistortion_camera_matrix, undistortion_distortion_coefficients, undistortion_new_camera_matrix):
    undist_coordinates = cv2.undistortPoints(
        src = coords, # (coords[0], coords[1]),
        cameraMatrix = undistortion_camera_matrix, 
        distCoeffs = undistortion_distortion_coefficients,
        P = undistortion_new_camera_matrix
    )[0][0]
    
    undist_coords_x, undist_coords_y = [int(i) for i in undist_coordinates]
    
    return undist_coords_x, undist_coords_y 

def coordinate_Distortion_Back(coords, undistortion_new_camera_matrix, undistortion_distortion_coefficients):
    # https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/
    # https://stackoverflow.com/questions/21615298/opencv-distort-back
    
    fx = undistortion_new_camera_matrix[0,0]
    fy = undistortion_new_camera_matrix[1,1]
    cx = undistortion_new_camera_matrix[0,2]
    cy = undistortion_new_camera_matrix[1,2]
    k1 = undistortion_distortion_coefficients[0][0] * -1
    k2 = undistortion_distortion_coefficients[0][1] * -1
    k3 = undistortion_distortion_coefficients[0][4] * -1
    p1 = undistortion_distortion_coefficients[0][2] * -1
    p2 = undistortion_distortion_coefficients[0][3] * -1
    
    x = coords[0]
    y = coords[1]
    
    x = (x - cx) / fx
    y = (y - cy) / fy

    r2 = x*x + y*y

    xDistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    yDistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    xDistort = xDistort * fx + cx;
    yDistort = yDistort * fy + cy;

    return round(xDistort), round(yDistort)

def image_Perspective_Warping(image, homography_matrix):
    dst = cv2.warpPerspective(
        image, 
        homography_matrix, 
        (image.shape[1], image.shape[0])
    )
    return dst

def image_Make_Border(image, border_range=[10,10,10,10], border_flag=cv2.BORDER_CONSTANT, border_value=[0,0,0]):
    dst = cv2.copyMakeBorder(
        image, 
        *border_range, 
        border_flag,
        None,
        border_value
    )
    return dst

def image_Calculate_Border_Range(image, max_w=100, max_h=100):
    top = int( (max_h // 2) - (image.shape[0] // 2) )
    bottom = top
    if top + bottom + image.shape[0] < max_h:
        bottom += 1
    
    left = int( (max_w // 2) - (image.shape[1] // 2) )
    right = left
    if left + right + image.shape[1] < max_w:
        right += 1
    return top, bottom, left, right

def remove_Pixel_Width_old(image, contours, pixel_width=2, pixel_width_color=[0,0,255], counter=0, is_color=False):
    #contours = np.array(contours, dtype=object)
    
    dict_faults = dict()
    list_dict_actual_faults = list()
    list_dict_noisy_pixels = list()
    
    for cnt in contours:
        if len(cnt) > 1:
            a = []
            b = []
            for c in cnt:
                a.append(c[0][0])
                b.append(c[0][1])
            
            # print(counter, "x : ",  a)
            # print(counter, "y : ",  b)
            
            x_diff = abs(np.diff(a))
            y_diff = abs(np.diff(b))
            
            areas = contour_Areas([cnt], is_chosen_max=True, is_contour_number_for_area=False)
            
            if (np.any(x_diff[x_diff >= pixel_width] >= pixel_width)) and (np.any(y_diff[y_diff >= pixel_width] >= pixel_width)): # NOT REMOVE #
                
                """
                stdo(1, "NOK {}: {} ---- | x:{} y:{} | ---- | CondX:{} CondY:{} | ---- Area:{}".format(
                    counter,
                    cnt.reshape(-1),
                    x_diff,
                    y_diff,
                    (len(cnt) / 2),
                    (len(cnt) / 2),
                    areas
                ))
                """
                for c in cnt:
                    list_dict_actual_faults.append(c[0])
                
                continue
            
            else: # ELIMINATE #
                
                """
                stdo(1, "OK {}: {} ---- | x:{} y:{} | ---- | CondX:{} CondY:{} | ---- Area:{}".format(
                        counter,
                        cnt.reshape(-1),
                        x_diff,
                        y_diff,
                        (len(cnt) / 2),
                        (len(cnt) / 2),
                        areas
                ))
                """
                    
                if is_color:
                    if 4 <= areas[0] <= 6:
                        cv2.drawContours(image, [cnt], 0, [255, 165, 0], 1)
                    
                    elif 6 <= areas[0] <= 8:
                        cv2.drawContours(image, [cnt], 0, [0, 165, 255], 1)
                        
                    else:
                        cv2.drawContours(image, [cnt], 0, pixel_width_color, 1)
                    
                else:
                    cv2.drawContours(image, [cnt], 0, [0, 0, 0], 1)
                
                for c in cnt:
                    list_dict_noisy_pixels.append(c[0])
    
    dict_faults['actual_faults'] = np.array(list_dict_actual_faults)
    dict_faults['noisy_pixels'] = np.array(list_dict_noisy_pixels)
    
    # stdo(1, "[{}] actual_faults: {}".format(counter, dict_faults['actual_faults'].reshape(-1)))
    # stdo(1, "[{}] noisy_pixels: {}".format(counter, dict_faults['noisy_pixels'].reshape(-1)))
                
    return image, dict_faults
    #return image

def remove_Pixel_Width(image, contours, pixel_width=2, pixel_width_color=[0,0,255], method=1, counter=0, pano_sector_index='L', symbol_hull_list=[], is_color=False, title=''):
    
    dict_faults = list()
    # dict_faults = {"pano_sector_index":pano_sector_index, "symbol_id":counter, "contour_id":0, "actual_faults":[], "noisy_pixels":[], "hull_w":[], "hull_h":[], "hull_area":[], "hull_count_pixel_area":[]}

    if method == 1:
        for id, cnt in enumerate(contours):
            if len(cnt) < 1:
                continue  # Çok küçük konturları atla
            
            contour_width = compute_Contour_Thickness_2(cnt, img_shape=image.shape)
            
            """
            stdo(1, "boundingRect {}: {}".format(
                id,
                contour_width
            ))
            """
            
            # Eğer gerçek genişlik pixel_width'ten küçükse, gürültü olarak kabul et
            if contour_width <= pixel_width:
                color = pixel_width_color if is_color else [0, 0, 0]
                cv2.drawContours(image, [cnt], -1, color, 1)
                dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
            else:
                dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
        
    if method == 2:
        for id, cnt in enumerate(contours):
            if len(cnt) < 1:
                continue  # Çok küçük konturları atla
            
            # Minimum alan kaplayan dikdörtgeni hesapla
            rect = cv2.minAreaRect(cnt)  # (center, (width, height), angle)
            (cx, cy), (w, h), angle = rect  # Merkezi, genişliği, yüksekliği ve açıyı al
            if max(w, h) < pixel_width:
                if is_color:
                    color = pixel_width_color
                else:
                    color = [0, 0, 0]

                cv2.drawContours(image, [cnt], -1, color, 1)
                dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
            else:
                dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
        
    if method == 3:
        for id, cnt in enumerate(contours):
            if len(cnt) < 1:
                continue  # Çok küçük konturları atla
        
            x_vals = [c[0][0] for c in cnt]
            y_vals = [c[0][1] for c in cnt]

            x_range = np.ptp(x_vals)  # max(x) - min(x)
            y_range = np.ptp(y_vals)  # max(y) - min(y)

            areas = contour_Areas([cnt], is_chosen_max=True, is_contour_number_for_area=False)
            x, y, w, h = cv2.boundingRect(cnt)
            
            stdo(1, "remove_Pixel_Width {}: {} | x:{} y:{} | CondX:{} CondY:{} | Area:{}".format(
                counter,
                cnt.reshape(-1),
                x_range,
                y_range,
                w,
                h,
                areas
            ))
            
            if x_range >= pixel_width or y_range >= pixel_width:  # Gürültü olmayanlar
                dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
            else:  # Gürültü olarak kabul edilenler
                if is_color:
                    color = [255, 165, 0] if 4 <= areas[0] <= 6 else [0, 165, 255] if 6 < areas[0] <= 8 else pixel_width_color
                else:
                    color = [0, 0, 0]

                cv2.drawContours(image, [cnt], -1, color, 1)
                dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
        
        
        """
        # Bounding box hesapla (x, y, genişlik, yükseklik)
        x, y, w, h = cv2.boundingRect(cnt)
        
        stdo(1, "boundingRect {}: {} | x:{} y:{}  w:{} h:{}".format(
            counter,
            cnt.reshape(-1),
            x,
            y,
            w,
            h,
        ))

        # Eğer hem genişlik hem yükseklik pixel_width'ten küçükse, gürültü olarak işaretle
        if w <= pixel_width or h <= pixel_width:
            if is_color:
                color = pixel_width_color
            else:
                color = [0, 0, 0]

            cv2.drawContours(image, [cnt], -1, color, 1)
            dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
        else:
            dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
        """
    
    if method == 4:
        # Bağlı bileşenleri belirle (her konturu ayrı bir bileşen olarak düşün)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(mask, [cnt], -1, 255, -1)  # Konturu doldur

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for cnt in contours:
            if len(cnt) < 2:
                continue  # Çok küçük konturları atla

            # Konturun hangi bileşene ait olduğunu bul
            x, y, w, h = cv2.boundingRect(cnt)
            component_label = labels[y + h // 2, x + w // 2]  # Konturun merkezindeki label

            # Eğer bu bileşenin genişliği küçükse, tamamen kaldır
            if stats[component_label, cv2.CC_STAT_WIDTH] < pixel_width and stats[component_label, cv2.CC_STAT_HEIGHT] < pixel_width:
                color = pixel_width_color if is_color else [0, 0, 0]
                cv2.drawContours(image, [cnt], -1, color, 1)
                dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
            else:
                # **Büyük hataların dış konturlarını koruyoruz!**
                dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
    
    if method == 5:
        
        if is_color:
            draw_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            draw_binary = image.copy()
        
        list_cnt = []
        for id, cnt in enumerate(contours):
            #if len(cnt) < 1:
            #    continue  # Çok küçük konturları atla

            # Konturun hangi bileşene ait olduğunu bul
            # x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt) # stats[1:, cv2.CC_STAT_AREA][id]
                
            # rect, contour_width = compute_Contour_Thickness(id, cnt)
        
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Köşe noktalarını tam sayı yap

            # Uzaklıkları hesapla (dört kenarın uzunlukları)
            edge_lengths = [
                np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)
            ]
            contour_width = min(edge_lengths)
            
            x = round(rect[0][0])
            y = round(rect[0][1])
            w = round(rect[1][0])
            h = round(rect[1][1])
            
            
            ##########
            # mina = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(mina)
            # box = np.intp(box)
            # x_ptp = np.ptp(box[:,0]) 
            # y_ptp = np.ptp(box[:,1])
            # if x_ptp == 0 and y_ptp == 0:
            #     ptp_area = 1
            # else:
            #     ptp_area = x_ptp * y_ptp
            
            if  ( 
                (x <= pixel_width) or (y <= pixel_width)
            ) or (
                (w <= pixel_width) or (h <= pixel_width)
            ):
                if (contour_width < pixel_width):
                    
                    list_cnt.append(cnt)
                    
                    # if is_color:
                        # draw_color = cv2.drawContours(draw_color, [cnt], -1, pixel_width_color, 1)
                    # else:
                        # draw_binary = cv2.drawContours(draw_binary, [cnt], -1, (0,0,0), 1)
                    # dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
                
            if area < (w*h)/10:
                list_cnt.append(cnt)
                
                # if is_color:
                    # draw_color = cv2.drawContours(draw_color, [cnt], -1, pixel_width_color, 1)
                # else:
                    # draw_binary = cv2.drawContours(draw_binary, [cnt], -1, (0,0,0), 1)
                # dict_faults["actual_faults"].extend(cnt.reshape(-1, 2))
        
            # else:
            #     dict_faults["noisy_pixels"].extend(cnt.reshape(-1, 2))
                
        if is_color:
            draw_color = cv2.drawContours(draw_color, list_cnt, -1, pixel_width_color, 1)
            image = draw_color
        else:
            draw_binary = cv2.drawContours(draw_binary, list_cnt, -1, (0,0,0), 1)
            image = draw_binary
            
        del list_cnt

    if method == 6:
        
        draw_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if is_color else image.copy()
        list_cnt = []

        for id, cnt in enumerate(contours):
            
            flag_noisy_pixel = False
            
            dict_faults.append(dict())
            # NOT: dict_faults[id] --> id = contour_id
            dict_faults[id]["pano_sector_index"] = pano_sector_index
            dict_faults[id]["symbol_id"] = counter
            dict_faults[id]["noisy_pixels"] = []
            dict_faults[id]["actual_faults"] = []
            dict_faults[id]["max_thickness"] = []
            
            area = cv2.contourArea(cnt)
            if area <= 1:
                list_cnt.append(cnt)
                dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                continue

            _, _, w, h = cv2.boundingRect(cnt)
            
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            rect_hull = cv2.minAreaRect(hull)
            # w_hull = rect_hull[1][1]
            # h_hull = rect_hull[1][0]
            # Köşe noktaları (float)
            box_f = cv2.boxPoints(rect_hull)  # float döner
            # box_f = np.array(box_f, dtype=np.float32) # İstersen float32 olarak sabitleyebilirsin

            # Kenar vektörleri
            edge1 = box_f[1] - box_f[0]
            edge2 = box_f[2] - box_f[1]

            # Yatay eksene yakın kenarı width olarak al
            if abs(edge1[0]) > abs(edge1[1]):
                w_hull = np.linalg.norm(edge1)  # float
                h_hull = np.linalg.norm(edge2)  # float
            else:
                w_hull = np.linalg.norm(edge2)  # float
                h_hull = np.linalg.norm(edge1)  # float
            dict_faults[id]["hull_w"] = round(w_hull, 2)
            dict_faults[id]["hull_h"] = round(h_hull, 2)
            dict_faults[id]["hull_area"] = round(area_hull, 2)
            
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
            masked = cv2.bitwise_and(image, image, mask=mask)
            count_pixel = cv2.countNonZero(masked)
            dict_faults[id]["hull_count_pixel_area"] = round(count_pixel, 2)
            
            dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
            max_thickness = dist_transform.max() * 2

            # if ((w_hull <= pixel_width) or (h_hull <= pixel_width)) or ((w <= pixel_width)) or ((h <= pixel_width) or max_thickness <= pixel_width):
            #     list_cnt.append(cnt)
            #     dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
            #     flag_noisy_pixel = True

            # # elif ( ((w*h) * 0.1) > area ) and ( ((w_hull*h_hull) * 0.8) > area_hull ): # if ((w*h)/2.5 > area_hull) and ((w*h)//10 > area): # Change 20.07.2025
            # #     list_cnt.append(cnt)
            # if ( ((w*h) * 0.2) > count_pixel ) and ( ((w_hull*h_hull) * 0.5) > count_pixel ):
            #     list_cnt.append(cnt)
            #     dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
            #     flag_noisy_pixel = True
            
            # if ( 
            #     (w_hull > image.shape[1] / 3) and (h_hull < image.shape[0] / 6) 
            # ) or (
            #     (h_hull > image.shape[0] / 3) and (w_hull < image.shape[1] / 6)
            # ): # Image Represent
            #     flag_noisy_pixel = False
            if ( 
                (w_hull > image.shape[1] / 3) and (h_hull < image.shape[0] / 6) 
            ) or (
                (h_hull > image.shape[0] / 3) and (w_hull < image.shape[1] / 6)
            ): # Image Represent
                list_cnt.append(cnt)
                flag_noisy_pixel = True
            
            else:
                if ((w_hull/h_hull) > 5 or (h_hull/w_hull) > 5): # Thin-Long Noise Fault #
                    
                    if ( (w * h) * 0.3 > count_pixel ) or ( (w_hull * h_hull) * 0.3 > count_pixel ):
                    
                        list_cnt.append(cnt)
                        dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                        flag_noisy_pixel = True
                        
                if ( (w * h) * 0.3 > count_pixel ) or ( (w_hull * h_hull) * 0.3 > count_pixel ):
                    list_cnt.append(cnt)
                    dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                    flag_noisy_pixel = True
            
            if title == 'inside':
                if (w_hull <= pixel_width) or (h_hull <= pixel_width):
                    list_cnt.append(cnt)
                    flag_noisy_pixel = True
            
            if not flag_noisy_pixel:
                dict_faults[id]["actual_faults"].extend(cnt.reshape(-1, 2))
                # dict_faults[id]["max_thickness"].extend(max_thickness)
            
            # stdo(1, "[{}][{}] - area:{} | - wh:({},{}) | w_hull:{}, h_hull:{}) | pixel_width:{} | hull_area:{}".format(
            #     counter, id, 
            #     area,
            #     w, h,
            #     w_hull, h_hull,
            #     pixel_width,
            #     area_hull,
            # ))

        # if is_color:
        #     image = cv2.drawContours(draw_color, list_cnt, -1, pixel_width_color, 1)
        # else:
        #     image = cv2.drawContours(image, list_cnt, -1, (0, 0, 0), 1)
        #     # stdo(1, "[{}]: remove_Pixel_Width: {}".format(counter, list_cnt))
        if is_color:
            image = cv2.drawContours(draw_color, list_cnt, -1, color=pixel_width_color, thickness=-1)
        else:
            image = cv2.drawContours(image, list_cnt, -1, color=(0, 0, 0), thickness=-1)
    
    if method == 7:
        
        draw_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if is_color else image.copy()
        list_cnt = []
        # separator = Fore.YELLOW + " -> " + Style.RESET_ALL
        
        for id, cnt in enumerate(contours):
            
            # flag_status = "NOK"
            flag_noisy_pixel = False
            
            dict_faults.append(dict())
            # NOT: dict_faults[id] --> id = contour_id
            dict_faults[id]["pano_sector_index"] = pano_sector_index
            dict_faults[id]["symbol_id"] = counter
            dict_faults[id]["noisy_pixels"] = []
            dict_faults[id]["actual_faults"] = []
            
            area = cv2.contourArea(cnt)
            if area < 1:
                list_cnt.append(cnt)
                dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                continue

            _, _, w, h = cv2.boundingRect(cnt)
            
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            rect_hull = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect_hull)
            box = np.intp(box)
            if is_color:
                cv2.drawContours(draw_color, [box], -1, (0, 255, 0), 1)
            
            w_hull = rect_hull[1][0]
            h_hull = rect_hull[1][1]
            dict_faults[id]["hull_w"] = round(w_hull, 2)
            dict_faults[id]["hull_h"] = round(h_hull, 2)
            dict_faults[id]["hull_area"] = round(area_hull, 2)
            
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
            masked = cv2.bitwise_and(image, image, mask=mask)
            count_pixel = cv2.countNonZero(masked)
            dict_faults[id]["hull_count_pixel_area"] = round(count_pixel, 2)

            # dashboard =  "[{}] - pixel_width:{} | area:{} - wh:({:.2f},{:.2f}) | [startx,starty]:[{},{}] hull_area:{} - hull_wh:({:.2f},{:.2f}) | count_pixel:{}".format(
            #     id, 
            #     pixel_width,
            #     area,
            #     w, h,
            #     box[0][0], box[0][1],
            #     area_hull,
            #     w_hull, h_hull,
            #     count_pixel
            # )
            
            if ( (w_hull <= pixel_width) or (h_hull <= pixel_width) ) or ( (w <= pixel_width) or (h <= pixel_width) ):
                
                if ( ((w*h) * 0.2) < count_pixel ) and ( ((w_hull*h_hull) * 0.5) < count_pixel ):
                    pass
                else:
                    list_cnt.append(cnt)
                    dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                    flag_noisy_pixel = True
                    # flag_status = "OK"
                    # decide_dashboard = Fore.RED + "NOISE" + Style.RESET_ALL
                    # dashboard = dashboard + separator  + decide_dashboard
                    # stdo(1, dashboard)
                    # continue

            # elif ( ((w*h) * 0.1) > area ) and ( ((w_hull*h_hull) * 0.8) > area_hull ):
            #     list_cnt.append(cnt)
            #     flag_status = "OK"
            if ( ((w*h) * 0.2) > count_pixel ) and ( ((w_hull*h_hull) * 0.5) > count_pixel ):
                list_cnt.append(cnt)
                dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                flag_noisy_pixel = True
                # flag_status = "OK"
                
            # if not is_color:
            #     if flag_status == "OK":
            #         decide_dashboard = Fore.RED + "NOISE" + Style.RESET_ALL
            #     else:
            #         decide_dashboard = Fore.GREEN + "REAL-FAULT" + Style.RESET_ALL
            #     dashboard = dashboard + separator  + decide_dashboard
            #     stdo(1, dashboard)

            if not flag_noisy_pixel:
                dict_faults[id]["actual_faults"].extend(cnt.reshape(-1, 2))
                
        if is_color:
            image = cv2.drawContours(draw_color, list_cnt, -1, color=pixel_width_color, thickness=cv2.FILLED)
        else:
            image = cv2.drawContours(image, list_cnt, -1, color=(0, 0, 0), thickness=cv2.FILLED)
    
    if method == 8:
        
        draw_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if is_color else image.copy()
        list_cnt = []

        for id, cnt in enumerate(contours):
            
            flag_noisy_pixel = False
            
            dict_faults.append(dict())
            # NOT: dict_faults[id] --> id = contour_id
            dict_faults[id]["pano_sector_index"] = pano_sector_index
            dict_faults[id]["symbol_id"] = counter
            dict_faults[id]["noisy_pixels"] = []
            dict_faults[id]["actual_faults"] = []
            dict_faults[id]["max_thickness"] = []
            
            area = cv2.contourArea(cnt)
            if area <= 0.3:
                list_cnt.append(cnt)
                dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                continue

            _, _, w, h = cv2.boundingRect(cnt)
            
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            rect_hull = cv2.minAreaRect(hull)
            # w_hull = rect_hull[1][1]
            # h_hull = rect_hull[1][0]
            # Köşe noktaları (float)
            box_f = cv2.boxPoints(rect_hull)  # float döner
            # box_f = np.array(box_f, dtype=np.float32) # İstersen float32 olarak sabitleyebilirsin

            # Kenar vektörleri
            edge1 = box_f[1] - box_f[0]
            edge2 = box_f[2] - box_f[1]

            # Yatay eksene yakın kenarı width olarak al
            if abs(edge1[0]) > abs(edge1[1]):
                w_hull = np.linalg.norm(edge1)  # float
                h_hull = np.linalg.norm(edge2)  # float
            else:
                w_hull = np.linalg.norm(edge2)  # float
                h_hull = np.linalg.norm(edge1)  # float
            dict_faults[id]["hull_w"] = round(w_hull, 2)
            dict_faults[id]["hull_h"] = round(h_hull, 2)
            dict_faults[id]["hull_area"] = round(area_hull, 2)
            
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
            masked = cv2.bitwise_and(image, image, mask=mask)
            count_pixel = cv2.countNonZero(masked)
            dict_faults[id]["hull_count_pixel_area"] = round(count_pixel, 2)
        
            if ( (w_hull/h_hull > 8) or (h_hull/w_hull > 8) ): # Thin-Long Noise Fault #
                
                if ( (h_hull < 3) and (h_hull * 15 < w_hull) ) or ( (h_hull <= 1) ):
                    list_cnt.append(cnt)
                    dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                    flag_noisy_pixel = True
                
            if ( (w_hull * h_hull) * 0.5 > count_pixel ):
                list_cnt.append(cnt)
                dict_faults[id]["noisy_pixels"].extend(cnt.reshape(-1, 2))
                flag_noisy_pixel = True
            
            if not flag_noisy_pixel:
                dict_faults[id]["actual_faults"].extend(cnt.reshape(-1, 2))
                # dict_faults[id]["max_thickness"].extend(max_thickness)
                    
        if is_color:
            image = cv2.drawContours(draw_color, list_cnt, -1, color=pixel_width_color, thickness=-1)
        else:
            image = cv2.drawContours(image, list_cnt, -1, color=(0, 0, 0), thickness=-1)
    
    # dict_faults["actual_faults"] = np.array(dict_faults["actual_faults"])
    # dict_faults["noisy_pixels"] = np.array(dict_faults["noisy_pixels"])

    return image, dict_faults
    
def detect_Line_Object_old(line_eliminate_image, draw_masking_image, background_color='white', flag_activate_debug_images=False):
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detect_horizontal = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for cnt in cnts:
        cv2.drawContours(draw_masking_image, [cnt], 0, (255,255,255), 3)
        cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 3)


    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    detect_vertical = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for cnt in cnts:
        cv2.drawContours(draw_masking_image, [cnt], 0, (255,255,255), 3)
        cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 3)    

    kernel=np.ones((2,2))
    line_eliminate_image = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, kernel)
    
    if flag_activate_debug_images:
        list_temp_save_image = [line_eliminate_image, draw_masking_image]
        filename = ["line_eliminate_image", "draw_masking_image"]
        save_image(list_temp_save_image, path="temp_files/detect_Line_Object", filename=filename, format="png")
    
    return line_eliminate_image, draw_masking_image

def detect_Line_Object(line_eliminate_image, draw_masking_image, background_color='white', pano_title='', flag_activate_debug_images=False):
    
    # if background_color == 'white' or background_color == 'gray':
    line_image = np.zeros((line_eliminate_image.shape[0], line_eliminate_image.shape[1], 3), dtype=np.uint8)
    # elif background_color == 'black' or background_color == 'piano-black':
    #     line_image = np.ones((line_eliminate_image.shape[0], line_eliminate_image.shape[1], 3), dtype=np.uint8)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,2))
    detect_horizontal = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for id, cnt in enumerate(cnts):
        
        # stdo(1, "[{}] detect_Line_Object-cntx: {} | {}".format(id, cnt, cnt[0][:,0]))
        
        #cnt[0][:,0] = 0 if cnt[0][:,0]-20 < line_eliminate_image.shape[1] else cnt[0][:,0]-20
        # cnt[0][:,0] = 0 if cnt[0][:,0]-20 < 0 else cnt[0][:,0]-20
        
        if background_color == 'white':
            cv2.drawContours(draw_masking_image, [cnt], 0, (255,255,255), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (255,255,255), -1)
        
        elif background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
            cv2.drawContours(draw_masking_image, [cnt], 0, (154,179,209), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (154,179,209), -1)
        
        elif background_color == 'black' or background_color == 'piano-black':
            cv2.drawContours(draw_masking_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (255,255,255), -1)
        

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,200))
    detect_vertical = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for cnt in cnts:
        
        # stdo(1, "[{}] detect_Line_Object-cnty: {} | {}".format(id, cnt, cnt[0][:,1]))
        
        #cnt[0][:,1] = 0 if cnt[0][:,1]-20 < line_eliminate_image.shape[0] else cnt[0][:,1]-20
        # cnt[0][:,1] = 0 if cnt[0][:,1]-20 < 0 else cnt[0][:,1]-20
        
        if background_color == 'white':
            cv2.drawContours(draw_masking_image, [cnt], 0, (255,255,255), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (255,255,255), -1)
        
        elif background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
            cv2.drawContours(draw_masking_image, [cnt], 0, (154,179,209), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (154,179,209), -1)
        
        elif background_color == 'black' or background_color == 'piano-black':
            cv2.drawContours(draw_masking_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_eliminate_image, [cnt], 0, (0,0,0), 10)
            cv2.drawContours(line_image, [cnt], 0, (255,255,255), -1)   

    kernel=np.ones((2,2))
    line_eliminate_image = cv2.morphologyEx(line_eliminate_image, cv2.MORPH_OPEN, kernel)
    
    # if background_color == 'black' or background_color == 'piano-black':
    #     line_image = cv2.bitwise_not(line_image)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)
    
    if flag_activate_debug_images:
        list_temp_save_image = [line_eliminate_image, line_image, draw_masking_image]
        filename = [
            str(pano_title)+"_0_line_eliminate_image", 
            str(pano_title)+"_1_line_image", 
            str(pano_title)+"_3_draw_masking_image"
        ]
        save_image(list_temp_save_image, path="temp_files/detect_Line_Object", filename=filename, format="jpg")
    
    return line_eliminate_image, line_image, draw_masking_image
    
def image_Flip(image, task='horizontaly'):
    if task == 'horizontaly':
        stdo(1, "image_Flip :::::::::::: {}".format(image.shape))
        return cv2.flip(image, 0)
    elif task == 'verticaly':
        return cv2.flip(image, 1)

def image_Rotate(image, task='cv2.ROTATE_180', angle_rad=None, border_value=0):
    
    if angle_rad != None:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle_degree = np.rad2deg(angle_rad)
        M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=border_value)
    
    else:
        if task == 'cv2.ROTATE_180':
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return rotated

def circular_Hough_Transform(image, max_radius=50, is_circle_detected_image=False):
    circles = cv2.HoughCircles(
        image, 
        cv2.HOUGH_GRADIENT, 
        dp=2, 
        minDist=50, 
        param1=500, 
        param2=0.8, 
        minRadius=1,
        maxRadius=max_radius//2
    )

    if is_circle_detected_image:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        color = -1
        
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            #if r >= 24:
            x, y, r = circles[0]
            
            """
            else:
                r = 24
                x = mask.shape[0]//2
                y = mask.shape[1]//2
            """
            if is_circle_detected_image:
                cv2.circle(color, (x, y), r, (0, 255, 0), 1)
                # image_pack = [image, color]
                # title_pack = ["image", "color"]
                # save_image(image_pack, path="temp_files/circular_Hough_Transform", filename=title_pack, format="png")
    else:
        r = image.shape[0]//10
        x = image.shape[0]//2
        y = image.shape[1]//2
    
    # stdo(1, "Image-shape: {} | Circle-params:({},{},{})".format(image.shape, x,y,r))
    return x, y, r, color

def create_Circular_Mask(h, w, center=None, radius=None):
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    
    return mask.astype("uint8")

def calculation_Crop_Parameters_Of_Image_To_Focused_Interest_Area(image, inner_circle=(0,0,0), direction='up'):
    # inner_circle= (x,y,r)
    
    dict_component_direction = {
        'up': {
            'start_x': inner_circle[0] - inner_circle[2],
            'start_y': 0,
            'end_x': inner_circle[0] + inner_circle[2],
            'end_y': inner_circle[1] - inner_circle[2]
        },
        'down': {
            'start_x': inner_circle[0] - inner_circle[2],
            'start_y': inner_circle[1] + inner_circle[2],
            'end_x': inner_circle[0] + inner_circle[2],
            'end_y': image.shape[0]
        },
        'left': {
            'start_x': 0,
            'start_y': inner_circle[1] - inner_circle[2],
            'end_x': inner_circle[0] - inner_circle[2],
            'end_y': inner_circle[1] + inner_circle[2]
        },
        'right': {
            'start_x': inner_circle[0] + inner_circle[2],
            'start_y': inner_circle[1] - inner_circle[2],
            'end_x': image.shape[1],
            'end_y': inner_circle[0] + inner_circle[2]
        },
    }
    
    return dict_component_direction[direction]

def corner_Detection(image, method='harris'):
    if method == 'harris':
        result = cv2.cornerHarris(image, blockSize=5, ksize=3, k=0.1)
        
    elif method == 'shi-tomasi':
        corners1 = cv2.goodFeaturesToTrack(image, maxCorners=10, qualityLevel=0.1, minDistance=20, useHarrisDetector=True)
        corners1 = np.int0(corners1)
        
        if len(image.shape) == 2:        
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            color_image = image.copy()
            
        for i in corners1:
            x,y = i.ravel()
            cv2.circle(color_image, (x,y), 20, 255, -1)
        result = color_image
        
    return result

def calculation_Draw_Parameters_Of_Image_To_Focused_Interest_Area(coords=[0,0,0,0], direction='up'):
    
    # [start_X, start_Y, W, H]
    
    dict_component_direction = {
        'up': {
            'start_x': coords[0] + (coords[2] // 2),
            'start_y': coords[1] + coords[3],
            'end_x': coords[0] + (coords[2] // 2),
            'end_y': coords[1]
        },
        'down': {
            'start_x': coords[0] + (coords[2] // 2),
            'start_y': coords[1],
            'end_x': coords[0] + (coords[2] // 2),
            'end_y': coords[1] + coords[3]
        },
        'left': {
            'start_x': coords[0] + coords[2],
            'start_y': coords[1] + (coords[3] // 2),
            'end_x': coords[0],
            'end_y': coords[1] + (coords[3] // 2)
        },
        'right': {
            'start_x': coords[0],
            'start_y': coords[1] + (coords[3] // 2),
            'end_x': coords[0] + coords[2],
            'end_y': coords[1] + (coords[3] // 2)
        },
    }
    
    return dict_component_direction[direction]

def image_Add_Noise(symbol_image, noise_type='salt'):
    if noise_type == 'salt':
        row,col,ch= symbol_image.shape
        s_vs_p = 0.5
        amount = 0.004
        noise = np.copy(symbol_image)
        # Salt mode
        num_salt = np.ceil(amount * symbol_image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in symbol_image.shape]
        noise[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* symbol_image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in symbol_image.shape]
        noise[coords] = 0

    elif noise_type == 'gauss':
        row, col, ch= symbol_image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noise = symbol_image + gauss
            
    return noise

def image_Padding(image, pad_w, pad_h):
    top = int( (pad_h // 2) - (image.shape[0] // 2) )
    bottom = top
    if top + bottom + image.shape[0] < pad_h:
        bottom += 1
    
    left = int( (pad_w // 2) - (image.shape[1] // 2) )
    right = left
    if left + right + image.shape[1] < pad_w:
        right += 1
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [0,0,0])
    if (padded_image.shape[1] > pad_w) or (padded_image.shape[0] > pad_h):
        padded_image = cv2.resize(padded_image, (int(pad_w), int(pad_h)))

    return padded_image

def average_Color(image, coords=(0,0), size=3):
    x = coords[0]
    y = coords[1]
    half_size = size // 2
    
    x1, x2 = max(x - half_size, 0), min(x + half_size, image.shape[1])
    y1, y2 = max(y - half_size, 0), min(y + half_size, image.shape[0])
    
    # lambda x1, x2, y1, y2: [i[0] if type(i) is np.ndarray else i for i in [x1, x2, y1, y2]]
    
    if type(x1) is np.ndarray:
        x1 = x1[0]
    if type(x2) is np.ndarray:
        x2 = x2[0]
    if type(y1) is np.ndarray:
        y1 = y1[0]
    if type(y2) is np.ndarray:
        y2 = y2[0]
    
    # Ensure that the area is correctly handled at the edges of the image
    if x1 < x2 and y1 < y2:
        region = image[y1:y2, x1:x2]
        average_bgr = np.mean(region, axis=(0, 1))
        return tuple(map(int, average_bgr))
    
    else:
        return (0, 0, 0)

def detect_Color(R, G, B, color_table):
    min_distance = float('inf')
    cname = ''
    for i in range(len(color_table)):
        d = abs(R - int(color_table.loc[i, "R"])) + abs(G - int(color_table.loc[i, "G"])) + abs(B - int(color_table.loc[i, "B"]))
        if d < min_distance:
            min_distance = d
            cname = color_table.loc[i, "color_name"]
    return cname

def line_Detection(edge_image, opening_image, org_image, resize_edge_image, resize_opening_image, resize_org_image, method='hough_lines', method_model=None, num_lines=4, object_color='white', pano_title='', flag_activate_debug_images=False):
    
    if method == 'hough_lines':
        
        img_height, img_width = edge_image.shape
        if flag_activate_debug_images:
            draw_opening_image = cv2.cvtColor(opening_image.copy(), cv2.COLOR_GRAY2BGR)
            draw_org_image = org_image.copy()
        
        ###################################################
        tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
        h, theta, d = hough_line(edge_image, theta=tested_angles)
        hpeaks = hough_line_peaks(h, theta, d, threshold=0.3 * h.max())

        counter = 0
        list_hough_lines = []
        list_angle = []
        for _, angle, dist in zip(*hpeaks):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            # angle_radians = np.tan(angle + np.pi/2)
            angle_curve = np.tan(angle + np.round(np.pi,3)/2)
            list_hough_lines.append((x0,y0,angle_curve))
            list_angle.append(angle)
            
            # stdo(1, "[{}][HL-{}] ({:.1f},{:.1f}) | angle:{:.1f} | dist:{}".format(pano_title, counter, x0, y0, angle_curve, dist))
            counter += 1

        list_hough_lines = filter_Close_Points(list_hough_lines)

        lines_with_slope = [(x0, y0, angle) for x0, y0, angle in list_hough_lines]
        ###################################################
        
        
        ###################################################
        list_hough_coords = []
        for x0, y0, m in lines_with_slope:
            b = y0 - m * x0  # Y kesme noktası
            
            # Görüntü sınırlarıyla kesişimleri bul
            intersections = []

            # Sol kenar (x=0)
            y_at_x0 = b
            if 0 <= y_at_x0 <= img_height:
                intersections.append((0, y_at_x0))

            # Sağ kenar (x=img_width)
            y_at_xmax = m * img_width + b
            if 0 <= y_at_xmax <= img_height:
                intersections.append((img_width, y_at_xmax))

            # Üst kenar (y=0)
            x_at_y0 = -b / m if m != 0 else float('inf')
            if 0 <= x_at_y0 <= img_width:
                intersections.append((x_at_y0, 0))

            # Alt kenar (y=img_height)
            x_at_ymax = (img_height - b) / m if m != 0 else float('inf')
            if 0 <= x_at_ymax <= img_width:
                intersections.append((x_at_ymax, img_height))

            # Eğer en az 2 kesişim noktası varsa, doğruyu çiz
            if len(intersections) >= 2:
                (x1, y1), (x2, y2) = intersections[:2]
                list_hough_coords.append((round(x1),round(y1),round(x2),round(y2)))
                
                if flag_activate_debug_images:
                    draw_Line(draw_opening_image, start_point=(round(x1),round(y1)), end_point=(round(x2),round(y2)), color=(0, 255, 0), thickness=7)
        ###################################################
        
        
        ###################################################
        list_line_intersection_coords = []
        # list_line_angles = []
        counter = 0
        # Tüm doğruları ikili olarak karşılaştır
        for (x0_1, y0_1, m1), (x0_2, y0_2, m2) in itertools.combinations(lines_with_slope, 2):
            # Eğer eğimler eşitse (m1 == m2), doğrular paraleldir ve kesişmez
            if np.isclose(m1, m2):
                continue

            # Doğru denklemlerinin sabit terimleri (b) hesapla
            b1 = y0_1 - m1 * x0_1
            b2 = y0_2 - m2 * x0_2

            # Kesişim noktası (x, y) hesapla
            x_intersect = (b2 - b1) / (m1 - m2)
            y_intersect = m1 * x_intersect + b1

            if (0 <= x_intersect <= img_width) and (0 <= y_intersect <= img_height):
                
                list_line_intersection_coords.append((round(x_intersect), round(y_intersect)))
                # list_line_angles.append(m1)

                # stdo(1, "[{}][I-{}] ({:.1f},{:.1f})".format(pano_title, counter, x_intersect, y_intersect))
                counter += 1
                # ax.plot(x_intersect, y_intersect, 'go')
                # draw_Circle(draw_opening_image, center_point=(round(x_intersect), round(y_intersect)), radius=20, color=(0, 0, 255), thickness=-1)

        list_line_intersection_coords = filter_Close_Points_2(list_line_intersection_coords, distance_threshold=200)
        ###################################################
        
        ###################################################
        list_norm_line_intersection_coords = []
        if len(list_line_intersection_coords) > 4:
            positions = determine_Line_Position(list_line_intersection_coords, img_height, img_width)
            positions = np.array(positions)
            filtered_intersections = filter_Closest_Bottom_Points(positions)
            counter = 0
            for x, y, _ in filtered_intersections:
                # stdo(1, "[{}][normI-{}] ({:.1f},{:.1f})".format(pano_title, counter, round(float(x)), round(float(y))))
                if flag_activate_debug_images:
                    draw_Circle(draw_opening_image, (round(float(x)),round(float(y))), radius=20, color=(0, 0, 255), thickness=-1)
                list_norm_line_intersection_coords.append((round(float(x)),round(float(y))))
                counter += 1
                
        elif len(list_line_intersection_coords) == 4:
            counter = 0
            for x, y in list_line_intersection_coords:
                # stdo(1, "[{}][getI-{}] ({:.1f},{:.1f})".format(pano_title, counter, round(float(x)), round(float(y))))
                if flag_activate_debug_images:
                    draw_Circle(draw_opening_image, (round(float(x)),round(float(y))), radius=20, color=(0, 0, 255), thickness=-1)
                list_norm_line_intersection_coords.append((round(float(x)),round(float(y))))
                counter += 1

        for id, lines in enumerate(list_hough_coords):
            # lines = [i - 10 for i in lines]
            if flag_activate_debug_images:
                draw_Line(draw_org_image, start_point=(lines[0], lines[1]), end_point=(lines[2], lines[3]), color=(0, 255, 0), thickness=7)

        list_line_intersection_coords_repadding = []
        for id, coords in enumerate(list_norm_line_intersection_coords):
            # coords = [i - 10 for i in coords]
            if flag_activate_debug_images:
                draw_Circle(draw_org_image, center_point=(coords[0], coords[1]), radius=20, color=(0, 0, 255), thickness=-1)
            list_line_intersection_coords_repadding.append(coords)
        ###################################################
        
        ###################################################
        # if not flag_activate_debug_images:
        #     draw_org_image = org_image.copy()
        #     draw_opening_image = opening_image.copy()
            
        # d_image, pts, angle_rad_top, angle_rad_bottom = sort_points_for_warping(draw_org_image.copy(), list_line_intersection_coords_repadding, flag_activate_debug_images=flag_activate_debug_images)
        # pts = np.float32(pts)
        # tl = pts[0]
        # tr = pts[1]
        # bl = pts[2]
        # br = pts[3]

        # x_range = round(tr[0] - tl[0])
        # y_range = round(bl[1] - tl[1])
        # pts2 = np.float32([[0,0],[x_range,0],[0,y_range],[x_range,y_range]])

        # M = cv2.getPerspectiveTransform(pts, pts2)
        # warped_image = cv2.warpPerspective(org_image.copy(), M, (x_range,y_range))
        #>>>>>> return draw_opening_image, draw_org_image, d_image, warped_image, rotated, org_rotated, org_points_rotated_image, pts, org_rotate_coords
        
        angle_rad_top, angle_rad_bottom = get_Angles_Of_Horizontal_Lines(list_line_intersection_coords_repadding)
        
        ###################################################
        
        ###################################################
        if flag_activate_debug_images:
            rotated = image_Rotate(draw_org_image, angle_rad=angle_rad_top)
        org_rotated = image_Rotate(org_image.copy(), angle_rad=angle_rad_top)
        ###################################################
        
        ###################################################
        org_rotate_coords = np.array(list_line_intersection_coords_repadding).reshape(-1,2)
        org_rotate_coords[:,0], org_rotate_coords[:,1] = coordinate_Scaling(
            x=org_rotate_coords[:,0], 
            y=org_rotate_coords[:,1],
            old_w=False, old_h=False, 
            new_w=org_image.shape[1], new_h=org_image.shape[0],
            degree=angle_rad_top, 
            task="ANGULAR_ROTATION", 
            is_dual=False
        )
        
        if flag_activate_debug_images:
            org_points_rotated_image = org_rotated.copy()
            for id, coords in enumerate(org_rotate_coords):
                draw_Circle(org_points_rotated_image, center_point=(coords[0], coords[1]), radius=20, color=(0, 0, 255), thickness=-1)
        
        #>>>>>> return draw_opening_image, draw_org_image, d_image, warped_image, rotated, org_rotated, org_points_rotated_image, pts, org_rotate_coords
        ###################################################
        
        if flag_activate_debug_images:
            list_frame = [
                org_image, draw_opening_image, draw_org_image, rotated, org_rotated, org_points_rotated_image,
            ]
            list_name = [
                pano_title + '_0_org', 
                pano_title + '_1_HoughLines', 
                pano_title + '_2_Mask_HL_org', 
                pano_title + '_3_Rotate', 
                pano_title + '_4_Rotate_org', 
                pano_title + '_5_Rotate_org_Points'
            ]
            save_image(list_frame, path="temp_files/Pano_Detection/hough_lines", filename=list_name, format="jpg")
        
        return org_rotated, org_rotate_coords

    elif method == 'ransac':
        
        start_seq = time.time()
        
        start_def = time.time()
        resize_edge_image = cv2.Canny(resize_opening_image, 50, 150)
        img_height, img_width = resize_edge_image.shape
        if flag_activate_debug_images:
            draw_resize_opening_image = cv2.cvtColor(resize_opening_image.copy(), cv2.COLOR_GRAY2BGR)
            draw_resize_org_image = resize_org_image.copy()
        stop_def = time.time() - start_def
        
        ###################################################
        start_ransac = time.time()
        
        y_coords, x_coords = np.nonzero(resize_edge_image)
        points = np.column_stack((x_coords, y_coords))
        
        coords_org = []
        for i in range(num_lines):
            
            try:
                model, inliers = ransac(
                    points,
                    method_model,
                    min_samples=2,
                    residual_threshold=4,
                    max_trials=500,
                )
            except:
                org_rotated = org_image
                org_rotate_coords = [(0,0), (0,org_image.shape[0]), (org_image.shape[1],org_image.shape[0]), (org_image.shape[1],0)]
                angle_rad_top = 0
                return org_rotated, org_rotate_coords, angle_rad_top
            
            point_on_line = model.params[0]
            line_direction = model.params[1]  

            if abs(line_direction[0]) < 1e-6:  
                x_start = x_end = round(point_on_line[0])
                x_start = max(0, min(img_width - 1, x_start))
                y_start, y_end = 0, img_height  
            else:
                x_start, x_end = 0, img_width
                y_start = int(point_on_line[1] + (x_start - point_on_line[0]) * (line_direction[1] / line_direction[0]))
                y_end = int(point_on_line[1] + (x_end - point_on_line[0]) * (line_direction[1] / line_direction[0]))
    
            coords = [[x_start, y_start], [x_end, y_end]]
            coords_org.append((coords))
            
            if flag_activate_debug_images:
                draw_Line(draw_resize_opening_image, start_point=(x_start,y_start), end_point=(x_end,y_end), color=(0, 255, 0), thickness=5)
    
            points = points[~inliers]
            
        # stdo(1, "ransac-coords_org:{}".format(coords_org))
        
        stop_ransac = time.time() - start_ransac
        ###################################################
        
        ###################################################
        start_intersection = time.time()
        
        list_intersection = []
        margin = 200
        for x in range(4):
            for y in range(x + 1, 4):
                intersection = line_Intersection(coords_org[x], coords_org[y], method='4')
                x_val, y_val = intersection
                if (x_val, y_val) == (False, False):
                    continue
                if -margin <= x_val <= img_width + margin and -margin <= y_val <= img_height + margin:
                    list_intersection.append((x_val, y_val))
                    
                    if flag_activate_debug_images:
                        draw_Circle(draw_resize_org_image, center_point=(x_val, y_val), radius=20, color=(0, 0, 255), thickness=-1)
        
        if flag_activate_debug_images:
            list_frame = [
                resize_edge_image, draw_resize_opening_image, draw_resize_org_image
            ]
            list_name = [
                pano_title + '_resize_edge_image', 
                pano_title + '_draw_resize_opening_image', 
                pano_title + '_draw_resize_org_image', 
            ]
            save_image(list_frame, path="temp_files/Pano_Detection/ransac", filename=list_name, format="jpg")
        
        # stdo(1, "ransac-list_intersection:{}".format(list_intersection))
        
        stop_intersection = time.time() - start_intersection
        ###################################################
        
        ###################################################
        start_angle = time.time()
        angle_rad_top, angle_rad_bottom = get_Angles_Of_Horizontal_Lines(list_intersection)
        stop_angle = time.time() - start_angle
        ###################################################
        
        ###################################################
        start_rotate = time.time()
        
        if object_color == 'white' or object_color == 'gray - siyah sembol' or object_color == 'gray - beyaz sembol':
            border_value = (255,255,255)
        elif object_color == 'black' or object_color == 'piano-black':
            border_value = (0,0,0)
        
        if flag_activate_debug_images:
            rotated = image_Rotate(draw_resize_org_image, angle_rad=angle_rad_top, border_value=border_value)
        resize_org_rotated = image_Rotate(resize_org_image.copy(), angle_rad=angle_rad_top, border_value=border_value)
        org_rotated = image_Rotate(org_image.copy(), angle_rad=angle_rad_top, border_value=border_value)
        stop_rotate = time.time() - start_rotate
        ###################################################
        
        ###################################################
        start_coordinate_scale = time.time()
        
        resize_org_rotate_coords = np.array(list_intersection).reshape(-1,2)
        resize_org_rotate_coords[:,0], resize_org_rotate_coords[:,1] = coordinate_Scaling(
            x=resize_org_rotate_coords[:,0], 
            y=resize_org_rotate_coords[:,1],
            old_w=False, old_h=False, 
            new_w=img_width, new_h=img_height,
            degree=angle_rad_top, 
            task="ANGULAR_ROTATION", 
            is_dual=False
        )
        
        org_rotate_coords = resize_org_rotate_coords.copy()
        org_rotate_coords[:,0], org_rotate_coords[:,1] = coordinate_Scaling(
            x=resize_org_rotate_coords[:,0], 
            y=resize_org_rotate_coords[:,1],
            old_w=img_width, old_h=img_height, 
            new_w=org_image.shape[1], new_h=org_image.shape[0],
            task="RESIZE", 
            is_dual=False
        )
        
        if flag_activate_debug_images:
            resize_org_points_rotated_image = resize_org_rotated.copy()
            org_points_rotated_image = org_rotated.copy()
            for id, coords in enumerate(resize_org_rotate_coords):
                draw_Circle(resize_org_points_rotated_image, center_point=(coords[0], coords[1]), radius=20, color=(0, 0, 255), thickness=-1)
            for id, coords in enumerate(org_rotate_coords):
                draw_Circle(org_points_rotated_image, center_point=(coords[0], coords[1]), radius=20, color=(0, 0, 255), thickness=-1)
                
        stop_coordinate_scale = time.time() - start_coordinate_scale
        ###################################################
        
        if flag_activate_debug_images:
            list_frame = [
                resize_org_image, draw_resize_opening_image, draw_resize_org_image, rotated, resize_org_rotated, resize_org_points_rotated_image, org_points_rotated_image
            ]
            list_name = [
                pano_title + '_0_resize_org', 
                pano_title + '_1_Ransac_resize_opening', 
                pano_title + '_2_Mask_Ransac_resize_org', 
                pano_title + '_3_Rotate_draw_resize_org', 
                pano_title + '_4_Rotate_resize_org', 
                pano_title + '_5_Rotate_resize_org_Points',
                pano_title + '_6_Rotate_org_Points'
            ]
            save_image(list_frame, path="temp_files/Pano_Detection/ransac", filename=list_name, format="jpg")
        
        stop_seq = time.time() - start_seq
        
        # stdo(1, "[{}] T:{:.2f} - def:{:.2f} - ransac:{:.2f} - intersection:{:.2f} - angle:{:.2f} - rotate:{:.2f} - cs:{:.2f}".format
        #     (
        #         "line_Detection",
        #         stop_seq,
        #         stop_def,
        #         stop_ransac,
        #         stop_intersection,
        #         stop_angle,
        #         stop_rotate,
        #         stop_coordinate_scale
        #     )
        # )
        
        return org_rotated, org_rotate_coords, angle_rad_top
        
def fill_Image(image):
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    fill = image.copy()
    cv2.floodFill(fill, mask, (11,11), 255, cv2.FLOODFILL_FIXED_RANGE)
    return fill

def calculation_Crop_Parameters_Of_Image_Displacement(image, rotated_cords, pano_sector_index='R'):
    
    points = sorted(rotated_cords, key=lambda p: p[0])  # Sort by x coordinate
    left_points = sorted(points[:2], key=lambda p: p[1])  # Top-bottom order
    right_points = sorted(points[2:], key=lambda p: p[1])  # Top-bottom order
    
    # pts = [left_points[0], right_points[0], left_points[1], right_points[1]] # TL, TR, BL, BR
    tl = left_points[0]
    tr = right_points[0]
    bl = left_points[1]
    br = right_points[1]
    
    if pano_sector_index=="R":
        left_range = max(tl[0], bl[0])
        right_range = max(tr[0], br[0])
        up_range = min(tl[1], bl[1])
        down_range = max(tr[1], br[1])
       
        displacement_left_range = max(0, right_range - 4200) #3900
        displacement_down_range = up_range + 2300
        crop = image[ up_range+50:displacement_down_range, displacement_left_range:right_range-200 ]
    
    elif pano_sector_index=="L":
        left_range = min(tl[0], bl[0])  
        right_range = min(tr[0], br[0])
        up_range = min(tl[1], bl[1])  
        down_range = max(tr[1], br[1])
 
        displacement_right_range = min(left_range + 4200, image.shape[1])
        displacement_down_range = up_range + 2300
        crop = image[ up_range+50:displacement_down_range, left_range+50:displacement_right_range ]
    
    elif pano_sector_index=="M":
        left_range = min(tl[0], bl[0])  
        right_range = min(tr[0], br[0])
        up_range = min(tl[1], bl[1])  
        down_range = max(tr[1], br[1])
 
        displacement_right_range = left_range + 4000 
        displacement_down_range = up_range + 2300
        crop = image[ up_range+50:displacement_down_range, left_range+50:displacement_right_range ]
    
    return crop

def get_Contours_and_Hull(image, canny_threshold=100):
    """
    Verilen görüntüden konturları ve konveks gövdeyi çıkarır.
   
    Parametreler:
    - image: Giriş görüntüsü (numpy array, BGR veya Grayscale olabilir)
    - canny_threshold: Canny edge detection eşiği (default: 100)
   
    Döndürür:
    - contours: Bulunan tüm konturlar (list of numpy arrays)
    - hull_list: Konveks gövdeler (list of numpy arrays)
    """
    # Eğer görüntü BGR ise griye çevir
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Gürültüyü azaltmak için bulanıklaştır
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
 
    # Canny ile kenarları bul
    canny_output = cv2.Canny(blurred, canny_threshold, canny_threshold * 2)
 
    # Konturları bul
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # Konveks gövdeleri bul
    hull_list = [cv2.convexHull(c) for c in contours]
 
    return contours, hull_list
      
def draw_Hulls_and_Bbox(image, contours, flag_activate_debug_images=False):
    if not contours:
        return image, None, None
    
    if flag_activate_debug_images:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
        max_area = 0
        max_box = None
        max_hull = None
    
    # Tüm konturları tek bir arrayde birleştir
    all_points = np.vstack(contours)
    # Birleştirilmiş konturların convex hull'unu hesapla
    merged_hull = cv2.convexHull(all_points)
    # Hull üzerinden minimum alanlı döndürülmüş dikdörtgeni al
    rect = cv2.minAreaRect(merged_hull)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # Eğer istenirse köşeleri iyileştir
    box = qualified_Edges(box) #[sol_üst, sol_alt, sağ_alt, sağ_üst]
    
    # for cnt in contours:
    #     hull = cv2.convexHull(cnt)  
    #     rect = cv2.minAreaRect(hull)  
    #     box = cv2.boxPoints(rect)  
    #     box = np.int0(box)  
    #     area = cv2.contourArea(box)  
 
    #     # En büyük alanlı hull'u seç
    #     if area > max_area:
    #         max_area = area
    #         max_box = box  
    #         max_box = qualified_Edges(max_box)
    #         max_hull = hull
           
    # if max_hull is not None:
    #     # _,cols = image.shape[:2]
    #     # [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    #     # lefty = int((-x*vy/vx) + y)
    #     # righty = int(((cols-x)*vy/vx)+y)
    #     # cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),1)
    #     cv2.drawContours(image, [max_hull], -1, (0, 255, 0), 2)  
    #     cv2.drawContours(image, [max_box], -1, (255, 0, 255), 1)  
    # return image, max_hull, max_box
    
    # Çizimleri yap
    if flag_activate_debug_images:
        cv2.drawContours(image, [merged_hull], -1, (0, 255, 0), 2)  # Yeşil: hull
        cv2.drawContours(image, [box], -1, (255, 0, 255), 1)         # Mor: bbox

    return image, merged_hull, box
 
def qualified_Edges(obb_points):
    """
    Verilen küçük resim (image) ve OBB köşe noktaları (obb_points) ile
    sol üst, sol alt, sağ alt, sağ üst şeklinde sıralar.
   
    Parametreler:
        image: (numpy array) Küçük resim (height, width, channels)
        obb_points: (numpy array) OBB köşe noktaları (4x2 matris)
   
    Dönüş:
        (numpy array) Sol üst, sol alt, sağ alt, sağ üst sıralı köşe noktaları
    """
 
    # Resmin genişliği ve yüksekliği
   
   
    # Ağırlık merkezi
    merkez = np.mean(obb_points, axis=0)
   
    # Açılar hesaplanıyor
    acilar = np.arctan2(obb_points[:, 1] - merkez[1], obb_points[:, 0] - merkez[0])
   
    # Açılara göre noktaları saat yönünde sırala
    sirali_noktalar = obb_points[np.argsort(acilar)]
   
    # Sol grup ve sağ grup belirle
    sol = sirali_noktalar[sirali_noktalar[:, 0] <= merkez[0]]
    sag = sirali_noktalar[sirali_noktalar[:, 0] > merkez[0]]
 
    # Sol üst ve sol alt ayrımı
    sol_üst = min(sol, key=lambda p: p[1])
    sol_alt = max(sol, key=lambda p: p[1])
 
    # Sağ üst ve sağ alt ayrımı
    sağ_alt = max(sag, key=lambda p: p[1])
    sağ_üst = min(sag, key=lambda p: p[1])
 
    return np.array([sol_üst, sol_alt, sağ_alt, sağ_üst])      

def get_Bbox_Position_Into_Image(image, bbox_coords):
    """
    Verilen görüntüdeki OBB köşe noktalarına göre 
    görüntünün solunda mı sağında mı olduğunu hesaplar.
   
    Parametreler:
        image: (numpy array) Giriş görüntüsü (height, width, channels)
        bbox_coords: (numpy array) OBB köşe noktaları (4x2 matris)
   
    Return: position 'left' ya da 'right'
    """
    
    # Sol üst köşe
    x_min = int(min(bbox_coords[:, 0]))
    y_min = int(min(bbox_coords[:, 1]))
    
    # Sağ alt köşe
    x_max = int(max(bbox_coords[:, 0]))
    y_max = int(max(bbox_coords[:, 1]))
    
    # OBB'nin genişliği ve yüksekliği
    width = x_max - x_min
    height = y_max - y_min

    # Görüntünün merkezi
    image_center_x = image.shape[1] / 2

    # BBox'ın merkezi (x)
    bbox_center_x = bbox_coords[:, 0].mean()

    # Sağ mı sol mu?
    position = 'left' if bbox_center_x < image_center_x else 'right'
    
    return position

def create_Mask(bbox_coords=[], image_shape=(1080,1920,3), is_line=False):
    
    h, w = image_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    
    if is_line:
        line_start = bbox_coords[0]
        line_stop = bbox_coords[1]
        line_thickness = 10
        return cv2.line(mask, line_start, line_stop, 255, line_thickness)
    else:
        array_bbox_coords = cv2.convexHull(np.array(bbox_coords))
        return cv2.fillPoly(mask, [array_bbox_coords], 255)

def sort_Bbox_Points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left])

def get_Largest_Contour_Object(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=lambda cnt: cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3])
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        eliminate_image = cv2.bitwise_and(image, mask)
    else:
        eliminate_image = image.copy()
    return eliminate_image
    
def diff_Image_Masking(org_image=None, diff_image=None, edge_image_sample=None, edge_image_ref=None):
    
    # 4. Renkli çıktı görüntüsü hazırla
    output = org_image.copy()
    
    # 5. diff_mask’teki beyaz piksellerin koordinatlarını al
    ys, xs = np.where(diff_image == 255)
    
    # 6. Her piksel edge’in içinde mi değil mi kontrol et
    list_inside = []
    list_outside = []
    
    for x, y in zip(xs, ys):
        if edge_image_sample[y, x] == 255:
            # output[y, x] = (0, 255, 0)  
            output[edge_image_sample == 255] = (255, 0, 0)
            list_inside.append([x,y])
        else:
            output[y, x] = (0, 0, 255)
            output[edge_image_sample == 255] = (255, 0, 0)
            list_outside.append([x,y])
        # if edge_image_ref[y, x] != 255:
        #     output[y, x] = (255, 0, 0) 
            
    return output, list_inside, list_outside

def edge_Image_Masking(diff_image=None, edge_image_sample=None):
    
    mask = np.zeros_like(diff_image)
    
    ys, xs = np.where(diff_image == 255)
    
    for x, y in zip(xs, ys):
        if edge_image_sample[y, x] != 255:
            mask[y, x] = 255  # Mark outside pixel in mask
            
    return mask

def get_Thickness_With_Distance_Transform(image, contours):
    
    list_thickness = []
    
    for cnt in contours:

        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
        dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
        max_thickness = dist_transform.max() * 2
        list_thickness.append(max_thickness)
    
    return list_thickness

def remove_Pixel_With_Distance_Transform(image, contours, thin_threshold=2):
    
    filtered_image = image.copy()
    list_cnt = []
    list_max_thickness = []
    
    for cnt in contours:

        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
        dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
        max_thickness = dist_transform.max() * 2 
        list_max_thickness.append(round(max_thickness, 2))
        
        if max_thickness <= thin_threshold:
            list_cnt.append(cnt)
        
    cv2.drawContours(filtered_image, list_cnt, -1, 0, thickness=cv2.FILLED)
        
    return filtered_image, list_max_thickness

def nearest_Edge(x, y, img_w, img_h):
    distances = {
        "left": x,
        "right": img_w - x,
        "top": y,
        "bottom": img_h - y
    }
    return min(distances, key=distances.get)

def draw_Arrows_With_Labels(image, coords, pad_top, pad_bottom, pad_left, pad_right, labels=[]):
    img_h, img_w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Orijinal resim boyutunu padding düşülmüş şekilde hesapla
    orig_w = img_w - (pad_left + pad_right)
    orig_h = img_h - (pad_top + pad_bottom)

    for (x, y), label in zip(coords, labels):
        # Orijinal koordinatlar (padding çıkarılmış)
        orig_x = x - pad_left
        orig_y = y - pad_top

        direction = nearest_Edge(orig_x, orig_y, orig_w, orig_h)

        # Hedef noktayı kenara göre belirle
        if direction == "left":
            start = (x-5, y)
            target = (pad_left-15, y)
            text_pos = (target[0]-38, y+5)
        elif direction == "right":
            start = (x+5, y)
            target = (img_w-pad_right+10, y)
            text_pos = (target[0] + 3, y+5)
        elif direction == "top":
            start = (x, y-8)
            target = (x, pad_top-10)
            text_pos = (x-15, target[1]-5)
        else:  # bottom
            start = (x, y+5)
            target = (x, img_h-pad_bottom+10)
            text_pos = (x-15, target[1]+15)

        cv2.arrowedLine(image, start, target, (0, 255, 0), 1, tipLength=0.3)

        label_ratio_mm = label * 0.0494 # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
        label_ratio_mm = f"{label_ratio_mm:.2f}"
        cv2.putText(image, label_ratio_mm, text_pos, font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return image

if __name__ == "__main__":
    for function in [f for f in vars(hashlib).values() if inspect.isfunction(f)]:
        print(function)
