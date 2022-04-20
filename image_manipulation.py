#cython: language_level=3, boundscheck=False
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


from tools import stdo
import cv2
# import circleTools
import numpy as np
from inspect import currentframe, getframeinfo
from image_tools import is_numpy_image


import time

from image_tools import show_image

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


def contour_Centroids(list_contour, get_bbox_centroid=False):
    contour_centroids = list()
    if get_bbox_centroid:
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


def adjust_contrast(img, contrast_factor):
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


def gamma_correction(img, gamma = 2):
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

def remove_Small_Object(src, is_chosen_max_area=True, is_contour_number_for_area=True, ratio=0, buffer_percentage=30, is_filter=True, filter_lower_ratio=7, filter_upper_ratio=253, aspect='lower', elim_coord=[]):
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
                    #print(index, "-", cv2.contourArea(cnt))
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

        
    """stdo(1, "RSO TIMES -- BUFFER: {:.3f} | DRAW: {:.3f}".format
                (
                    stop_buffer,
                    stop_draw
                )
            )"""
    
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
    
def draw_Text(image, text=[], center_point=(0,0), fontscale=1, color=(0,255,0), thickness=2):
    
    count_index = 0
    for _, text_ in enumerate(text):
        count_index += 1
        
    if count_index == 1:
        text_format = "{}".format(*text)
    else:
        text_format = "{}-{:.2f}".format(*text)
        
    cv2.putText(image,
                text_format, 
                (center_point[0], center_point[1]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness, cv2.LINE_AA
            )
    return image

def draw_Line(image, start_point, end_point, color=(255, 255, 255), thickness=-1):
    cv2.line(image, start_point, end_point, color, thickness)

def draw_Circle(image, center_point, radius=1, color=(255, 255, 255), thickness=-1):
    cv2.circle(image, center_point, radius, color, thickness)

def draw_Rectangle(image, start_point, end_point, color=(255, 255, 255), thickness=-1):
    cv2.rectangle(image, start_point, end_point, color, thickness)

def transparent_Draw(src_frame, alpha=0.5, beta=1, radius=5, pts=[], fill_color=(0,0,255)):
    frame_display = src_frame.copy()
    roi = frame_display.copy()
    
    #import pdb; pdb.set_trace()

    if len(pts) != 0:
        for center_point in pts:
            #if not center_point.all():
            #    continue
            
            draw_circle(
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
            "An error occurred while doing 'crop' image manupilation -> "
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
            "An error occurred while doing 'resize' image manupilation -> "
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
        color = configs[0]  # Custom Colour

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
            "An error occurred while doing 'expand' image manupilation -> "
            "An error occured while doing 'expand' image manupilation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def grayscale(img, configs):
    # TODO: Add CV2 convertion
    #return rgb2gray(img)  # convert to grey to reduce details
    return -1

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
            "An error occurred while doing 'convert_RGB' image manupilation -> Grayscale image arrived.",
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
            "An error occured while doing 'convert_RGB' image manupilation -> Invalid image channel size",
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
            "An error occured while doing 'invert_color' image manupilation -> "
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
            "An error occured while doing 'bilateral_blur' image manupilation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

def median_blur(img, configs):
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.medianBlur(img, int(configs[0]))  # Blur to reduce noise
    else:
        return cv2.medianBlur(img, 3)  # Blur to reduce noise
    """
    except Exception as error:
        stdo(
            3,
            "An error occured while doing 'median_blur' image manupilation -> "
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
            "An error occured while doing 'gaussian_blur' image manupilation -> "
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
                "Trying different 'gaussian_blur' image manupilation parameters: Current{} -> Recovery{}".format(
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
                "An error occured while doing 'gaussian_blur' image manupilation recovery action -> "
                + error.__str__(),
                getframeinfo(currentframe()),
            )
            return img
    """

def canny_edge_detection(img, configs):
    # Canny edge detection.
    #try:
    if configs != "" and configs != [""] and configs != []:
        return cv2.Canny(img, int(configs[0]), int(configs[1]))
    else:
        return cv2.Canny(img, 100, 120)
    """
    except Exception as error:
        stdo(
            3,
            "An error occured while doing 'canny_edge_detection' image manupilation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
def threshold(img, configs):

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
            "An error occured while doing 'threshold' image manupilation -> "
            + error.__str__(),
            getframeinfo(currentframe()),
        )
        return img
    """

# https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html#gsc.tab=0
def contours(img, configs):
    #try:
    """
    if configs != "" and configs != [""] and configs != []:
        return img
    else:
    """

    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # I tried threshold but canny_edge_detection better for more optimized output
    cEDImage = canny_edge_detection(img, [])

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
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img

    """
    except Exception as error:
        stdo(
            3,
            "An error occured while doing 'contours' image manupilation -> "
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
            "An error occured while doing 'label_connected' image manupilation -> "
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
    if abs(m["mu02"]) < 1e-2:  # If there is no need to deskewing, return original image
        return img

    if configs == [0, 0]:  # Auto re-Deskew mode off - One time Deskewing
        deskewCount = 1
        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)

    elif configs == [1, 0]:  # Auto re-Deskew mode on - Multi Deskewing
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
    ):  # Custom number of re-Deskew mode on - Multi Deskewing with given number
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
            """   '- Wrong configurations for deskewing - configs: {}""".format(
                configs
            ),
        )
        return img

    stdo(1, """   '- {} number of Deskewing applied to image""".format(deskewCount))

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

def sharpening(image, kernel_size=(5, 5), alpha=1.5, beta=-0.7, gamma=0, over_run = 0):
    image_blurred = cv2.GaussianBlur(image, kernel_size, 20)
    image_sharpened = cv2.addWeighted(image, alpha, image_blurred, beta, gamma, image_blurred)

    for _ in range(over_run):
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
    1. Method Extraction of Coefficient with Repetitional Values at Sequence and Optimizing Non Tolerance Values;
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

def manipulate(img, preprocess, configs):

    if preprocess == "" or preprocess == [] or preprocess == [""]:
        stdo(2, "Image manupilation failed because of no option")
        return img

    else:
        manipulatedImg = img  # For backup

        switcher = {
            "draw_circle": draw_circle,
            "draw_rectangle": draw_rectangle,
            # "fast_fourier_transform": fast_fourier_transform,
            "erosion": erosion,
            "dilation": dilation,
            "crop": crop,
            "resize": resize,
            "expand": expand,
            "grayscale": grayscale,
            "convert_RGB": convert_RGB,
            "invert_color": invert_color,
            "bilateral_blur": bilateral_blur,
            "median_blur": median_blur,
            "gaussian_blur": gaussian_blur,
            "canny_edge_detection": canny_edge_detection,
            "threshold": threshold,
            "contours": contours,
            "label_connected": label_connected,
            "deskew": deskew,
            # "detectCircles": detectCircles,
        }
        parameter_contour = {
            "draw": 3,
            # "fast_fourier_transform": 0,
            "erosion": 3,
            "dilation": 3,
            "crop": 3,
            "resize": 2,
            "expand": 3,
            "grayscale": 0,
            "convert_RGB": 0,
            "invert_color": 0,
            "bilateral_blur": 3,
            "median_blur": 1,
            "gaussian_blur": 3,
            "canny_edge_detection": 2,
            "threshold": 2,
            "contours": 0,
            "label_connected": 0,
            "deskew": 2,
            "detectCircles": 0,
        }

        for prp in preprocess:  # Do all manipulations given as argument (preprocess)
            stdo(1, "Image Manipulating with {0} preprocess option.".format(str(prp)))

            manipulation = switcher.get(prp)

            sConfigs = []  # Specific configs of one function manupilation
            pC = parameter_contour.get(prp)  # Get Parameter number

            if len(configs) != 0:
                for i in range(pC):
                    sConfigs.append(configs[i])

            stdo(
                1,
                "'{0}' image manupilation started with {1} configs".format(
                    str(prp), str(sConfigs)
                ),
            )
            manipulatedImg = manipulation(manipulatedImg, sConfigs)

            if len(configs) != 0:
                for i in range(pC):
                    configs.pop(0)

        return manipulatedImg

def color_Range_Mask(
        img, 
        color_palette_lower=(0, 0, 0), 
        color_palette_upper=(255, 255, 255), 
        is_HSV=False,
        get_Max=False 
    ):
    
    if is_HSV:
        color_tranformed = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        color_tranformed = img
        
    mask = cv2.inRange(color_tranformed, color_palette_lower, color_palette_upper)
    
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

def image_Undistortion(image, undistortion_camera_matrix, undistortion_distortion_coefficients, undistortion_new_camera_matrix):
    dst = cv2.undistort(
            image,
            undistortion_camera_matrix, 
            undistortion_distortion_coefficients, 
            None, 
            undistortion_new_camera_matrix
        )
    return dst

def image_Perspective_Warping(image, homography_matrix):
    dst = cv2.warpPerspective(
            image, 
            homography_matrix, 
            (image.shape[1], image.shape[0])
        )
    return dst

if __name__ == "__main__":
    for function in [f for f in vars(hashlib).values() if inspect.isfunction(f)]:
        print(function)
