import cv2
import numpy as np
from image_manipulation import sobel_gradient, remove_Small_Object
from image_tools import show_image
from extractor_centroid import coord
from stdo import stdo


def preprocess_of_Template_Matching(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    #smoothing = cv2.GaussianBlur(gray, (9,9), 0)
    #smoothing = cv2.blur(gray, (15,15))
    #smoothing = cv2.medianBlur(gray, 15)
    kernel = np.ones((6, 6), np.float32)/36
    smoothing = cv2.filter2D(gray, -1, kernel)
    #smoothing = cv2.bilateralFilter(gray, 9, 50, 50)

    #laplacian = cv2.Laplacian(smoothing, cv2.CV_64F, 1)
    #abs_laplacian = cv2.convertScaleAbs(laplacian)

    #alpha = 2
    #beta = 0
    #brightness = cv2.convertScaleAbs(smoothing, alpha=alpha, beta=beta)

    #sharpen = cv2.subtract(gray, smoothing)

    xp = [0, 4, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    xee = np.arange(256)
    table = np.interp(xee, xp, fp).astype('uint8')
    lut = cv2.LUT(smoothing, table)

    #histeq = cv2.equalizeHist(lut)

    dst = lut
    return dst


def preprocess_of_Count_Area(src, pattern_id='0', show_result=False, show_specified_component=0):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = sobel_gradient(src, scale=5)
    src = cv2.normalize(
        src,
        None,
        alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX
    )
    kernel = np.ones((3, 3), np.uint8)
    src = cv2.erode(src, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    if (show_result == True) and (show_specified_component == int(pattern_id)):
        show_image([src], open_order=1, figsize=((7, 7)))

    dst = src
    return dst


def preprocess_of_Fiducial_Detection(src, threshold=(50, 255), threshold_method=cv2.THRESH_BINARY, ratio=100, shape_method=False, shape_type='circle', dp=1.0, minDist=1, param1=100, param2=100, minRadius=10, maxRadius=100):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((9, 9), np.float32)/81
    smoothing = cv2.filter2D(gray, -1, kernel)
    # smoothing = cv2.GaussianBlur(gray, (15, 15), 0)

    ret, threshold_image = cv2.threshold(
        smoothing,
        threshold[0], threshold[1],
        threshold_method
    )

    if threshold_method == cv2.THRESH_BINARY:
        bw_not = cv2.bitwise_not(threshold_image)
    else:
        bw_not = threshold_image

    rso = remove_Small_Object(bw_not.copy(), ratio=ratio)[0]

    # kernel = np.ones((20, 20),np.uint8)
    # closing = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel)
    
    drawn = src.copy()
    c_cnt_r = []
    if shape_method:
        if shape_type == 'circle':
            circles = cv2.HoughCircles(
                image=rso,
                # circles=cv2.CV_32FC3,
                method=cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=param1,
                param2=param2,
                minRadius=minRadius,
                maxRadius=maxRadius
            )

        #     if circles is not None:
        #        c_cnt_r = np.round(circles[0, :-1]).astype("int")
            if circles is not None:
               circles = np.round(circles[0, :]).astype("int")
               for (x, y, r) in circles:
                   # print("(x, y), r:", (x, y), r)
                   # draw the circle in the output image, then draw a rectangle
                   # corresponding to the center of the circle
                   cv2.circle(drawn, (x, y), r, (0, 255, 0), 2)
                   cv2.rectangle(
                       drawn,
                       (x + r//2, y + r//2),
                       (x + r//2, y + r//2),
                       (0, 255, 0),
                       -1
                   )
                   c_cnt_r.append([x, y])
    else:
        drawn, bb_cnt_r, bb_ext_r, c_cnt_r, c_ext_r = coord(rso)

    return drawn, c_cnt_r
