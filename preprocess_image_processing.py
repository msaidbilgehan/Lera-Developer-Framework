import cv2
import numpy as np
import time

from image_manipulation import sobel_gradient, sharpening, remove_Small_Object, threshold, median_blur, gaussian_blur, adjust_Contrast, gamma_Correction, grayscale_Conversion, look_Up_Table, create_Circular_Mask, draw_Rectangle, contour_Areas, fill_Image, sort_Bbox_Points
from image_tools import show_image, save_image
from math_tools import coordinate_Scaling
from extractor_centroid import detect_Centroid_Of_Segmented_Symbols
from stdo import stdo


def preprocess_of_Template_Matching(src, method=1, pattern_id='0', configuration_id='0', counter=0, flag_activate_debug_images=False):
    
    if method == 1:
        smoothing_mb = cv2.medianBlur(src, 3)
        color = cv2.applyColorMap(smoothing_mb, cv2.COLORMAP_HOT)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, smoothing_mb, color, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_smoothing_mb", 
                str(configuration_id) + "_" + str(pattern_id) +"_3_color", 
                str(configuration_id) + "_" + str(pattern_id) +"_4_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-1", filename=filename, format="png")
        
    elif method == 2:    
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
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, gray, smoothing, lut]
            filename = [
                str(configuration_id)+ "_" + str(pattern_id) + "_1_src", 
                str(configuration_id)+ "_" + str(pattern_id) + "_2_gray", 
                str(configuration_id)+ "_" + str(pattern_id) + "_3_smoothing", 
                str(configuration_id)+ "_" + str(pattern_id) + "_4_lut"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-2", filename=filename, format="png")
            
    elif method == 3:
        smoothing_mb = cv2.medianBlur(src, 3)
        color = cv2.applyColorMap(smoothing_mb, cv2.COLORMAP_JET)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, smoothing_mb, color, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_smoothing_mb", 
                str(configuration_id) + "_" + str(pattern_id) +"_3_color", 
                str(configuration_id) + "_" + str(pattern_id) +"_4_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-3", filename=filename, format="png")
    
    elif method == 4:
        smoothing_mb = cv2.medianBlur(src, 3) #cv2.GaussianBlur(src, (3,3), 0) #
        color = cv2.applyColorMap(smoothing_mb, cv2.COLORMAP_HSV)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, smoothing_mb, color, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_smoothing_mb", 
                str(configuration_id) + "_" + str(pattern_id) +"_3_color", 
                str(configuration_id) + "_" + str(pattern_id) +"_4_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-4", filename=filename, format="png")
    
    elif method == 5:
        smoothing_mb = cv2.medianBlur(src, 3)
        gray = cv2.cvtColor(smoothing_mb, cv2.COLOR_BGR2GRAY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, smoothing_mb, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_smoothing_mb",
                str(configuration_id) + "_" + str(pattern_id) +"_3_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-5", filename=filename, format="png")
    
    elif method == 6:
        color = cv2.applyColorMap(src, counter) # COLORMAP_HSV -> 9, COLORMAP_HOT -> 11, COLORMAP_JET -> 2
        sharp = sharpening(color, kernel_size=(3, 3), alpha=1.5, beta=-0.5, gamma=0, iteration=5)
        gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, sharp, color, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_sharp",
                str(configuration_id) + "_" + str(pattern_id) +"_3_color",
                str(configuration_id) + "_" + str(pattern_id) +"_4_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-6", filename=filename, format="png")
    
    elif method == 7:
        ac = adjust_Contrast(src, method='Stretching', alpha=1.1, beta=0)
        color = cv2.applyColorMap(ac, counter)
        sharp = sharpening(color, kernel_size=(3, 3), alpha=1.5, beta=-0.5, gamma=0, iteration=1)
        gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        dst = gray
        
        if flag_activate_debug_images:
            list_temp_save_image = [src, ac, color, sharp, gray]
            filename = [
                str(configuration_id) + "_" + str(pattern_id) +"_1_src", 
                str(configuration_id) + "_" + str(pattern_id) +"_2_ac",
                str(configuration_id) + "_" + str(pattern_id) +"_3_color",
                str(configuration_id) + "_" + str(pattern_id) +"_4_sharp",
                str(configuration_id) + "_" + str(pattern_id) +"_5_gray"
            ]
            save_image(list_temp_save_image, path="temp_files/preprocess_of_Template_Matching/method-7", filename=filename, format="png")
    
    return dst


def preprocess_of_Component_Direction(image, method='C', configuration_id='0', counter=0, flag_activate_debug_images=False):
    
    if method == 'C':
        image = adjust_Contrast(image, method='Adjust', alpha=1, beta=0)
        image_gray = grayscale_Conversion(image)
        image_lut, _ = look_Up_Table(image_gray, down_table=[0, 4, 128, 192, 255], up_table=[0, 16, 128, 240, 255], is_gray_scale=True) 
        image_sobel = sobel_gradient(image_lut, scale=1)
        
        image_blur = cv2.blur(image_gray, (3,3))
        image_equhist = cv2.equalizeHist(image_blur)
        
        mask = create_Circular_Mask(image.shape[0], image.shape[1])
        image_gray_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
        image_sobel_masked = cv2.bitwise_and(image_sobel, image_sobel, mask=mask)
        image_equhist_masked = cv2.bitwise_and(image_equhist, image_equhist, mask=mask)
        
        if flag_activate_debug_images:
            image_pack = [image, image_gray, image_lut, image_sobel, image_gray_masked, image_sobel_masked, image_equhist_masked]
            title_pack = [
                str(counter) + "_0_image_" + configuration_id, 
                str(counter) + "_1_image_gray_" + configuration_id, 
                str(counter) + "_2_image_lut_" + configuration_id, 
                str(counter) + "_3_image_sobel_" + configuration_id, 
                str(counter) + "_4_image_gray_masked_" + configuration_id, 
                str(counter) + "_5_image_sobel_masked_" + configuration_id, 
                str(counter) + "_6_image_equhist_masked_" + configuration_id
            ]
            save_image(image_pack, path="temp_files/preprocess_of_Component_Direction/"+method, filename=title_pack, format="png")
        
        return image_gray_masked, image_sobel_masked, image_equhist_masked
    
    elif method == 'K':
        image_gray = grayscale_Conversion(image)
        image_medianblur = median_blur(image_gray, configs=[3])
        image_threshold = threshold(image_medianblur.copy(), configs=[230, 255, cv2.THRESH_BINARY])
        # image_blur = cv2.blur(image_gray, (3,3))
        # image_equhist = cv2.equalizeHist(image_blur)
        
        if flag_activate_debug_images:
            image_pack = [image, image_gray, image_medianblur, image_threshold]
            title_pack = [
                str(counter) + "_0_image_" + configuration_id, 
                str(counter) + "_1_image_gray_" + configuration_id, 
                str(counter) + "_2_image_image_medianblur_" + configuration_id, 
                str(counter) + "_3_image_image_threshold_" + configuration_id
            ]
            save_image(image_pack, path="temp_files/preprocess_of_Component_Direction/"+method, filename=title_pack, format="png")
            
        return -1, -1, image_threshold


def preprocess_of_Component_Size(image, method='C', configuration_id='0', counter=0, threshold_params=100, flag_activate_debug_images=False):
    
    if method == 'K':
        image_gray = grayscale_Conversion(image)
        image_medianblur = median_blur(image_gray, configs=[3])
        image_threshold = threshold(image_medianblur.copy(), configs=[threshold_params, 255, cv2.THRESH_BINARY])
        # image_blur = cv2.blur(image_gray, (3,3))
        # image_equhist = cv2.equalizeHist(image_blur)
        
        if flag_activate_debug_images:
            image_pack = [image, image_gray, image_medianblur, image_threshold]
            title_pack = [
                str(counter) + "_0_image_" + configuration_id, 
                str(counter) + "_1_image_gray_" + configuration_id, 
                str(counter) + "_2_image_image_medianblur_" + configuration_id, 
                str(counter) + "_3_image_image_threshold_" + configuration_id
            ]
            save_image(image_pack, path="temp_files/preprocess_of_Component_Size/"+method, filename=title_pack, format="png")
            
        return -1, -1, image_threshold
    
    
def preprocess_of_Count_Area(image, configuration_id='0', method='1', threshold_params=100, morph_kernel=[3,3], rso_ratio=1, counter=0, show_result=False, show_specified_component=0):
    
    if method == '1':
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_sobel = sobel_gradient(image_gray, scale=5)
        image_norm = cv2.normalize(
            image_sobel,
            None,
            alpha=0, beta=1,
            norm_type=cv2.NORM_MINMAX
        )
        kernel = np.ones((3, 3), np.uint8)
        image_morph = cv2.erode(image_norm, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        image_result = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, kernel)

        if show_result:
            image_pack = [image, image_gray, image_sobel, image_norm, image_result]
            title_pack = [
                str(configuration_id) + "_0_image_" + str(counter), 
                str(configuration_id) + "_1_image_gray_" + str(counter), 
                str(configuration_id) + "_2_image_sobel_" + str(counter), 
                str(configuration_id) + "_3_image_norm_" + str(counter),
                str(configuration_id) + "_4_image_morph_" + str(counter)
            ]
            save_image(image_pack, path="temp_files/preprocess_of_Count_Area/"+method, filename=title_pack, format="png")
    
    elif method == '2':
        image_mb = median_blur(image, configs=[3])
        image_gray = grayscale_Conversion(image_mb)
        image_s = sharpening(image_gray, kernel_size=(3, 3), alpha=1.5, beta=-0.7, gamma=0, over_run=0)
        image_th = threshold(image_s.copy(), configs=[threshold_params, 255, cv2.THRESH_BINARY])
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel[0], morph_kernel[1]))
        morph_open = cv2.morphologyEx(image_th, cv2.MORPH_OPEN, kernel)
        rso, _, all_contour_area, _, _ = remove_Small_Object(morph_open, ratio=rso_ratio)
        # stdo(1, "[{}]: Matched Area: {}".format(counter, all_contour_area))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_close = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_open_2 = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
        
        fill = morph_open_2.copy()
        h, w = image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
        fill_inv = cv2.bitwise_not(fill)
        
        combine = morph_open_2 | fill_inv
        
        image_result, _, all_contour_area, _, _ = remove_Small_Object(combine, ratio=rso_ratio)
        # stdo(1, "[{}]: Matched Area: {}".format(counter, all_contour_area))
        
        disp, bb_cnt, _, _, _ = detect_Centroid_Of_Segmented_Symbols(image_result, configuration_id=configuration_id, flag_activate_debug_images=show_result)
        
        bb_cnt_norm = list()
        bb_cnt_elim = list()
        for coords in bb_cnt:
            if (coords[0] == 0) or (coords[1] == 0) or (image_result.shape[1]-2 < coords[0]+coords[2] == image_result.shape[1]) or (image_result.shape[0]-2 < coords[1]+coords[3] == image_result.shape[0]):
                bb_cnt_elim.append(coords)
            else:
                bb_cnt_norm.append(coords)

        mask = image_result.copy()
        for coords in bb_cnt_elim:
            mask = draw_Rectangle(
                mask, 
                start_point=(coords[0], coords[1]), 
                end_point=(coords[0]+coords[2], coords[1]+coords[3]), 
                color=(0, 0, 0), thickness=-1
            )
        
        if len(bb_cnt_norm) > 0:
            max_matched_frame_coords = bb_cnt_norm[0]
        else:
            max_matched_frame_coords = bb_cnt_norm
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt_areas = contour_Areas(contours)
            max_matched_ratio = max(np.array(cnt_areas))
        else:
            max_matched_ratio = 0

        if show_result:
            image_pack = [image, image_gray, image_s, image_th, morph_open, rso, morph_close, morph_open_2, combine, image_result, disp, mask]
            title_pack = [
                str(configuration_id) + "_0_image_" + str(counter), 
                str(configuration_id) + "_1_image_gray_" + str(counter), 
                str(configuration_id) + "_2_image_sharp_" + str(counter), 
                str(configuration_id) + "_3_image_th_" + str(counter),
                str(configuration_id) + "_4_image_morph_open_" + str(counter),
                str(configuration_id) + "_5_image_rso1_" + str(counter),
                str(configuration_id) + "_6_image_morph_close_" + str(counter),
                str(configuration_id) + "_7_image_morph_open_2_" + str(counter),
                str(configuration_id) + "_8_image_combine_" + str(counter),
                str(configuration_id) + "_9_image_rso2_" + str(counter),
                str(configuration_id) + "_10_image_disp_" + str(counter),
                str(configuration_id) + "_11_image_mask_" + str(counter)
            ]
            save_image(image_pack, path="temp_files/preprocess_of_Count_Area/"+method, filename=title_pack, format="png")
        
    return max_matched_ratio, max_matched_frame_coords


def preprocess_of_Fiducial_Detection(
        src, 
        threshold=(50, 255), 
        threshold_method=cv2.THRESH_BINARY, 
        ratio=100, 
        shape_method=False, 
        shape_type='circle', 
        dp=1.0, 
        minDist=1, 
        param1=100, 
        param2=100, 
        minRadius=10, 
        maxRadius=100
    ):
    
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
        drawn, bb_cnt_r, bb_ext_r, c_cnt_r, c_ext_r = detect_Centroid_Of_Segmented_Symbols(rso)

    return drawn, c_cnt_r


def preprocess_of_First_Masking_Laser_Printing(
        src_frame, 
        object_view, 
        object_color, 
        crop_parameters, 
        is_middle_object=False, 
        middle_object_roi=[], 
        bi_surface_bigger_middle=False, 
        is_resized=False, 
        is_elim=True, 
        elim_roi=[], 
        first_crop_kernel=80, 
        first_crop_threshold=[], 
        is_label=False, 
        gpu_obj=None,
        activate_debug_images=False
    ):
    
        start = time.time()
        
        if is_resized:
            crop_parameters = np.array(crop_parameters)
            crop_parameters = crop_parameters.reshape(-1,2)
            crop_parameters[:,0], crop_parameters[:,1] = coordinate_Scaling(
                crop_parameters[:,0], crop_parameters[:,1],
                5472, 3648,
                src_frame.shape[1], src_frame.shape[0],
                task='RESIZE',
                is_dual=False
            )
            crop_parameters = crop_parameters.reshape(-1)
        
        # cropped_frame = src_frame[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3], :]
        cropped_frame = src_frame
        mask_pre = np.zeros((src_frame.shape[:2]), np.uint8)
        
        # print(cropped_frame.shape, "|", crop_parameters)

        if (object_color == 'white') or (object_color == 'gray'):
            hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            xp = [0, 10, 100, 90, 255]
            fp = [0, 5, 100, 255, 255]

            xee = np.arange(256)
            table = np.interp(xee, xp, fp).astype('uint8')
            lut = cv2.LUT(hsv, table)

            lower_red = (0,0,60)
            upper_red = (255,100,255)
            color_range = cv2.inRange(lut, lower_red, upper_red)
            
            #_, th = cv2.threshold(color_range.copy(), first_crop_threshold[0], first_crop_threshold[1], eval(first_crop_threshold[2]))
            th = threshold(
                color_range.copy(), 
                configs=[first_crop_threshold[0], first_crop_threshold[1], first_crop_threshold[2]]
            )

            rso, _, _, _, _ = remove_Small_Object(
                th.copy(),
                ratio= 30000,
            )
            
            kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
            closing = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel_closing)
            
            kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT, (first_crop_kernel,first_crop_kernel)) #resized=16,16#
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)

            if gpu_obj:
                gpu_obj.upload(rso)
                gpu_obj.morph_Operations(method=cv2.MORPH_CLOSE, kernel=kernel_closing, iterations=1)
                gpu_obj.morph_Operations(method=cv2.MORPH_OPEN, kernel=kernel_opening, iterations=1)
                opening = gpu_obj.download()

            else:
                closing = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel_closing)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)

            
            if is_middle_object:
                mask_cekmece = closing.copy()
                #mask_cekmece[300:1080, :] = 255

                if bi_surface_bigger_middle:
                    mask_cekmece[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 255
                    mask_pre[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]] = mask_cekmece
                    mask = cv2.bitwise_and(src_frame, src_frame, mask=mask_pre)
                    canny = cv2.Canny(mask_pre, 150, 200, None, 3)
                    mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_cekmece)

                    #show_pack = [hsv, lut, color_range, th, rso, closing, opening, mask_cekmece, mask_pro]
                    #title = ["hsv", "lut", "color_range", "th", "rso", "closing", "opening", "mask_cekmece", "mask_pro"]
                    #show_image(show_pack, title, open_order=3)
                    
                else:
                    mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask_cekmece)
                    
                    if is_label:
                        fill = mask_cekmece.copy()
                        mask = np.zeros((mask_cekmece.shape[0]+2, mask_cekmece.shape[1]+2), np.uint8)
                        cv2.floodFill(fill, mask, (0,0), 255, 1, 2, 0)
                        # fill_inv = cv2.bitwise_not(fill)
                        temp = fill
                    else:
                        temp = mask_cekmece
                
                    mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=temp)

                    mask_pre[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]] = temp
                    mask = cv2.bitwise_and(src_frame, src_frame, mask=mask_pre)
                    canny = cv2.Canny(mask_pre, 150, 200, None, 3)
                    
                    temp = cv2.bitwise_not(temp)
                    frankeisthine = cv2.add(closing, temp) 

                    mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=frankeisthine)

                    list_temp_save_image = [cropped_frame, hsv, lut, color_range, th, rso, closing, opening, temp, mask_pro, mask_cekmece, frankeisthine, cropped_frame]
                    filename = [
                        "0cropped_frame", "1hsv", "2lut", "3color_range", "4th", 
                        "5rso", "6closing", "7opening", "8temp", 
                        "9mask_pro", "10mask_cekmece", "11frankeisthine", "12cropped_frame"
                    ]
                    save_image(list_temp_save_image, path="temp_files/preprocess_of_First_Masking_Laser_Printing", filename=filename, format="jpg")
                    
                stop = time.time() - start
                
                stdo(1, "T:{:.2f} ms".format(stop))

                return canny, mask, mask_pro

            fill = opening.copy()
            mask = np.zeros((opening.shape[0]+2, opening.shape[1]+2), np.uint8)
            cv2.floodFill(fill, mask, (0,0), 255, 1, 2, 0)
            fill_inv = cv2.bitwise_not(fill)
            
            if not object_view:
                rso2 = remove_Small_Object(
                    fill_inv.copy(),
                    ratio = 100000,
                    aspect = 'upper'
                )[0]
            else:
                rso2 = remove_Small_Object(
                    fill_inv.copy(),
                    ratio = 10000,
                    aspect = 'lower'
                )[0]
            
            combine = opening | rso2
            
            
            mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=combine)
            
            
            mask_pre[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3]] = combine
            
            mask = cv2.bitwise_and(src_frame, src_frame, mask=mask_pre)
            
            canny = cv2.Canny(mask_pre, 150, 200, None, 3)
            
            #show_pack = [hsv, lut, color_range, th, rso, closing, opening, fill_inv, rso2, combine, mask_pro, mask]
            #save_image(show_pack, path="temp_files/preprocess_of_First_Masking_Laser_Printing/", format="png")
            #show_image(show_pack, open_order=3)
            
        elif (object_color == 'black') or (object_color == 'piano-black'):
            
            gamma = 2
            lookUpTable = np.empty((1,256), np.uint8)
            for j in range(256):
                lookUpTable[0,j] = np.clip(pow(j / 255.0, float(gamma)) * 255.0, 0, 255)
            gc = cv2.LUT(cropped_frame, lookUpTable)
            
            gray = cv2.cvtColor(gc, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0,255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            
            mask = np.zeros((cropped_frame.shape[:2]), dtype='uint8')
            if is_elim:
                elim_coord_xmin = elim_roi[2]
                elim_coord_xmax = elim_roi[3]
                elim_coord_ymin = elim_roi[0]
                elim_coord_ymax = elim_roi[1]
                elim_coord = [elim_coord_xmin, elim_coord_xmax, elim_coord_ymin, elim_coord_ymax]
                
                rso = remove_Small_Object(
                    th.copy(),
                    ratio= 1,
                    aspect = 'lower',
                    elim_coord=elim_coord
                )[0]
                
                mask[elim_roi[0]:elim_roi[1], elim_roi[2]:elim_roi[3]] = 255
                
            
            rso = th
            
            mask_pro = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
            canny = cv2.Canny(rso, 150,200, None, 3)

            if activate_debug_images:
                list_temp_save_image = [th, rso, mask_pro, cropped_frame]
                filename = ["4th", "5rso", "9mask_pro", "12cropped_frame"]
                save_image(list_temp_save_image, path="temp_files/preprocess_of_First_Masking_Laser_Printing/black", filename=filename, format="png")
            
        return rso, mask, mask_pro


def preprocess_of_Extractor_Difference_Laser_Printing(
        image, 
        method=1, 
        is_sharp=True, 
        is_sobel=True, 
        is_threshold=True, 
        window=False, 
        sharp_scale=3, 
        open_order=1, 
        sobel_scale=0.05, 
        threshold_config=[-1, -1], 
        is_middle_object=False, 
        middle_object_roi=[],
        pano_title='',
        pano_sector='',
        counter=0,
        activate_debug_images=False,
        is_adjust_contrast=False,
        is_gamma_correction=False,
    ):
    
    # GRAYSCALE
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
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
    else:
        image_sharpened = image.copy()

    # SOBEL
    if is_sobel and sobel_scale != 0:
        image_sharpened_sobel = sobel_gradient(image_sharpened, sobel_scale)
    else:
        image_sharpened_sobel = image_sharpened.copy()

    # THRESHOLD
    if is_threshold:
        image_sharpened_sobel_threshold = threshold(image_sharpened_sobel.copy(), configs=[0, 255, 'cv2.THRESH_OTSU'])
        # FOR Strecth
        # image_sharpened_sobel_threshold = cv2.adaptiveThreshold(image_sharpened_sobel.copy(), threshold_config[0], cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_config[2], 251, 33)
    else:
        image_sharpened_sobel_threshold = image_sharpened_sobel.copy()
    
    # if is_middle_object:
    #     mask_cekmece = image_sharpened_sobel_threshold.copy()
        
    #     mask_cekmece[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = 0

    #     image_sharpened_sobel_threshold = mask_cekmece
        
    #     if activate_debug_images:
    #         rrggbb = cv2.cvtColor(image_sharpened_sobel_threshold.copy(), cv2.COLOR_GRAY2BGR)
    #         cv2.rectangle(rrggbb, (middle_object_roi[2], middle_object_roi[0]), (middle_object_roi[3], middle_object_roi[1]), (0,255,0), 3)
            
    
    if activate_debug_images:
        rrggbb = cv2.cvtColor(image_sharpened_sobel_threshold.copy(), cv2.COLOR_GRAY2BGR)
        list_temp_save_image = [image, image_sharpened, image_sharpened_sobel, image_sharpened_sobel_threshold, rrggbb]
        filename = [
            pano_title+"_"+pano_sector+"_"+"1_image", 
            pano_title+"_"+pano_sector+"_"+"2_image_sharpened", 
            pano_title+"_"+pano_sector+"_"+"3_image_sharpened_sobel", 
            pano_title+"_"+pano_sector+"_"+"4_image_sharpened_sobel_threshold",
            pano_title+"_"+pano_sector+"_"+"5_rrggbb_middle_object_roi",
        ]
        save_image(list_temp_save_image, path="temp_files/preprocess_of_Extractor_Difference_Laser_Printing", filename=filename, format="jpg")

    return image_sharpened_sobel_threshold, image_sharpened, image_sharpened_sobel


def preprocess_of_Symbol_Centroid_Laser_Printing(
        src_frame,
        object_color='white',
        is_inlay=False, 
        control_scratch=False, 
        kernel=20, 
        gpu_obj=None, 
        pano_title="",
        flag_activate_debug_images=False
    ):
    
    # if object_color == 'white': # 15.07.2025
    #     src_frame = cv2.bitwise_not(src_frame) # 15.07.2025
    
    fill = src_frame.copy()
    h, w = src_frame.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
    fill_inv = cv2.bitwise_not(fill)
    
    combine = src_frame | fill_inv

    if control_scratch:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        combine = cv2.erode(combine, kernel_erode)

    if flag_activate_debug_images:
        image_pack = [src_frame, fill, fill_inv, combine]
        title_pack = [
            pano_title + "_0_src_frame", 
            pano_title + "_1_fill", 
            pano_title + "_2_fill_inv", 
            pano_title + "_3_combine"
        ]
        save_image(image_pack, path="temp_files/preprocess_of_Symbol_Centroid_Laser_Printing", filename=title_pack, format="jpg")
    
    return src_frame, fill_inv, combine


def preprocess_Masking_for_Pano_Localization(
        src_image=None,
        object_color='white',
        method='hough_lines',
        color_transform_type='BGR',
        dict_color_palette=None,
        gpu_obj=None,
        pano_sector='L',
        pano_title='',
        pano_familia='BM-056',
        flag_activate_debug_images=False
    ):
    
    if pano_familia == 'BM-056' or pano_familia == 'BM-075':
        if pano_sector == 'L':
            slider_object_roi = [900, 2100, 0, 340]
            src_image[slider_object_roi[0]:slider_object_roi[1], slider_object_roi[2]:slider_object_roi[3]] = [0,255,255] # Yellow mask for machine slider axis 
        elif pano_sector == 'R':
            slider_object_roi = [900, 2100, 4200, src_image.shape[1]]
            src_image[slider_object_roi[0]:slider_object_roi[1], slider_object_roi[2]:slider_object_roi[3]] = [0,255,255] # Yellow mask for machine slider axis
    elif pano_familia == 'BX':
        if pano_sector == 'L':
            slider_object_roi = [900, 2100, 0, 340]
            src_image[slider_object_roi[0]:slider_object_roi[1], slider_object_roi[2]:slider_object_roi[3]] = [0,255,255] # Yellow mask for machine slider axis 
        elif pano_sector == 'R':
            slider_object_roi = [1000, 2200, 3600, src_image.shape[1]]
            src_image[slider_object_roi[0]:slider_object_roi[1], slider_object_roi[2]:slider_object_roi[3]] = [0,255,255] # Yellow mask for machine slider axis 
    
    
    if method == 'hough_lines':
        
        blur = cv2.medianBlur(src_image, 9)
        if color_transform_type =='HSV':
            cvt = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        else:
            cvt = blur
        blur_2 = cv2.medianBlur(cvt, 9)
        
        lower = dict_color_palette[object_color][color_transform_type]['lower']
        upper = dict_color_palette[object_color][color_transform_type]['upper']
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        color_range = cv2.inRange(blur_2, lower, upper) # color_range = cv2.inRange(blur_2, (40,0,0), (130,120,255))
        rso, _, _, _, _ = remove_Small_Object(color_range.copy(), ratio=30000)
        
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        closing = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel_closing)
        
        padded = cv2.copyMakeBorder(closing, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])
        
        fill = fill_Image(padded)
        fill_inv = cv2.bitwise_not(fill)
        combine = padded | fill_inv
        
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        opening = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel_opening)
        
        gaussian = cv2.GaussianBlur(opening, (3, 3), 0)
        laplacian = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
        edge = cv2.convertScaleAbs(laplacian)
        
        src_image_padding = cv2.copyMakeBorder(src_image.copy(), 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0,0,0])
        mask = cv2.bitwise_and(src_image_padding, src_image_padding, mask=opening)
    
    elif method == 'ransac':
        start_seq = time.time()
        
        # blur = cv2.medianBlur(src_image, 9)
        if color_transform_type =='HSV':
            cvt = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        else:
            cvt = src_image
        # blur_2 = cv2.medianBlur(cvt, 9)
        
        lower = dict_color_palette[object_color][color_transform_type]['lower']
        upper = dict_color_palette[object_color][color_transform_type]['upper']
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        color_range = cv2.inRange(cvt, lower, upper)
        
        # if color_transform_type =='HSV':
        #     color_range = cv2.medianBlur(color_range, 9)
        # else:
        #     color_range = color_range
        
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        closing = cv2.morphologyEx(color_range, cv2.MORPH_CLOSE, kernel_closing)
        
        if object_color == 'white' or object_color == 'gray - siyah sembol' or object_color == 'gray - beyaz sembol':
            border_value = 0
        elif object_color == 'black' or object_color == 'piano-black':
            border_value = 255
        
        if pano_sector == 'L': # border_value = 255
            padded_fill_operation = cv2.copyMakeBorder(closing, 0,0,0,10, cv2.BORDER_CONSTANT, None, value=[border_value]) # For the fill process is issued in find blank object to seek ransac lines. Only pano L and R sector
        elif pano_sector == 'R': # border_value = 255
            padded_fill_operation = cv2.copyMakeBorder(closing, 0,0,10,0, cv2.BORDER_CONSTANT, None, value=[border_value]) # For the fill process is issued in find blank object to seek ransac lines. Only pano L and R sector
        elif pano_sector == 'M':
            padded_fill_operation = cv2.copyMakeBorder(closing, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])

        
        fill = fill_Image(padded_fill_operation)
        fill_inv = cv2.bitwise_not(fill)
        if pano_sector == 'L':
            fill_inv = fill_inv[:,:-10]
        elif pano_sector == 'R':
            fill_inv = fill_inv[:,10:]
        elif pano_sector == 'M':
            fill_inv = fill_inv[10:-10,10:-10]
        
        padded_combine = cv2.copyMakeBorder(closing, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])
        padded_fill_inv = cv2.copyMakeBorder(fill_inv, 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0])
        combine = padded_combine | padded_fill_inv
        
        start_opening = time.time()
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        opening = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel_opening)
        stop_opening = time.time() - start_opening
        
        start_rso = time.time()
        rso, _, _, _, _ = remove_Small_Object(opening.copy(), ratio=400000)
        opening = rso
        stop_rso = time.time() - start_rso
        
        start_edge = time.time()
        edge = cv2.Canny(rso, 50, 150)
        stop_edge = time.time() - start_edge
        
        src_image_padding = cv2.copyMakeBorder(src_image.copy(), 10,10,10,10, cv2.BORDER_CONSTANT, None, value=[0,0,0])
        
        start_mask = time.time()
        mask = cv2.bitwise_and(src_image_padding, src_image_padding, mask=opening)
        if pano_sector == 'L':
            mask = mask[10:-10, 10:-20]
        elif pano_sector == 'R':
            mask = mask[10:-10, 20:-10] # 15.07.2025 - Remove the padding and - white and gray R sector +10 preprocess_Touch_Screen_Masking with left crop. Because when detect angle and rotate image, there are some black pixels in the left side of the image.
        elif pano_sector == 'M':
            mask = mask[10:-10, 10:-10]
        stop_mask = time.time() - start_mask
            
        stop_seq = time.time() - start_seq
        # stdo(1, "[{}] T:{:.2f} - opening:{:.2f} | rso:{:.2f} - edge:{:.2f} - mask:{:.2f}".format
        #     (
        #         "Mask",
        #         stop_seq,
        #         stop_opening,
        #         stop_rso,
        #         stop_edge,
        #         stop_mask
        #     )
        # )
        
    if flag_activate_debug_images:
        list_frame = [
            src_image, cvt, color_range, closing, padded_fill_operation, fill, fill_inv, combine, opening, rso, edge, mask
        ]
        list_name = [
            pano_sector + "_" + str(pano_title) + "_0_org",
            # pano_sector + "_" + str(pano_title) + "_1_median_blur",
            pano_sector + "_" + str(pano_title) + "_2_hsv",
            # pano_sector + "_" + str(pano_title) + "_3_median_blur",
            pano_sector + "_" + str(pano_title) + "_4_color_range",
            pano_sector + "_" + str(pano_title) + "_6_closing",
            pano_sector + "_" + str(pano_title) + "_7_padded_fill_operation",
            pano_sector + "_" + str(pano_title) + "_8_fill",
            pano_sector + "_" + str(pano_title) + "_9_fill_inv",
            pano_sector + "_" + str(pano_title) + "_10_combine",
            pano_sector + "_" + str(pano_title) + "_11_opening",
            pano_sector + "_" + str(pano_title) + "_12_rso",
            pano_sector + "_" + str(pano_title) + "_13_edge",
            pano_sector + "_" + str(pano_title) + "_14_mask",
            
        ]
        save_image(list_frame, path="temp_files/preprocess_Masking_for_Pano_Localization", filename=list_name, format="jpg")
    
    return opening, edge, mask


def preprocess_of_Roi_Masking_Laser_Printing(
        src_image=None,
        object_color='white',
        is_middle_object=False, 
        middle_object_roi=[],
        pano_sector_index='L', 
        pano_title='',
        flag_activate_debug_images=False
    ):
    
    if src_image.shape[1] - middle_object_roi[3] < 200:
        middle_object_roi[3] = src_image.shape[1]
    if middle_object_roi[2] < 200:
        middle_object_roi[2] = 0
    if src_image.shape[0] - middle_object_roi[1] < 200:
        middle_object_roi[1] = src_image.shape[0]
    if middle_object_roi[0] < 200:
        middle_object_roi[0] = 0
    
    if (object_color == 'white') or (object_color == 'gray - siyah sembol') or (object_color == 'gray - beyaz sembol'):
        
        if is_middle_object:
            
            if pano_sector_index == 'M':
                radius = (middle_object_roi[3] - middle_object_roi[2]) // 2
                xc = middle_object_roi[2] + radius
                yc = middle_object_roi[0] + radius
                src_image = cv2.circle(src_image, (xc,yc), radius, (255,255,255), -1)
                
            else:
                src_image[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = [255,255,255]
        
    elif (object_color == 'black') or (object_color == 'piano-black'):
        
        if is_middle_object:
            
            if pano_sector_index == 'M':
                radius = (middle_object_roi[3] - middle_object_roi[2]) // 2
                xc = middle_object_roi[2] + radius
                yc = middle_object_roi[0] + radius
                src_image = cv2.circle(src_image, (xc,yc), radius, (0,0,0), -1)
                
            else:
                src_image[middle_object_roi[0]:middle_object_roi[1], middle_object_roi[2]:middle_object_roi[3]] = [0,0,0]
                
    if flag_activate_debug_images:
        list_frame = [src_image]
        list_name = [pano_sector_index + "_" + str(pano_title) + "_src_image"]
        save_image(list_frame, path="temp_files/preprocess_of_Roi_Masking_Laser_Printing", filename=list_name, format="jpg")
    
    return src_image


def preprocess_Touch_Screen_Masking(
        src_frame, 
        pano_title='ref', 
        object_color='gray', 
        threshold_config=[150,255,"cv2.THRESH_BINARY_INV"], 
        dict_color_palette={},
        dict_color_symbol_extraction={},
        pano_sector_index='',
        rso_ratio=10, 
        activate_debug_images=False
    ):

    #### #### #### #### #### ######
    # Inside-of-the-touch-screen #
    #### #### #### #### #### ######

    if object_color == 'white' or object_color == 'gray - siyah sembol' or object_color == 'gray - beyaz sembol':
        src_frame = src_frame[:-50,:]
    
    color_range = cv2.inRange(
        src_frame, 
        np.array(dict_color_palette["black"]['BGR']['lower'], dtype="uint8"), 
        np.array(dict_color_palette["black"]['BGR']['upper'], dtype="uint8")
    )
    
    # contours, _ = cv2.findContours(color_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ca1 = contour_Areas(contours)
    rso_1, _, _, _, _ = remove_Small_Object(
        color_range.copy(),
        ratio=2000000,
    )
    
    color_range_not = cv2.bitwise_not(rso_1)
    # contours, _ = cv2.findContours(color_range_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ca2 = contour_Areas(contours)
    rso_2, _, _, _, _ = remove_Small_Object(
        color_range_not.copy(),
        ratio= 1000000,
    )
    
    # stdo(1, "Touch Screen Masking - ca1 area: {} | ca2 area:{}".format(ca1, ca2))
    
    rso_2_not = cv2.bitwise_not(rso_2)
    
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    closing = cv2.morphologyEx(rso_2_not, cv2.MORPH_CLOSE, kernel_closing)
    
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    erode = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel)
    
    masked_inside_src_frame = cv2.bitwise_and(src_frame.copy(), src_frame.copy(), mask=erode)
    
    # masked_inside_src_frame_gray = cv2.cvtColor(masked_inside_src_frame, cv2.COLOR_RGB2GRAY)
    # _, masked_inside_src_frame_threshold = cv2.threshold(masked_inside_src_frame_gray, 200-threshold_config[0], 255, cv2.THRESH_BINARY)
    if object_color == 'white':
        masked_inside_src_frame_threshold = cv2.inRange(
            masked_inside_src_frame, 
            np.array([70, 70, 50], dtype="uint8"),
            np.array([255, 255, 255], dtype="uint8")
        )
    else:
        masked_inside_src_frame_threshold = cv2.inRange(
            masked_inside_src_frame, 
            np.array([80, 80, 80], dtype="uint8"), #np.array(dict_color_symbol_extraction["black"]['BGR']['lower'], dtype="uint8"), 
            np.array([255, 255, 255], dtype="uint8"), #np.array(dict_color_symbol_extraction["black"]['BGR']['upper'], dtype="uint8")
        )
    
    contours, _ = cv2.findContours(masked_inside_src_frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    draw_inside = np.ones((masked_inside_src_frame.shape[:2]), dtype=np.uint8)*255 # masked_inside_src_frame.copy()
    for cnt in contours:
        # masked_outside_src_frame = cv2.drawContours(masked_outside_src_frame, [cnt], -1, color, 3)
        masked_inside_src_frame_fillpoly = cv2.fillPoly(draw_inside, pts=[cnt], color=(0), lineType=cv2.LINE_AA)
    else:
        masked_inside_src_frame_fillpoly = draw_inside
    masked_inside_src_frame_fillpoly_not = cv2.bitwise_not(masked_inside_src_frame_fillpoly)
    
    rso_ratio_px = rso_ratio * 20.244 # Add 21.07.2025
    masked_inside_src_frame_fillpoly_not_rso, _, _, _, _ = remove_Small_Object(
        masked_inside_src_frame_fillpoly_not.copy(),
        ratio=rso_ratio_px,
        counter=0
    )
    
    if activate_debug_images:
        list_temp_save_image = [src_frame, color_range, rso_1, color_range_not, rso_2, rso_2_not, closing, opening, erode, masked_inside_src_frame, masked_inside_src_frame_fillpoly, masked_inside_src_frame_fillpoly_not, masked_inside_src_frame_fillpoly_not_rso]
        filename = [
            pano_title+"_"+pano_sector_index+"_"+"1_src_frame",
            pano_title+"_"+pano_sector_index+"_"+"2_color_range", 
            pano_title+"_"+pano_sector_index+"_"+"3_rso_1", 
            pano_title+"_"+pano_sector_index+"_"+"4_color_range_not", 
            pano_title+"_"+pano_sector_index+"_"+"5_rso_2", 
            pano_title+"_"+pano_sector_index+"_"+"6_rso_2_not",
            pano_title+"_"+pano_sector_index+"_"+"7_closing",
            pano_title+"_"+pano_sector_index+"_"+"8_opening",
            pano_title+"_"+pano_sector_index+"_"+"9_erode",
            pano_title+"_"+pano_sector_index+"_"+"10_1_masked_inside_src_frame",
            pano_title+"_"+pano_sector_index+"_"+"10_2_masked_inside_src_frame_fillpoly",
            pano_title+"_"+pano_sector_index+"_"+"10_3_masked_inside_src_frame_fillpoly_not",
            pano_title+"_"+pano_sector_index+"_"+"10_4_masked_inside_src_frame_fillpoly_not_rso",
        ]
        save_image(list_temp_save_image, path="temp_files/preprocess_Touch_Screen_Masking", filename=filename, format="png")
    
    #### #### #### #### #### ######
    #### #### #### #### #### ######
    #### #### #### #### #### ######
        
        
        
    #### #### #### #### #### #####
    # Outside-of-the-touch-screen #
    #### #### #### #### #### #####
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel)
    
    ## For detect_stain process ## 15.07.2025
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    sorted_bbox = sort_Bbox_Points(box)
    bbox_for_detect_stain = sorted_bbox
    # stdo(1, f"Touch preprocess_Touch_Screen_Masking - bbox_for_detect_stain: {sorted_bbox}")
    
    opening_not = cv2.bitwise_not(dilate)
    masked_outside_src_frame = cv2.bitwise_and(src_frame.copy(), src_frame.copy(), mask=opening_not)
    
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    draw_outside = masked_outside_src_frame.copy()
    for cnt in contours:
        masked_outside_src_frame_fillpoly = cv2.fillPoly(draw_outside, pts=[cnt], color=(255,255,255), lineType=cv2.LINE_8)
    else:
        masked_outside_src_frame_fillpoly = draw_outside
        
    # masked_outside_src_frame_fillpoly_gray = cv2.cvtColor(masked_outside_src_frame_fillpoly, cv2.COLOR_RGB2GRAY)
    # _, masked_outside_src_frame_fillpoly_threshold = cv2.threshold(masked_outside_src_frame_fillpoly_gray, threshold_config[0], 255, cv2.THRESH_BINARY_INV)
    # masked_outside_src_frame_fillpoly_threshold = threshold(masked_outside_src_frame_fillpoly_gray, configs=threshold_config)
    
    if object_color == 'white':
        masked_outside_src_frame_fillpoly_threshold = cv2.inRange(
            masked_outside_src_frame_fillpoly, 
            np.array(dict_color_symbol_extraction[object_color]['BGR']['lower'], dtype="uint8"),
            np.array(dict_color_symbol_extraction[object_color]['BGR']['upper'], dtype="uint8")
        )
    else:
        masked_outside_src_frame_fillpoly_threshold = cv2.inRange(
            masked_outside_src_frame_fillpoly, 
            np.array([0, 0, 0], dtype="uint8"), # np.array(dict_color_symbol_extraction[object_color]['BGR']['lower'], dtype="uint8"), 
            np.array([255, 100, 170], dtype="uint8") # np.array(dict_color_symbol_extraction[object_color]['BGR']['upper'], dtype="uint8") # 255,120,160
        )
    
    masked_outside_src_frame_fillpoly_threshold_for_blue_symbol = cv2.inRange(
        masked_outside_src_frame_fillpoly, 
        np.array([200, 140, 0], dtype="uint8"),
        np.array([255, 220, 100], dtype="uint8")
    )
    masked_outside_src_frame_fillpoly_combine = cv2.bitwise_or(masked_outside_src_frame_fillpoly_threshold, masked_outside_src_frame_fillpoly_threshold_for_blue_symbol)
    

    #### #### #### #### #### ######
    #### #### #### #### #### ######
    #### #### #### #### #### ######
    
    # combine = masked_inside_src_frame_fillpoly_not | masked_outside_src_frame_fillpoly_threshold # Close 21.07.2025
    combine = masked_inside_src_frame_fillpoly_not_rso | masked_outside_src_frame_fillpoly_combine # Edit 21.07.2025
    combine_not = cv2.bitwise_not(combine)
    combine_not_rgb = cv2.cvtColor(combine_not, cv2.COLOR_GRAY2BGR)
        
    if activate_debug_images:
        list_temp_save_image = [dilate, opening_not, masked_outside_src_frame, masked_outside_src_frame_fillpoly, masked_outside_src_frame_fillpoly_threshold, masked_outside_src_frame_fillpoly_threshold_for_blue_symbol, masked_outside_src_frame_fillpoly_combine, combine, combine_not, combine_not_rgb]
        filename = [
            pano_title+"_"+pano_sector_index+"_"+"11_dilate",
            pano_title+"_"+pano_sector_index+"_"+"12_opening_not",
            pano_title+"_"+pano_sector_index+"_"+"13_masked_outside_src_frame",
            pano_title+"_"+pano_sector_index+"_"+"14_masked_outside_src_frame_fillpoly",
            pano_title+"_"+pano_sector_index+"_"+"15_masked_outside_src_frame_fillpoly_threshold",
            pano_title+"_"+pano_sector_index+"_"+"16.1_masked_outside_src_frame_fillpoly_threshold_for_blue_symbol",
            pano_title+"_"+pano_sector_index+"_"+"16.2_masked_outside_src_frame_fillpoly_combine",
            pano_title+"_"+pano_sector_index+"_"+"17_combine",
            pano_title+"_"+pano_sector_index+"_"+"18_combine_not",
            pano_title+"_"+pano_sector_index+"_"+"19_combine_not_rgb",
        ]
        save_image(list_temp_save_image, path="temp_files/preprocess_Touch_Screen_Masking", filename=filename, format="png")
    
    return combine_not_rgb, bbox_for_detect_stain


def preprocess_Object_Tracking(src_image=None, threshold_config=[]):
    
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_blur(gray, configs=[3,3,0])
    thresh = threshold(blurred, threshold_config)
    return thresh