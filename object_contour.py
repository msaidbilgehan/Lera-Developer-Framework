import libs

import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from ordered_enum import OrderedEnum
from scipy.signal import find_peaks

from image_manipulation import auto_Interpolation, remove_Small_Object, contour_Centroids, threshold
from image_tools import save_image
from extractor_centroid import coord


absolute_path = os.path.dirname(os.path.abspath(__file__))

class PROCESS_TOOLS:

    def plot(self, src, cond, numofploting=1): #TODO: Remove
        if cond == 0:
            plt.figure(figsize=(10,10))
            if len(src.shape) == 3:
                plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(src, cmap='gray')
    
        else:
            plt.subplots(figsize=(10, 10))
            for i in range(len(src)):
                ax=plt.subplot(len(src), 1, i+1)
                #print(src[i].shape)
                if len(src[i][0].shape) == 3:
                    ax.imshow(cv2.cvtColor(src[i][0], cv2.COLOR_BGR2RGB))
                else:
                    ax.plot(src[i][0])
                    ax.plot(src[i][1], src[i][0][src[i][1]], "x")
        plt.show()

    def find_Max_Min_Coordinate_of_Variables(self, src_array, indicator):
        if src_array.shape[0] == 1 or len(src_array.shape) <= 2:
            if len(src_array.shape) == 3:
                x = src_array[0][:,0]
                y = src_array[0][:,1]
            else:
                x = src_array[:,0]
                y = src_array[:,1]

            
            if indicator == 'LR' or indicator == 'RL':
                index_max = np.where(y==np.max(y))[0]
                if len(index_max) > 1:
                    ymax = np.max(y)
                    xmax = np.max(np.take(x, index_max))
                else:
                    index_max = np.argmax(y)
                    ymax = np.max(y)
                    xmax = x[index_max]
                    
                index_min = np.where(y==np.min(y))[0]
                if len(index_min) > 1:
                    ymin = np.min(y)
                    xmin = np.min(np.take(x, index_min))
                else:
                    index_min = np.argmin(y)
                    ymin = np.min(y)
                    xmin = x[index_min]
            
            elif indicator == 'UD' or indicator == 'DU':
                index_max = np.where(x==np.max(x))[0]
                if len(index_max) > 1:
                    xmax = np.max(x)
                    ymax = np.max(np.take(y, index_max))
                else:
                    index_max = np.argmax(x)
                    xmax = np.max(x)
                    ymax = y[index_max]
                    
                index_min = np.where(x==np.min(x))[0]
                if len(index_min) > 1:
                    xmin = np.min(x)
                    ymin = np.min(np.take(y, index_min))
                else:
                    index_min = np.argmin(x)
                    xmin = np.min(x)
                    ymin = y[index_min]




            """index_max = np.argmax(y)
            ymax = np.max(y)
            xmax = x[index_max]

            index_min = np.argmin(y)
            ymin = np.min(y)
            xmin = x[index_min]"""

        else:
            xmax_a = []
            xmin_a = []
            ymax_a = []
            ymin_a = []

            if indicator == 'LR' or indicator == 'RL':
                for i in range(len(src_array)):
                    if len(src_array.shape) == 3:
                        x = src_array[i][:,0]
                        y = src_array[i][:,1]
                    else:
                        x = src_array[:,0]
                        y = src_array[:,1]

                    index_max = np.argmax(y)
                    ymax_a.append(np.max(y))
                    xmax_a.append(x[index_max])

                    index_min = np.argmin(y)
                    ymin_a.append(np.min(y))
                    xmin_a.append(x[index_min])

                    #print(str(i)+"- Y:", ymin_a, " \n"+ str(i)+"- X:", xmin_a)
                
                ymax_a = np.array(ymax_a); xmax_a = np.array(xmax_a); ymin_a = np.array(ymin_a); xmin_a = np.array(xmin_a)

                index_max = np.argmax(ymax_a)
                ymax = np.max(ymax_a)
                xmax = xmax_a[index_max]

                index_min = np.argmin(ymin_a)
                ymin = np.min(ymin_a)
                xmin = xmin_a[index_min]
            
            elif indicator == 'UD' or indicator == 'DU':
                for i in range(len(src_array)):
                    x = src_array[i][:,0]
                    y = src_array[i][:,1]

                    index_max = np.argmax(x)
                    xmax_a.append(np.max(x))
                    ymax_a.append(y[index_max])

                    index_min = np.argmin(x)
                    xmin_a.append(np.min(x))
                    ymin_a.append(y[index_min])

                    #print(str(i)+"- Y:", ymin_a, " \n"+ str(i)+"- X:", xmin_a)
                
                ymax_a = np.array(ymax_a); xmax_a = np.array(xmax_a); ymin_a = np.array(ymin_a); xmin_a = np.array(xmin_a)

                index_max = np.argmax(xmax_a)
                xmax = np.max(xmax_a)
                ymax = ymax_a[index_max]

                index_min = np.argmin(xmin_a)
                xmin = np.min(xmin_a)
                ymin = ymin_a[index_min]

        return xmin,ymin,xmax,ymax

    def detect_Angle(self, src_array, w, h):
        left_min = src_array[0]
        left_max = src_array[1]
        deltay = left_max[1] - left_min[1]
        deltax = left_max[0] - left_min[0]
        radian = math.atan2(deltay, deltax)
        angle = math.degrees(radian)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return M, angle

    def detect_Line_Intersection(self, line1, line2):
        for i in range(2):
            for j in range(2):
                if line1[i][j] <= 0:
                    line1[i][j] = 0
                elif line2[i][j] <= 0:
                    line2[i][j] = 0

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            at = np.dtype("int32")
            at = (a[0] * b[1]) - (a[1] * b[0])
            return at

        div = det(xdiff, ydiff)
        if div == 0:
            return False
            #raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        #print(round(x),round(y))
        return round(x), round(y)

    def remove_Small_Object(self, src, ratio, aspect_direction='smallest'): #TODO: Remove
        contours, hierarchy = cv2.findContours(src , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for i,cnt in enumerate(contours):
                print(hierarchy[0][i][2], cv2.contourArea(cnt))
                #if hierarchy[0][i][2] == -1:
                if aspect_direction == 'smallest':
                    if cv2.contourArea(cnt) < ratio:
                        cv2.drawContours(src, [cnt], 0, (0,0,0), -1)
                elif aspect_direction == 'biggest':
                    if cv2.contourArea(cnt) > ratio:
                        cv2.drawContours(src, [cnt], 0, (0,0,0), -1)
            return src
        else:
            return -1


    def extract_Signal(self, frame, startx, starty, endx, endy):
        start_point = (startx, starty)
        end_point = (endx, endy)
        x0 = start_point[0]
        y0 = start_point[1]
        x1 = end_point[0]
        y1 = end_point[1]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if x0 < x1:
            ix = 1
        else:
            ix = -1

        if y0 < y1:
            iy = 1
        else:
            iy = -1
        e = 0

        currentColor = []
        for j in range(0, dx+dy, 1):
            currentColor.append((j, frame[y0][x0]))

            e1 = e + dy
            e2 = e - dx
            if abs(e1) < abs(e2):
                x0 += ix
                e = e1
            else:
                y0 += iy
                e = e2
        currentColor = np.array(currentColor)
        return currentColor[:,1]

    def find_Saddle_Point_of_Signal(self, src_signal):
        difference_signal = np.diff(src_signal)
        max_difference_signal = np.argmax(difference_signal)
        for i in range(max_difference_signal, len(difference_signal), 1):
            if i == len(difference_signal)-1:
                continue
            if difference_signal[i] - difference_signal[i+1] == 0:
                return i

    def find_Peak_Point_of_Signal(self, src_signal, height, distance):
        peakx, _ = find_peaks(src_signal, height=height, distance=distance)
        if not peakx.any():
            return 0
        peakx = np.max(peakx)
        return peakx

    def _old_pre_Process(self, src_frame): #FOR images in the deneme_tunnel_lights folder# #TODO: Remove
        h, w = src_frame.shape[:2]
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        xp = [0, 64, 128, 190, 255]
        fp = [0, 16, 128, 240, 255]
        xee = np.arange(256)
        table = np.interp(xee, xp, fp).astype('uint8')
        lut = cv2.LUT(hsv, table)

        lower = np.array([0,50,100], np.uint8)
        upper = np.array([120,180,255], np.uint8)
        mask = cv2.inRange(lut, lower, upper)

        kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing)

        fill = closing.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(fill, mask, (0,0), 255)
        fill_inv = cv2.bitwise_not(fill)
        combine = closing | fill_inv

        canny = cv2.Canny(combine, 150,200, None, 3)
        return canny, fill_inv

    def pre_Process(self, src_frame, object_view, object_id_crop, is_resized=False, is_gpu_active=False):
        
        if is_resized == True:
            object_id_crop = np.array(object_id_crop)
            object_id_crop = object_id_crop.reshape(-1,2)
            object_id_crop[:,0], object_id_crop[:,1] = self.coordinate_Scaling \
            (
                object_id_crop[:,0], object_id_crop[:,1],
                5472, 3648,
                src_frame.shape[1], src_frame.shape[0],
                task='RESIZE',
                is_dual=False
            )
            object_id_crop = object_id_crop.reshape(-1)

        if (object_view == '1765050009-UP') or (object_view == '1765050009-SIDE'):
            if object_view == '1765050009-UP':
                cropped_frame = src_frame[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]]
                mask_pre = np.zeros((src_frame.shape), np.uint8)
                mask_pre[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]] = cropped_frame
                xp = [0, 64, 100, 90, 255]
                fp = [0, 16, 128, 240, 255]

            if object_view == '1765050009-SIDE':
                cropped_frame = src_frame[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]]
                mask_pre = np.zeros((src_frame.shape), np.uint8)
                mask_pre[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]] = cropped_frame
                xp = [0, 40, 10, 10, 255]
                fp = [0, 16, 128, 240, 255]

            h, w = src_frame.shape[:2]
            
            
            if is_gpu_active:
                #FUNC#
                """
                def hsv_Conversion(src):
                    dst = cv2.cuda.cvtColor(src, cv2.COLOR_BGR2HSV)
                
                def lut_Transform(src, table):
                    cv2.cuda.createLookUpTable(table)
                    cv2.cuda.LookUpTable.transform(src, dst)
                
                def threshold_Conversion(src, lower, higher, flags=cv2.THRESH_BINARY):
                    cv2.cuda.threshold(src, dst, lower, higher, flags=flags)
                
                def morphological_Filter(src, method, kernel, iterations=1):
                    cv2.cuda.createMorphologyFilter(method, src, kernel, iterations=1)  #NOT: dst_frame is overwriten on src_frame#
                
                def bitwise_Operations(src1, src2=None, method):
                    if method == 'bitwise_not':
                        cv2.cuda.bitwise_not(src1, dst)
                    elif method == 'bitwise_or':
                        cv2.cuda.bitwise_or(src1, src2, dst)
                
                def canny_Detector(src, low, high):
                    cv2.cuda.CannyEdgeDetector.setLowThreshold(low)
                    cv2.cuda.CannyEdgeDetector.setHighThreshold(high)
                    cv2.cuda.CannyEdgeDetector.setAppertureSize(3)
                    cv2.cuda.CannyEdgeDetector.detect(src, dst)
                
                """
                
                #SEQ#
                """
                self._obj_gpu.upload(mask_pre)
                self._obj_gpu.hsv_Conversion()
                xee = np.arange(256)
                table = np.interp(xee, xp, fp).astype('uint8')
                self._obj_gpu.lut_Transform(table)
                self._obj_gpu.threshold_Conversion(230.0, 255.0, flags=cv2.THRESH_BINARY)
                kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
                self._obj_gpu.morphological_Filter(method=cv2.MORPH_CLOSE, kernel_closing, iterations=1)
                kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
                self._obj_gpu.morphological_Filter(method=cv2.MORPH_OPEN, kernel_opening, iterations=1) 
                closing = self._obj_gpu.download()
                
                fill = opening.copy()
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(fill, mask, (0,0), 255)
                fill_inv = cv2.bitwise_not(fill)
                combine = opening | fill_inv
                canny = cv2.Canny(combine, 150,200, None, 3)
                """
            else:
                hsv = cv2.cvtColor(mask_pre, cv2.COLOR_BGR2HSV)
                xee = np.arange(256)
                table = np.interp(xee, xp, fp).astype('uint8')
                lut = cv2.LUT(hsv, table)
                gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 230,255, cv2.THRESH_BINARY)

                kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
                closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_closing)

                kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening)

                fill = opening.copy()
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(fill, mask, (0,0), 255)
                fill_inv = cv2.bitwise_not(fill)
                combine = opening | fill_inv
                            
                canny = cv2.Canny(combine, 150,200, None, 3)

        elif (object_view == '1765050262-UP') or (object_view == '1765050262-SIDE'):
            if (object_view == '1765050262-UP'):
                cropped_frame = src_frame[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]]
                mask_pre = np.zeros((src_frame.shape), np.uint8)
                mask_pre[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]] = cropped_frame
            if (object_view == '1765050262-SIDE'):
                cropped_frame = src_frame[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]]
                mask_pre = np.zeros((src_frame.shape), np.uint8)
                mask_pre[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]] = cropped_frame
            
            h, w = src_frame.shape[:2]
            hsv = cv2.cvtColor(mask_pre, cv2.COLOR_BGR2HSV)
            xp = [0, 40, 10, 10, 255]
            fp = [0, 16, 128, 240, 255]
            xee = np.arange(256)
            table = np.interp(xee, xp, fp).astype('uint8')
            lut = cv2.LUT(hsv, table)
            
            gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 230,255, cv2.THRESH_BINARY)
            if (object_view == '1765050262-UP'):
                kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(30, 30))
            if (object_view == '1765050262-SIDE'):
                kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_opening)

            fill = opening.copy()
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(fill, mask, (0,0), 255)
            fill_inv = cv2.bitwise_not(fill)
            combine = opening | fill_inv

            canny = cv2.Canny(combine, 150,200, None, 3)

        elif (object_view == '1780742500-UP') or (object_view == '1780742500-SIDE'):
            cropped_frame = src_frame[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]]
            mask_pre = np.zeros((src_frame.shape[:2]), np.uint8)
            
            h, w = cropped_frame.shape[:2]
            hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            xp = [0, 40, 50, 10, 255]
            fp = [0, 16, 128, 240, 255]
            xee = np.arange(256)
            table = np.interp(xee, xp, fp).astype('uint8')
            lut = cv2.LUT(hsv, table)

            gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)
            _,th = cv2.threshold(gray, 230,255, cv2.THRESH_BINARY)
            
            kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(70, 70))
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_opening)
            
            kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
            blackhat = cv2.morphologyEx(opening, cv2.MORPH_BLACKHAT, kernel_blackhat)
            blackhat = opening + blackhat

            """fill = blackhat.copy()
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
            fill_inv = cv2.bitwise_not(fill)

            rbo = self.remove_Small_Object(fill_inv.copy(), 500000, aspect_direction='biggest')
            if type(rbo) != int:
                print("A")
                rso = self.remove_Small_Object(rbo.copy(), 10000, aspect_direction='smallest')
                if type(rso) != int:
                    combine = blackhat | rso
                else:
                    combine = blackhat
            else:
                print("B")
                combine = self.remove_Small_Object(blackhat.copy(), 500000, aspect_direction='smallest')"""

            mask_pre[object_id_crop[0]:object_id_crop[1], object_id_crop[2]:object_id_crop[3]] = blackhat
            canny = cv2.Canny(mask_pre, 150,200, None, 3)
            
        return canny 

    def pre_Process_Machine(self, src_frame, object_view, object_color, crop_parameters, is_middle_object=False, middle_object_roi=[], bi_surface_bigger_middle=False, is_resized=False, is_elim=True, elim_roi=[], first_crop_kernel=80, first_crop_threshold=[], is_label=False, gpu_obj=None):
        
        if is_resized:
            crop_parameters = np.array(crop_parameters)
            crop_parameters = crop_parameters.reshape(-1,2)
            crop_parameters[:,0], crop_parameters[:,1] = self.coordinate_Scaling \
            (
                crop_parameters[:,0], crop_parameters[:,1],
                5472, 3648,
                src_frame.shape[1], src_frame.shape[0],
                task='RESIZE',
                is_dual=False
            )
            crop_parameters = crop_parameters.reshape(-1)
        
        cropped_frame = src_frame[crop_parameters[0]:crop_parameters[1], crop_parameters[2]:crop_parameters[3], :]
        mask_pre = np.zeros((src_frame.shape[:2]), np.uint8)
        
        print(cropped_frame.shape, "|", crop_parameters)

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
            th = threshold(color_range.copy(), configs=[first_crop_threshold[0], first_crop_threshold[1], first_crop_threshold[2]])

            rso, _, _, _, _ = remove_Small_Object(
                th.copy(),
                ratio= 30000,
            )
            
            kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)) #15,15
            closing = cv2.morphologyEx(rso, cv2.MORPH_CLOSE, kernel_closing)
            
            kernel_opening = cv2.getStructuringElement(cv2.MORPH_RECT,(first_crop_kernel,first_crop_kernel)) #resized=16,16#
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

                    list_temp_save_image = [hsv, lut, color_range, th, rso, closing, opening, temp, mask_pro, mask_cekmece, frankeisthine, cropped_frame]
                    filename = ["1hsv", "2lut", "3color_range", "4th", "5rso", "6closing", "7opening", "8temp", "9mask_pro", "10mask_cekmece", "11frankeisthine", "12cropped_frame"]
                    #show_image(show_pack, title, open_order=3)
                    save_image(list_temp_save_image, path="temp_files/pre_Process_Machine", filename=filename, format="png")
                        

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
            #show_image(show_pack, open_order=3)
            
        elif (object_color == 'black') or (object_color == 'piano-black'):
            
            gamma = 2
            lookUpTable = np.empty((1,256), np.uint8)
            for j in range(256):
                lookUpTable[0,j] = np.clip(pow(j / 255.0, float(gamma)) * 255.0, 0, 255)
            gc = cv2.LUT(cropped_frame, lookUpTable)
            
            gray = cv2.cvtColor(gc, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 200,255, cv2.THRESH_BINARY)
            
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


            list_temp_save_image = [th, rso, mask_pro, cropped_frame]
            filename = ["4th", "5rso", "9mask_pro", "12cropped_frame"]
            #show_image(show_pack, title, open_order=3)
            save_image(list_temp_save_image, path="temp_files/pre_Process_Machine/black", filename=filename, format="png")
            
            
            #show_image([cropped_frame, gc, gray, th, rso, mask, mask_pro, canny], open_order=1)
            
        return rso, mask, mask_pro


    """
    def inlay_Pre_Process(self, src_frame): #TODO: Remove
        ###################FOR SCRATCH
        xp = [0, 40, 10, 10, 255]
        fp = [0, 16, 128, 240, 255]

        h, w = src_frame.shape[:2]
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        xee = np.arange(256)
        table = np.interp(xee, xp, fp).astype('uint8')
        lut = cv2.LUT(hsv, table)
        gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)

        not_gray = cv2.bitwise_not(gray)

        sobelx = cv2.Sobel(lut, cv2.CV_8U,1,0,ksize=5)
        ######################
        
        xp = [0, 64, 100, 100, 255] #ET40
        fp = [0, 16, 128, 240, 255] #ET40

        h, w = src_frame.shape[:2]
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        xee = np.arange(256)
        table = np.interp(xee, xp, fp).astype('uint8')
        lut = cv2.LUT(hsv, table)
        gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)

        not_gray = cv2.bitwise_not(gray)

        _, inner_mask = cv2.threshold(not_gray, 40,255, cv2.THRESH_BINARY)
        #rso = self.remove_Small_Object(inner_mask.copy(), 100000, aspect_direction='smallest')
        rso = remove_Small_Object(inner_mask.copy(), 100000, buffer_percentage=99.1)[0]

        fill = rso.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(fill, mask, (0,0), 255, cv2.FLOODFILL_FIXED_RANGE)
        inlay_symbols = cv2.bitwise_not(fill)

        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
        closing = cv2.morphologyEx(inlay_symbols, cv2.MORPH_CLOSE, kernel_closing)

        combine = rso | inlay_symbols
        canny = cv2.Canny(combine, 150,200, None, 3)

        return hsv, lut, gray, inner_mask, rso, inlay_symbols, closing, combine, canny

    def inlay_Extract_Button(self, src_frame): #TODO: Remove
        xp = [0, 40, 10, 10, 255]
        fp = [0, 16, 128, 240, 255]

        h, w = src_frame.shape[:2]
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        xee = np.arange(256)
        table = np.interp(xee, xp, fp).astype('uint8')
        lut = cv2.LUT(hsv, table)
        gray = cv2.cvtColor(lut, cv2.COLOR_BGR2GRAY)

        not_gray = cv2.bitwise_not(gray)

        _, inner_mask = cv2.threshold(not_gray, 40,255, cv2.THRESH_BINARY)

        sobely = cv2.Sobel(inner_mask, cv2.CV_64F, 0,1, ksize=3)
        sobely = np.absolute(sobely)
        sobely = np.uint8(sobely)
        rso2 = remove_Small_Object(sobely.copy(), ratio=0, buffer_percentage=99)[0]
        cv2.imwrite(absolout_path + "../../test_area/test_image_results/src_frame.png", src_frame)

        return hsv, lut, gray, not_gray, inner_mask, sobely, rso2

    def corner_Detection(self, src, Xapproachx, Xapproachy, Xindicator, Yapproachx, Yapproachy, Yindicator): #TODO: Remove
        norm_x = 0
        norm_y = 0
        ret_y = False
        count_y = 0
        count_x = 0 
        index_x = []
        index_y = []
        exit = False
        for i in range(Xapproachx, Xapproachy, Xindicator):
            for j in range(Yapproachx, Yapproachy, Yindicator):
                if ret_y is False:
                    if (src[j][i] == 255):
                        norm_y = i,j
                        ret_y = True
                        break

                if ret_y is True:
                    if (src[j][i] == 255):
                        index_y.append(j)
                        if (len(index_y) >= 2):
                            if (index_y[-1] == index_y[-2]):
                                count_y += 1
                                break
                            else:
                                index_x.append(i)
                                if (len(index_x) >= 2):
                                    if abs(index_x[-1] - index_x[-2]) >= 5:
                                        norm_x = index_x[-1], j
                                        count_x += 1
                                        exit = True
                        break
            if exit is True:
                break

        x = norm_y[0]
        y = norm_x[1]
        return x, y
    """


    def inlay_Extract_Symbols(self, src_frame, threshold_config=[], object_color='white', counter=0, is_middle_object=False, middle_object_roi=[], middle_object_roi_2=[], kernel=50):
        gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
        if object_color == 'white':
            k_th = 40
        elif (object_color == 'gray') or (object_color == 'black'):
            #k_th = 50 # eski panolar
            k_th = 40 #60 09.07.2021 eski gray panolar
        elif object_color == 'piano-black':
            k_th = 60
            
        #import pdb; pdb.set_trace()
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

        display_image_symbols, bound_rect, _, centroids_of_symbols, _ = coord(rso)


        #display_image_symbols = cv2.cvtColor(display_image_symbols, cv2.COLOR_BGR2RGB)
        
        #cv2.imwrite(absolout_path + "../../test_area/test_image_results/rso.png", display_image_symbols)
        
        #show_image([gray, inner_mask, not_inner_mask, inlay_symbols, closing, rso], open_order=1, window=True)


        #list_temp_save_image = [gray, pre_threshold, inner_mask, not_inner_mask, inlay_symbols, closing, rso, display_image_symbols]
        #filename = [str(counter)+"_1gray", str(counter)+"_2pre_threshold", str(counter)+"_3inner_mask", str(counter)+"_4not_inner_mask", str(counter)+"_5inlay_symbols", str(counter)+"_6closing", str(counter)+"_7rso", str(counter)+"_8display"]
        #save_image(list_temp_save_image, path="temp_files/extractor_inlay_process", filename=filename, format="png")

        
        return np.array(centroids_of_symbols), rso_inlay_zone, contours, bound_rect, rso, display_image_symbols
        #return rso, np.array(centroids_of_symbols), bound_rect, rso_inlay_zone

class APPLIABLE_ALGORITHMS(OrderedEnum):
    SHI_THOMASI = 0
    DOUGLAS_PEUCKER = 1
    BMLHT = 2 #FOOTNOT: This is an our developed algorithm# #Based Multiple Line Histogram Transaction | TODO: Change Algorithm name :) #

class FIND_COUNTERS:
    def __init__(self):
        self.src_frame = None
        self.object_id = None
        self.indicator = None
        self.startx, self.starty, self.endx, self.endy = 0, 0, 0, 0
        self.number_of_lines = 0
        self.scale = 0
        
        self.dict_crop_lut = {
            "1765050009-UP": [1000,3000,1000,4000],
            "1765050262-UP": [1000,3000,1000,4000],
            "1765050009-SIDE": [1000,2500,500,4500],
            "1765050262-SIDE": [1000,2500,500,4500],
            "1780742500-UP": [700,2000,1000,4700],
            "1780742500-SIDE": [700,2000,1000,4700]
        }
        
        self.data_signal_for_graphing = []
        self.object_contour_points = dict()

        self._obj_PROCESS_TOOLS = PROCESS_TOOLS()

    def calculate(self, src_frame, object_id, indicator=None, box_coordinates=(0,0,0,0), number_of_lines=10, algorithm=APPLIABLE_ALGORITHMS.BMLHT, is_autointerpolation_active=False, object_color='white', crop_parameters=[0,0,0,0]):
        self.src_frame = src_frame
        old_h, old_w = self.src_frame.shape[:2]
        self.object_id = object_id
        self.indicator = indicator
        self.startx, self.starty, self.endx, self.endy = box_coordinates
        self.number_of_lines = number_of_lines
        
        self.data_signal_for_graphing = []
        self.object_contour_points = dict()

        #print(self.startx, self.starty, self.endx, self.endy)

        self.object_color = object_color
        self.crop_parameters = crop_parameters

        if algorithm == APPLIABLE_ALGORITHMS.BMLHT:
            
            #self.src_frame = self._obj_PROCESS_TOOLS.pre_Process(self.src_frame, object_view=self.object_id, object_id_crop=self.dict_crop_lut[self.object_id], is_resized=True, is_gpu_active=False)
            
            print('Crop:', self.crop_parameters)
            print('1:', self.src_frame.shape)
            print('1:', box_coordinates)

            self.src_frame = self._obj_PROCESS_TOOLS.pre_Process_Machine(
                self.src_frame, 
                object_view=self.object_id, 
                object_color=self.object_color, 
                crop_parameters=self.crop_parameters, 
                is_resized=False)[0]
        
            box_coordinates = np.array(box_coordinates)
            box_coordinates = box_coordinates.reshape(-1,2)
            box_coordinates[:,0], box_coordinates[:,1] = self._obj_PROCESS_TOOLS.coordinate_Scaling(
                x=box_coordinates[:,0], y=box_coordinates[:,1],
                old_w=old_w, old_h=old_h,
                new_w=self.src_frame.shape[1], new_h=self.src_frame.shape[0],
                crop_x=self.crop_parameters[2], crop_y=self.crop_parameters[0],
                task='CROP',
                is_dual=False
            )
            box_coordinates = box_coordinates.reshape(-1)
            self.startx, self.starty, self.endx, self.endy = box_coordinates
            
            
            print('2:', self.src_frame.shape)
            print('2:', box_coordinates)
            
            """alpha = 0.5
            beta = (1.0 - alpha)
            rrggbb = cv2.cvtColor(self.src_frame, cv2.COLOR_GRAY2BGR)
            dst = cv2.addWeighted(rrggbb, alpha, mask, beta, 0.0)
            cv2.imwrite("C:/Users/mahmut.yasak/Desktop/dst.png", dst)"""
            
            if self.indicator == 'LR':
                run = self.starty
                finish = self.endy
                step = (self.endy-self.starty)//self.number_of_lines
            
            if self.indicator == 'RL':
                run = self.endy
                finish = self.starty
                step = -(self.endy-self.starty)//self.number_of_lines

            if self.indicator == 'UD':
                run = self.startx
                finish = self.endx
                step = (self.endx-self.startx)//self.number_of_lines

            if self.indicator == 'DU':
                run = self.endx
                finish = self.startx
                step = -(self.endx-self.startx)//self.number_of_lines

            counter = 0
            for i in range(run, finish, step):
                self.object_contour_points[counter] = dict()
                self.object_contour_points[counter]['x'] = list()
                self.object_contour_points[counter]['y'] = list()
                self.object_contour_points[counter]['lines'] = list()

                if self.indicator == 'LR':
                    signal = self._obj_PROCESS_TOOLS.extract_Signal(self.src_frame, self.startx, i, self.endx, i)
                    
                    #if (self.object_id != '1780742500-UP') or (self.object_id != '1780742500-SIDE'):
                        #self.data_signal_for_graphing.append(signal)
                    peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=50, distance=300)
                    
                    """elif (self.object_id == '1780742500-UP') or (self.object_id == '1780742500-SIDE'):
                        signal = signal * -1
                        #self.data_signal_for_graphing.append(signal)
                        peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=-200, distance=300)"""

                    self.object_contour_points[counter]['x'].append(peakx + self.startx)
                    self.object_contour_points[counter]['y'].append(i)
                    self.object_contour_points[counter]['lines'].append((self.startx, i, self.endx, i))
                    self.data_signal_for_graphing.append((signal, peakx, peakx + self.startx, i))

                elif self.indicator == 'RL':
                    signal = self._obj_PROCESS_TOOLS.extract_Signal(self.src_frame, self.endx, i, self.startx, i)
                    
                    #if (self.object_id != '1780742500-UP') or (self.object_id != '1780742500-SIDE'):
                        #self.data_signal_for_graphing.append(signal)
                    peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=50, distance=300)
                    
                    """elif (self.object_id == '1780742500-UP') or (self.object_id == '1780742500-SIDE'):
                        signal = signal * -1
                        #self.data_signal_for_graphing.append(signal)
                        peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=-200, distance=300)"""
                    
                    self.object_contour_points[counter]['x'].append(self.endx - peakx)
                    self.object_contour_points[counter]['y'].append(i)
                    self.object_contour_points[counter]['lines'].append((self.endx, i, self.startx, i))
                    self.data_signal_for_graphing.append((signal, peakx, self.endx - peakx, i))
                
                elif self.indicator == 'UD':
                    signal = self._obj_PROCESS_TOOLS.extract_Signal(self.src_frame, i, self.starty, i, self.endy)
                    
                    #if (self.object_id != '1780742500-UP') or (self.object_id != '1780742500-SIDE'):
                        #self.data_signal_for_graphing.append(signal)
                    peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=20, distance=300)
                    
                    """elif (self.object_id == '1780742500-UP') or (self.object_id == '1780742500-SIDE'):
                        signal = signal * -1
                        #self.data_signal_for_graphing.append(signal)
                        peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=-200, distance=300)"""
                    
                    self.object_contour_points[counter]['x'].append(i)
                    if peakx != 0:
                        self.object_contour_points[counter]['y'].append(peakx + self.starty)
                        temp_y = peakx + self.starty
                    else:
                        self.object_contour_points[counter]['y'].append(0)
                        temp_y = 0
                    self.object_contour_points[counter]['lines'].append((i, self.starty, i, self.endy))
                    self.data_signal_for_graphing.append((signal, peakx, i, temp_y))
                
                elif self.indicator == 'DU':
                    signal = self._obj_PROCESS_TOOLS.extract_Signal(self.src_frame, i, self.endy, i, self.starty)
                    
                    #if (self.object_id != '1780742500-UP') or (self.object_id != '1780742500-SIDE'):
                        #self.data_signal_for_graphing.append(signal)
                    peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=20, distance=300)
                    
                    """elif (self.object_id == '1780742500-UP') or (self.object_id == '1780742500-SIDE'):
                        signal = signal * -1
                        #self.data_signal_for_graphing.append(signal)
                        peakx = self._obj_PROCESS_TOOLS.find_Peak_Point_of_Signal(signal, height=-200, distance=300)"""
                    
                    self.object_contour_points[counter]['x'].append(i)
                    if peakx != 0:
                        self.object_contour_points[counter]['y'].append(self.endy - peakx)
                        temp_y = self.endy - peakx
                    else:
                        self.object_contour_points[counter]['y'].append(0)
                        temp_y = 0
                    self.object_contour_points[counter]['lines'].append((i, self.endy, i, self.starty))
                    self.data_signal_for_graphing.append((signal, peakx, i, temp_y))
                
                counter += 1
                
            if is_autointerpolation_active:
                self.data_signal_for_graphing = np.array(self.data_signal_for_graphing, dtype=object)

                # PARAMS: auto_Interpolation(list_line_points, method=1, loop_over=1, tolerance=3, is_fixed_optimization=False, point_filter=[0], buffer_len=5)
                # DO NOT FORGET TO CHANGE TOLERANCE and FIXED OPTIMIZATION params
                #res = auto_Interpolation(
                #    self.data_signal_for_graphing[:, 1], method=3, tolerance=4, is_fixed_optimization=True, buffer_len=5)[0]

                res = auto_Interpolation(self.data_signal_for_graphing[:, 1], method=4, tolerance=4)[0]
                    #auto_Interpolation(self.data_signal_for_graphing[:, 1], method=3, tolerance=4, is_fixed_optimization=True, buffer_len=5)[0]

                if self.indicator == 'LR':
                    """
                    print(self.data_signal_for_graphing[:, 1])
                    alfa = auto_Interpolation(self.data_signal_for_graphing[:, 1])
                    print(alfa[0], alfa[1])"""

                    self.data_signal_for_graphing[:, 2] = self.startx + res
                    self.data_signal_for_graphing[:, 1] = res
                    list_data_signal_for_graphing = self.data_signal_for_graphing[:, 2].tolist()
                    for dict_index in range(len(list_data_signal_for_graphing)):
                        self.object_contour_points[dict_index]['x'][0] = list_data_signal_for_graphing[dict_index]

                elif self.indicator == 'RL':
                    self.data_signal_for_graphing[:, 2] = self.endx - res
                    self.data_signal_for_graphing[:, 1] = res
                    list_data_signal_for_graphing = self.data_signal_for_graphing[:, 2].tolist()
                    for dict_index in range(len(list_data_signal_for_graphing)):
                        self.object_contour_points[dict_index]['x'][0] = list_data_signal_for_graphing[dict_index]

                elif self.indicator == 'UD':
                    self.data_signal_for_graphing[:, 3] = self.starty + res
                    self.data_signal_for_graphing[:, 1] = res
                    list_data_signal_for_graphing = self.data_signal_for_graphing[:, 3].tolist()
                    for dict_index in range(len(list_data_signal_for_graphing)):
                        self.object_contour_points[dict_index]['y'][0] = list_data_signal_for_graphing[dict_index]
                
                elif self.indicator == 'DU':
                    self.data_signal_for_graphing[:, 3] = self.endy - res
                    self.data_signal_for_graphing[:, 1] = res
                    list_data_signal_for_graphing = self.data_signal_for_graphing[:, 3].tolist()
                    for dict_index in range(len(list_data_signal_for_graphing)):
                        self.object_contour_points[dict_index]['y'][0] = list_data_signal_for_graphing[dict_index]

            return self.object_contour_points, self.data_signal_for_graphing, self.number_of_lines

        elif algorithm == APPLIABLE_ALGORITHMS.SHI_THOMASI:
            self.dst_frame = self._obj_PROCESS_TOOLS.inlay_Extract_Button(self.src_frame)
            contours, _ = cv2.findContours(self.dst_frame[6] , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            centroids_of_symbols = contour_Centroids(contours)

            display_image_symbols = self.src_frame.copy()
            display_image_inlay_section = cv2.cvtColor(self.dst_frame[-1], cv2.COLOR_GRAY2BGR)

            for i in centroids_of_symbols:
                cv2.circle(display_image_symbols, (i[0], i[1]) , 10, (0,255,0), -1)
            
            """corners = cv2.goodFeaturesToTrack(self.dst_frame[-1], 50, 0.01, 10)
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(display_image_inlay_section, (x, y) , 1, (0,0,255), -1)


            contours, _ = cv2.findContours(self.dst_frame[-1] , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
            approx_corners = sorted(np.concatenate(approx_corners).tolist())
            for index, c in enumerate(approx_corners):
                #dp.append((c[0], c[1]))
                #character = chr(65 + index)
                #print(character, ':', c)
                cv2.circle(display_image_inlay_section, (c[0], c[1]), 1, (255, 0, 0), -1)



            tl_x, tl_y = self._obj_PROCESS_TOOLS.corner_Detection(self.dst_frame[-1].copy(), 0,self.dst_frame[-1].shape[1],1, 0,self.dst_frame[-1].shape[0],1)
            bl_x, bl_y = self._obj_PROCESS_TOOLS.corner_Detection(self.dst_frame[-1].copy(), 0,self.dst_frame[-1].shape[1],1, self.dst_frame[-1].shape[0]-1,0,-1)
            br_x, br_y = self._obj_PROCESS_TOOLS.corner_Detection(self.dst_frame[-1].copy(), self.dst_frame[-1].shape[1]-1,0,-1, self.dst_frame[-1].shape[0]-1,0,-1)
            tr_x, tr_y = self._obj_PROCESS_TOOLS.corner_Detection(self.dst_frame[-1].copy(), self.dst_frame[-1].shape[1]-1,0,-1, 0,self.dst_frame[-1].shape[0],1)

            cv2.circle(display_image_inlay_section, (tl_x, tl_y), 1, (32, 255, 12), -1)
            cv2.circle(display_image_inlay_section, (bl_x, bl_y), 1, (32, 255, 12), -1)
            cv2.circle(display_image_inlay_section, (br_x, br_y), 1, (32, 255, 12), -1)
            cv2.circle(display_image_inlay_section, (tr_x, tr_y), 1, (32, 255, 12), -1)



            show_pack = [self.src_frame, display_image_symbols, display_image_inlay_section]"""
            
            show_pack = [self.src_frame, *self.dst_frame, display_image_symbols]
            return show_pack
            
