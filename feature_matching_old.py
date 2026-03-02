#cython: language_level=3, boundscheck=False
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ordered_enum import OrderedEnum

from image_manipulation import contour_Centroids, draw_Rectangle, draw_Text, api_Draw_Complementarily, remove_Small_Object, contour_Extreme_Points
from image_tools import show_image, save_image
from stdo import stdo
from preprocess_image_processing import preprocess_of_Extractor_Difference_Laser_Printing, preprocess_of_Symbol_Centroid_Laser_Printing


class APPLICABLE_ALGORITHMS(OrderedEnum):
    ORB = 0
    SIFT = 1
    FLANN_based_SIFT = 2
    CWCFT = 3 # FOOTNOT: Counter Weighted Centroid Feature Transform #

class FEATURE_MATCHING:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()

        FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)

        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf = cv2.BFMatcher()

    def finder(
        self,
        reference_image,
        sample_image,
        algorithm=APPLICABLE_ALGORITHMS.ORB,
        min_match_count=5,
        configuration_id='N/A',
        pattern_id='0',
        counter=0,
        coord_thresh=5,
        ref_coords=[],
        sample_coords=[],
        ref_bb=[],
        sample_bb=[],
        is_inlay_activated=True,
        crop_ratio=10,
        termination_dirty_points_elim=False,
        detect_stains_and_eliminate_matched=True,
        stain_threshold=[100, 255, cv2.THRESH_BINARY_INV],
        stain_ratio=10,
        kernel=[4,5]
    ):

        if algorithm == APPLICABLE_ALGORITHMS.FLANN_based_SIFT:
            kp1, des1 = self.sift.detectAndCompute(reference_image, None)
            kp2, des2 = self.sift.detectAndCompute(sample_image, None)

            flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                #if m.distance < 0.7*n.distance:
                good.append(m)

            if len(good) > min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                homography_image = cv2.warpPerspective(sample_image, M, (w, h))

            else:
                print("Not enough matches are found - %d/%d" %(len(good), min_match_count))
                matchesMask = None
                return -1, -1

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

            result_image = cv2.drawMatches(reference_image,kp1,sample_image,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return result_image, homography_image

        elif algorithm == APPLICABLE_ALGORITHMS.SIFT:
            kp1, des1 = self.sift.detectAndCompute(reference_image.copy(), None)
            kp2, des2 = self.sift.detectAndCompute(sample_image.copy(), None)
            list_kp =[kp1, kp2]

            if (len(kp2) <= 0) or (des2.any() == None):
                return -1, -1, -1, [-1,-1]

            matches = self.bf.knnMatch(des1, des2, k=2)
            # matches = sorted(matches, key=lambda x:x.distance)
            matches = np.array(matches)

            good = []
            if len(matches.shape) == 2:
                if len(matches[:,0]) != len(matches[:,1]):
                    return -1, -1, -1, [-1,-1]
            else:
                return -1, -1, -1, [-1,-1]

            for m, n in matches:
                if m.distance < 0.9*n.distance:
                    good.append(m)


            result_image = cv2.drawMatches(
                reference_image,
                kp1,
                sample_image,
                kp2,
                good,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            image_pack = [reference_image, sample_image, result_image]
            title_pack = [str(counter) + "_0_reference_image_" + configuration_id, str(counter) + "_1_sample_image_" + configuration_id, str(counter) + "_2_result_image_" + configuration_id]
            save_image(image_pack, path="temp_files/FEATURE_MATCHING", filename=title_pack, format="png")

            result_image = None
            homography_image = None

            """
            if len(good) > min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                homography_image = cv2.warpPerspective(sample_image, M, (w, h))
            """

            return result_image, homography_image, good, list_kp

        elif algorithm == APPLICABLE_ALGORITHMS.ORB:
            kp1, des1 = self.orb.detectAndCompute(reference_image,None)
            kp2, des2 = self.orb.detectAndCompute(sample_image,None)

            matches = self.bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)

            result_image = cv2.drawMatches(reference_image,kp1,sample_image,kp2,matches[:10],
                                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good)>min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                homography_image = cv2.warpPerspective(sample_image, M, (w, h))
            return result_image, homography_image

        elif algorithm == APPLICABLE_ALGORITHMS.CWCFT:
            #print(len(ref_coords), len(sample_coords))
            #if (len(ref_coords) == min_match_count or len(sample_coords) == min_match_count):

            # FOR DEBUG THE DETECT STAIN AND DUST
            # list_drawing_function = list()
            # list_drawing_function_params = list()

            if is_inlay_activated:

                frame_ref_sample_difference = dict()
                for i in range(len(ref_coords)):
                    frame_ref_sample_difference[i] = dict()
                    #frame_ref_sample_difference[i]['ref_frame'] = list()
                    #frame_ref_sample_difference[i]['sample_frame'] = list()
                    #frame_ref_sample_difference[i]['crop'] = list()

                    crop_length_w = 70
                    crop_length_h = 60
                    if ((80 < sample_coords[i][0] < 380) and (380 < sample_coords[i][1] < 460)) or ((1300 < sample_coords[i][0] < 1600) and (380 < sample_coords[i][1] < 460)):
                        crop_length_w = 90
                        crop_length_h = 30
                    elif (500 < sample_coords[i][0] < 800) and (530 < sample_coords[i][1] < 630):
                        crop_length_w = 90
                        crop_length_h = 60

                    ref_start_x = round(ref_coords[i][0]) - crop_length_w
                    ref_start_y = round(ref_coords[i][1]) - crop_length_h
                    ref_end_x = round(ref_coords[i][0]) + crop_length_w
                    ref_end_y = round(ref_coords[i][1]) + crop_length_h
                    ref_cropped_frame = reference_image[ref_start_y:ref_end_y, ref_start_x:ref_end_x]

                    sample_start_x = round(sample_coords[i][0]) - crop_length_w
                    sample_start_y = round(sample_coords[i][1]) - crop_length_h
                    sample_end_x = round(sample_coords[i][0]) + crop_length_w
                    sample_end_y = round(sample_coords[i][1]) + crop_length_h
                    sample_cropped_frame = sample_image[sample_start_y:sample_end_y, sample_start_x:sample_end_x]

                    frame_ref_sample_difference[i]['ref_frame'] = ref_cropped_frame
                    frame_ref_sample_difference[i]['sample_frame'] = sample_cropped_frame
                    frame_ref_sample_difference[i]['crop'] = (sample_start_x, sample_start_y) #for coordinate rescalling#
                    return frame_ref_sample_difference, -1

            else:
                frame_ref_sample_difference = list()
                frame_not_found = list()
                sum_index = 0
                not_found_index = 0
                # is_switched = False

                if len(ref_bb) >= len(sample_bb):
                    pre_bb = ref_bb
                    last_bb = sample_bb
                    pre_image = reference_image
                    last_image = sample_image
                    pre_c = ref_coords
                    last_c = sample_coords
                    is_switched = False

                else:
                    pre_bb = sample_bb
                    last_bb = ref_bb
                    pre_image = sample_image
                    last_image = reference_image
                    pre_c = sample_coords
                    last_c = ref_coords
                    is_switched = True

                """
                display_r = pre_image
                display_s = last_image
                for i, ref in enumerate(pre_bb):
                    for j, sample in enumerate(last_bb):
                        cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 5, (0,255,0), -1)
                        cv2.rectangle(display_r, (int(ref[0]), int(ref[1]), int(ref[2]), int(ref[3])), (24,132,255), 2)

                        cv2.circle(display_s, (last_c[j][0], last_c[j][1]) , 5, (0,255,0), -1)
                        cv2.rectangle(display_s, (int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])), (24,132,255), 2)
                show_image([display_r, display_s], title=["FUCK","YOU"], open_order=2)
                """

                if detect_stains_and_eliminate_matched:
                    ref_cropped_frame_for_detection_stains = pre_image.copy()
                    sample_cropped_frame_for_detection_stains = last_image.copy()

                blind_points = list()
                drawed_points = list()

                all_points = [*pre_c, *last_c]

                for i, ref in enumerate(pre_bb):

                    flag_is_succesfull = False

                    for j, sample in enumerate(last_bb):


                        if (pre_c[i][0] == -1 or pre_c[i][1] == -1) or (last_c[j][0] == -1 or last_c[j][1] == -1):
                            #flag_is_succesfull = True
                            #print("{} - REF: {} | SAMPLE: {} /// ALFOQ: ({},{})".format(i, -1, -1, pre_c[i][0], last_c[j][0]))
                            #cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 10, (0,0,255), -1)

                            continue

                        if type(coord_thresh) is not int:
                            coord_thresh_1 = coord_thresh[0]
                            coord_thresh_2 = coord_thresh[1]
                        else:
                            coord_thresh_1 = coord_thresh
                            coord_thresh_2 = coord_thresh

                        if (abs(pre_c[i][0]-last_c[j][0]) <= coord_thresh_1) and (abs(pre_c[i][1]-last_c[j][1]) <= coord_thresh_2):


                            """
                            cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 5, (0,255,0), -1)
                            cv2.rectangle(display_r, (int(ref[0]), int(ref[1]), int(ref[2]), int(ref[3])), (24,132,255), 2)

                            cv2.circle(display_s, (last_c[j][0], last_c[j][1]) , 5, (0,255,0), -1)
                            cv2.rectangle(display_s, (int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])), (24,132,255), 2)
                            """


                            flag_is_succesfull = True

                            crop_length_w = crop_ratio
                            crop_length_h = crop_ratio

                            p_cx = pre_c[i][0]
                            p_cy = pre_c[i][1]
                            p_sx = ref[0]
                            p_sy = ref[1]
                            p_ex = ref[2] + ref[0]
                            p_ey = ref[3] + ref[1]


                            l_cx = last_c[j][0]
                            l_cy = last_c[j][1]
                            l_sx = sample[0]
                            l_sy = sample[1]
                            l_ex = sample[2] + sample[0]
                            l_ey = sample[3] + sample[1]

                            if termination_dirty_points_elim:

                                #cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 10, (0,0,255), -1)

                                if (500 < ref[2] or 200 < ref[3]):
                                    drawed_points.append(pre_c[i])
                                    drawed_points.append(last_c[j])
                                    #print("PRE-Long points:", "w:", ref[2], " | h:", ref[3])
                                    continue

                                elif (500 < sample[2] or 200 < sample[3]):
                                    drawed_points.append(last_c[j])
                                    #print("LAST-Long points:", "w:", sample[2], " | h:", sample[3])
                                    continue


                            if ref[2] <= sample[2]:
                                #REF#
                                ratio_sx = l_cx - l_sx
                                norm_p_sx = p_cx - ratio_sx - (crop_length_w)

                                ratio_ex = l_ex - l_cx
                                norm_p_ex = p_cx + ratio_ex + (crop_length_w)

                                #SAMPLE#
                                norm_s_sx = l_sx - (crop_length_w)
                                norm_s_ex = l_ex + (crop_length_w)

                            else:
                                #SAMPLE#
                                ratio_sx = p_cx - p_sx
                                norm_s_sx = l_cx - ratio_sx - (crop_length_w)

                                ratio_ex = p_ex - p_cx
                                norm_s_ex = l_cx + ratio_ex + (crop_length_w)

                                #REF#
                                norm_p_sx = p_sx - (crop_length_w)
                                norm_p_ex = p_ex + (crop_length_w)



                            if ref[3] <= sample[3]:
                                #REF#
                                ratio_sy = l_cy - l_sy
                                norm_p_sy = p_cy - ratio_sy - (crop_length_h)

                                ratio_ey = l_ey - l_cy
                                norm_p_ey = p_cy + ratio_ey + (crop_length_h)

                                #SAMPLE#
                                norm_s_sy = l_sy - (crop_length_h)
                                norm_s_ey = l_ey + (crop_length_h)

                            else:
                                #SAMPLE#
                                ratio_sy = p_cy - p_sy
                                norm_s_sy = l_cy - ratio_sy - (crop_length_h)

                                ratio_ey = p_ey - p_cy
                                norm_s_ey = l_cy + ratio_ey + (crop_length_h)

                                #REF#
                                norm_p_sy = p_sy - (crop_length_h)
                                norm_p_ey = p_ey + (crop_length_h)


                            # from image_tools import show_image; show_image(sample_image[sample_bb[0][1]:sample_bb[0][1] + sample_bb[0][3], sample_bb[0][0]: sample_bb[0][0] + sample_bb[0][2]])
                            # import pdb; pdb.set_trace()


                            ref_cropped_frame = pre_image[ norm_p_sy:norm_p_ey, norm_p_sx:norm_p_ex ]
                            sample_cropped_frame = last_image[ norm_s_sy:norm_s_ey, norm_s_sx:norm_s_ex ]

                            if detect_stains_and_eliminate_matched:
                                ref_cropped_frame_for_detection_stains =  draw_Rectangle(
                                    ref_cropped_frame_for_detection_stains,
                                    start_point=(norm_p_sx-2, norm_p_sy-2),
                                    end_point=(abs(norm_p_ex-norm_p_sx+2), abs(norm_p_ey-norm_p_sy+2)),
                                    color=(255, 255, 255),
                                    thickness=-1
                                )
                                sample_cropped_frame_for_detection_stains =  draw_Rectangle(
                                    sample_cropped_frame_for_detection_stains,
                                    start_point=(norm_s_sx-2, norm_s_sy-2),
                                    end_point=(abs(norm_s_ex-norm_s_sx+2), abs(norm_s_ey-norm_s_sy+2)),
                                    color=(255, 255, 255),
                                    thickness=-1
                                )

                                # list_image = [ref_cropped_frame_for_detection_stains, sample_cropped_frame_for_detection_stains]
                                # list_image_name = ["ref_cropped_frame_for_detection_stains", "sample_cropped_frame_for_detection_stains"]
                                # save_image(list_image, path="temp_files/feature_matching", filename=list_image_name, format="png")

                            #print("[FEATURE_MATCHING-FINDER]  REF-SHAPE:", ref_cropped_frame.shape, " | SAMPLE-SHAPE:", sample_cropped_frame.shape)


                            if (
                                ref_cropped_frame.shape != sample_cropped_frame.shape
                                ) or (
                                    ref_cropped_frame.shape[0] <= 1 or sample_cropped_frame.shape[0] <= 1
                                    ) or (
                                    ref_cropped_frame.shape[1] <= 1 or sample_cropped_frame.shape[1] <= 1
                                ):
                                #print("ERROR:", "REF:", ref_cropped_frame.shape, " | SAMPLE:", sample_cropped_frame.shape)
                                print("My shapes are not equal dear :*")
                                #cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 10, (0,0,255), -1)

                                continue

                                if (ref_cropped_frame.shape[0] > sample_cropped_frame.shape[0]) or (ref_cropped_frame.shape[1] > sample_cropped_frame.shape[1]):
                                    ref_cropped_frame = cv2.resize(ref_cropped_frame, (sample_cropped_frame.shape[1], sample_cropped_frame.shape[0]), cv2.INTER_LINEAR)
                                else:
                                    sample_cropped_frame = cv2.resize(sample_cropped_frame, (ref_cropped_frame.shape[1], ref_cropped_frame.shape[0]), cv2.INTER_LINEAR)



                            #print("{} - REF: {} | SAMPLE: {} /// sum_index: {}".format(i, ref_cropped_frame.shape, sample_cropped_frame.shape, sum_index))

                            if is_switched:
                                temp = ref_cropped_frame.copy()
                                ref_cropped_frame = sample_cropped_frame.copy()
                                sample_cropped_frame = temp

                                sample_start_x = norm_p_sx
                                sample_start_y = norm_p_sy

                            else:
                                sample_start_x = norm_s_sx
                                sample_start_y = norm_s_sy

                            """
                            frame_ref_sample_difference[sum_index]['ref_frame'] = ref_cropped_frame
                            frame_ref_sample_difference[sum_index]['sample_frame'] = sample_cropped_frame
                            frame_ref_sample_difference[sum_index]['crop'] = (sample_start_x, sample_start_y) #for coordinate rescalling#
                            """

                            #cv2.imwrite("C:/Users/mahmut.yasak/Desktop/black_m/" +str(i)+ "-ref.png", ref_cropped_frame)
                            #cv2.imwrite("C:/Users/mahmut.yasak/Desktop/black_m/" +str(i)+ "-sample.png", sample_cropped_frame)



                            ############### FOR-DATA-COLLECTOR ############
                            if not is_switched:
                                startx_dc = l_cx - (p_cx - p_sx) - (crop_length_w)
                                starty_dc = l_cy - (p_cy - p_sy) - (crop_length_h)
                                endx_dc = l_cx + (p_ex - p_cx) + (crop_length_w)
                                endy_dc = l_cy + (p_ey - p_cy ) + (crop_length_h)
                                cropped_image_data_collector = last_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                                """
                                temp_image = last_image.copy()
                                cv2.rectangle(temp_image, (startx_dc, starty_dc, endx_dc-startx_dc, endy_dc-starty_dc), (24,132,255), 2)
                                show_image(temp_image)
                                """

                            else:
                                startx_dc = p_cx - (l_cx - l_sx) - (crop_length_w)
                                starty_dc = p_cy - (l_cy - l_sy) - (crop_length_h)
                                endx_dc = p_cx + (l_ex - l_cx) + (crop_length_w)
                                endy_dc = p_cy + (l_ey - l_cy ) + (crop_length_h)
                                cropped_image_data_collector = pre_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                                """
                                temp_image = pre_image.copy()
                                cv2.rectangle(temp_image, (startx_dc, starty_dc, endx_dc-startx_dc, endy_dc-starty_dc), (24,132,255), 2)
                                show_image(temp_image)
                                """
                            ###############################################


                            frame_ref_sample_difference.append(dict())

                            frame_ref_sample_difference[sum_index]['ref_frame'] = ref_cropped_frame
                            frame_ref_sample_difference[sum_index]['sample_frame'] = sample_cropped_frame
                            frame_ref_sample_difference[sum_index]['crop'] = (sample_start_x, sample_start_y) #for coordinate rescalling#
                            frame_ref_sample_difference[sum_index]['data_collector'] = cropped_image_data_collector #collecting data for training deep learning model#

                            frame_ref_sample_difference[sum_index]['ref_coordinates'] = [norm_p_sy, norm_p_ey, norm_p_sx, norm_p_ex, p_cx, p_cy] #for paper-diagram
                            frame_ref_sample_difference[sum_index]['sample_coordinates'] = [norm_s_sy, norm_s_ey, norm_s_sx, norm_s_ex, l_cx, l_cy] #for paper-diagram


                            sum_index += 1
                            #last_bb.pop(j)
                            drawed_points.append(pre_c[i])
                            drawed_points.append(last_c[j])
                            break

                    #import pdb; pdb.set_trace()
                    if not flag_is_succesfull:
                        # frame_not_found.append(dict()) Command at 13.03.2023 for implementation the detect stain algorithm

                        #print("{} - REF: {} | SAMPLE: {} /// not_found_index: {} /// COORDS: ({})".format(i, display_r.shape, display_s.shape, not_found_index, pre_c[i]))

                        """ Command at 13.03.2023 for implementation the detect stain algorithm
                        frame_not_found[not_found_index]['ref_frame'] = -1
                        frame_not_found[not_found_index]['sample_frame'] = -1

                        startx = ref[0]
                        starty = ref[1]
                        endx = startx + ref[2]
                        endy = starty + ref[3]

                        centroids_x = int((startx + endx) / 2)
                        centroids_y = int((starty + endy) / 2)
                        """

                        """
                        area = (endx - startx) * (endy - starty)
                        stdo(1, "BB Area: {} || [Width Height]: [{} {}]".format(area, ref[2], ref[3]))

                        list_drawing_function.append(draw_Rectangle)
                        list_drawing_function_params.append([
                            (startx, starty),
                            (ref[2], ref[3]),
                            (0, 0, 255),
                            1
                        ])
                        list_drawing_function.append(draw_Text)
                        list_drawing_function_params.append([
                            [str(ref[2]) + "," +  str(ref[3])],
                            (startx-10, starty-10),
                            1,
                            (0, 255, 0),
                            2
                        ])
                        """

                        # temp_sample_image = draw_Rectangle(sample_image.copy(), start_point=((startx), (starty)), end_point=((ref[2]), (ref[3])), color=(0, 0, 255), thickness=1)
                        # temp_sample_image = cv2.rectangle(sample_image.copy(), (int(startx-10), int(starty-10), int(ref[2]+10), int(ref[3]+10)), (0, 255, 0), 1)
                        # save_image(temp_sample_image, path=None, filename=[str(centroids_x) + "_" + str(centroids_y) + "_stain"], format="png")

                        # temp_sample_image = draw_Text(temp_sample_image, text=[ str(ref[2]) + "," +  str(ref[3]) ], center_point=(startx-10, starty-10), fontscale=1, color=(0,255,0), thickness=2)
                        # save_image(temp_sample_image, path=None, filename=[str(centroids_x) + "_" + str(centroids_y) + "_stain"], format="png")

                        #cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 5, (255,0,0), -1)

                        # frame_not_found[not_found_index]['crop'] = centroids_x, centroids_y  #for coordinate rescalling #Command at 13.03.2023 for implementation the detect stain algorithm
                        # not_found_index += 1
                        drawed_points.append(pre_c[i])


                """
                temp_sample_image = sample_image.copy()
                api_Draw_Complementarily(
                    temp_sample_image,
                    list_drawing_function,
                    list_drawing_function_params
                )
                save_image(temp_sample_image, path=None, filename=["stains"], format="png")
                """

                if detect_stains_and_eliminate_matched:

                    frame_r, _, _ = preprocess_of_Extractor_Difference_Laser_Printing(
                        cv2.cvtColor(ref_cropped_frame_for_detection_stains.copy(), cv2.COLOR_RGB2GRAY),
                        is_sobel=False,
                        sobel_scale=0,
                        is_sharp=False,
                        sharp_scale=0,
                        threshold_config=stain_threshold,
                        is_middle_object=False,
                        window=False,
                        flag_activate_debug_images=False
                    )
                    frame_s, _, _ = preprocess_of_Extractor_Difference_Laser_Printing(
                        cv2.cvtColor(sample_cropped_frame_for_detection_stains.copy(), cv2.COLOR_RGB2GRAY),
                        is_sobel=False,
                        sobel_scale=0,
                        is_sharp=False,
                        sharp_scale=0,
                        threshold_config=stain_threshold,
                        is_middle_object=False,
                        window=False,
                        flag_activate_debug_images=False
                    )

                    stain_combine = cv2.add(frame_r, frame_s)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
                    dilate = cv2.dilate(stain_combine, kernel, iterations=1)
                    fill_padded_dilate = cv2.rectangle(dilate.copy(), (0, 0), (dilate.shape[1], dilate.shape[0]), (0, 0, 0), 40)

                    stain_ratio_px = stain_ratio * 11.35 # (454px -> 40mm | 1px -> 0.088mm | 1mm -> 11.35px) mm to px

                    """
                    for id, cnt in enumerate(contours):
                        bbox = contour_Extreme_Points(cnt, get_centroid=True, is_width_height=True)
                        stdo(1, "[{}]: Unmatched Area - bbox: {}".format(id, bbox))
                    """

                    _, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
                        fill_padded_dilate.copy(),
                        ratio=int(stain_ratio_px*stain_ratio_px),
                        counter=not_found_index
                    )
                    # stdo(1, "[{}]: Unmatched Area - px_count: {} | Unmatched Area - px2: {}".format(not_found_index, all_contour_area, int(stain_ratio_px*stain_ratio_px)))

                    centroid_contour = contour_Centroids(not_removed_buffer)
                    for id, coords in enumerate(centroid_contour):
                        frame_not_found.append(dict())

                        centroids_x = int(coords[0])
                        centroids_y = int(coords[1])

                        frame_not_found[id]['ref_frame'] = -1
                        frame_not_found[id]['sample_frame'] = -1
                        frame_not_found[id]['crop'] = centroids_x, centroids_y #for coordinate rescalling#

                        # stdo(1, "[{}]: Unmatched Area - not_removed_buffer: {}".format(id, all_contour_area[id]))


                    ### CENTROID ###
                    contours_poly = [None]*len(contours)
                    bb_cnt = [None]*len(contours)
                    for i, c in enumerate(contours):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        bb_cnt[i] = cv2.boundingRect(contours_poly[i])

                    # centroids_of_symbols = contour_Centroids(bb_cnt, get_bbox_centroid=True)
                    # display_image_symbols = cv2.cvtColor(fill_padded_dilate.copy(), cv2.COLOR_GRAY2BGR)
                    # for i, value in enumerate(centroids_of_symbols):
                    #     cv2.circle(display_image_symbols, (value[0], value[1]) , 1, (0,255,0), -1)
                    #    cv2.rectangle(display_image_symbols, (int(bb_cnt[i][0]), int(bb_cnt[i][1]), int(bb_cnt[i][2]), int(bb_cnt[i][3])), (24,132,255), 1)

                    # list_frame = [frame_r, frame_s, stain_combine, dilate, fill_padded_dilate, display_image_symbols]
                    # list_name = ["frame_r", "frame_s", "stain_combine",  "dilate", "fill_padded_dilate", "display_image_symbols"]
                    # save_image(list_frame, path="temp_files/feature_matching/detect_stain", filename=list_name, format="png")


                #import pdb; pdb.set_trace()
                #drawed_points = np.array(drawed_points)
                #all_points = np.array(all_points)

                #np.delete(all_points, np.where(all_points == drawed_points)[0])
                #mask = np.isin(all_points, drawed_points)
                #blind_points = all_points[mask == False].reshape(-1, 2)

                mask = list()
                flag_is_draw_eq = False
                for i in range(len(all_points)):
                    flag_is_draw_eq = False

                    for j in range(len(drawed_points)):
                        if (all_points[i][0] == drawed_points[j][0]) and (all_points[i][1] == drawed_points[j][1]):
                            flag_is_draw_eq = True
                            break

                    if not flag_is_draw_eq:

                        mask.append(i)
                all_points = np.array(all_points)
                blind_points = all_points[mask]

                #import pdb; pdb.set_trace()
                # res = [sub['crop'] for sub in frame_not_found]
                # stdo(1, "Unmatched: {}".format(res))

                not_found_index = id
                for point in blind_points:
                    frame_not_found.append(dict())

                    frame_not_found[not_found_index]['ref_frame'] = -1
                    frame_not_found[not_found_index]['sample_frame'] = -1
                    frame_not_found[not_found_index]['crop'] = point[0], point[1]
                    not_found_index += 1

                    #cv2.circle(display_r, (point[0], point[1]) , 7, (255,20,200), -1)


                # TODO: The
                #from image_manupilation import
                #import pdb; pdb.set_trace()
                # i = 0; from image_tools import show_image;
                #
                # i += 1; print("i:", i); show_image([frame_ref_sample_difference[i]["ref_frame"], frame_ref_sample_difference[i]["sample_frame"]], open_order=2);
                """
                    i = 0; from image_tools import show_image;

                    show_image([pre_image, last_image], open_order=2);

                    i += 1; print("i:", i); show_image([frame_ref_sample_difference[i]["ref_frame"], frame_ref_sample_difference[i]["sample_frame"], frame_ref_sample_difference[i+1]["ref_frame"], frame_ref_sample_difference[i+1]["sample_frame"], frame_ref_sample_difference[i+2]["ref_frame"], frame_ref_sample_difference[i+2]["sample_frame"]], open_order=2);
                """
                #frame_ref_sample_difference.append((ref_cropped_frame, sample_cropped_frame))
            #else:
            #    return -1

            # import pdb; pdb.set_trace()

            #show_image([display_r, display_s], open_order=2)

            frame_not_found = frame_not_found[:-1]

            # stdo(1, "[]: frame_not_found: {}".format(frame_not_found))


            return frame_ref_sample_difference, frame_not_found
