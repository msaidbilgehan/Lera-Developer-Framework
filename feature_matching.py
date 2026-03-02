#cython: language_level=3, boundscheck=False
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ordered_enum import OrderedEnum

from image_manipulation import contour_Centroids, draw_Rectangle, draw_Text, api_Draw_Complementarily, remove_Small_Object, contour_Extreme_Points, image_Make_Border
from image_tools import show_image, save_image
from stdo import stdo
from preprocess_image_processing import preprocess_of_Extractor_Difference_Laser_Printing, preprocess_of_Symbol_Centroid_Laser_Printing
from detect_stains import preprocess_of_Detect_Stain


class APPLICABLE_ALGORITHMS(OrderedEnum):
    ORB = 0
    SIFT = 1
    FLANN_based_SIFT = 2
    CWCFT = 3 #FOOTNOT: Counter Weighted Centroid Feature Transform#

class FEATURE_MATCHING:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()

        FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def finder(
        self,
        reference_image,
        sample_image,
        algorithm=APPLICABLE_ALGORITHMS.ORB,
        min_match_count=5,
        coord_thresh=5,
        ref_coords=[],
        sample_coords=[],
        ref_bb=[],
        sample_bb=[],
        is_inlay_activated=False,
        crop_ratio=10,
        termination_durty_points_elim=False,
        detect_stains_and_eliminate_matched=True,
        is_middle_object=False,
        middle_object_roi=[],
        stain_threshold=[100, 255, cv2.THRESH_BINARY_INV],
        stain_ratio=0.1,
        kernel=[4,5],
        is_line=False,
        line_frame_preprocessed=None,
        threshold_config=[100, 255, cv2.THRESH_BINARY_INV],
        is_mask_reference_image_by_sample_image=False,
        sector_index='',
        counter=0,
        flag_activate_debug_images=False
    ):

        if algorithm == APPLICABLE_ALGORITHMS.FLANN_based_SIFT:

            # default definition #
            homography_image = sample_image
            masked_reference_image = reference_image
            ######################

            kp1, des1 = self.sift.detectAndCompute(reference_image, None)
            kp2, des2 = self.sift.detectAndCompute(sample_image, None)

            good = []
            if des1 is not None or des2 is not None:
                flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                for m,n in matches:
                    # if m.distance < 0.7*n.distance:
                    good.append(m)

                if len(good) > min_match_count:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()

                    homography_image = cv2.warpPerspective(sample_image, M, (reference_image.shape[1], reference_image.shape[0]), borderValue=(255,255,255))

                    if is_mask_reference_image_by_sample_image:
                        corners = np.float32([
                            [0,0], [sample_image.shape[1]-1 ,0], [sample_image.shape[1]-1, sample_image.shape[0]-1], [0, sample_image.shape[0]-1]
                        ]).reshape(-1,1,2)
                        transformed_corners = cv2.perspectiveTransform(corners, M)
                        mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)
                        points = np.array([
                            [round(transformed_corners[0][0][0]), round(transformed_corners[0][0][1])],
                            [round(transformed_corners[1][0][0]), round(transformed_corners[1][0][1])],
                            [round(transformed_corners[2][0][0]), round(transformed_corners[2][0][1])],
                            [round(transformed_corners[3][0][0]), round(transformed_corners[3][0][1])],
                        ])

                        mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                        mask_inv = cv2.bitwise_not(mask)

                        first_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
                        mask_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                        masked_reference_image = cv2.add(first_image, mask_rgb)

                else:
                    stdo(2, "Not enough matches are found - %d/%d" %(len(good), min_match_count))
                    matchesMask = None
                    transformed_corners = np.float32([
                            [0,0], [sample_image.shape[1]-1 ,0], [sample_image.shape[1]-1, sample_image.shape[0]-1], [0, sample_image.shape[0]-1]
                    ]).reshape(-1,1,2)

            if flag_activate_debug_images:
                warped_image_border = cv2.polylines(homography_image.copy(), [np.int32(transformed_corners)], True, (0,255,0), 1, cv2.LINE_AA)

                draw_params = dict(
                    matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2
                )
                result_image = cv2.drawMatches(reference_image, kp1, sample_image, kp2, good, None, **draw_params)

                list_temp_save_image = [warped_image_border, result_image, homography_image, masked_reference_image]
                filename = [str(counter)+"_1_warped_image_border", str(counter)+"_2_result_image", str(counter)+"_3_homography_image", str(counter)+"_4_masked_reference_image"]
                save_image(list_temp_save_image, path="temp_files/feature_matching/"+str(algorithm), filename=filename, format="png")

            return masked_reference_image, homography_image

        elif algorithm == APPLICABLE_ALGORITHMS.SIFT:
            kp1, des1 = self.sift.detectAndCompute(reference_image,None)
            kp2, des2 = self.sift.detectAndCompute(sample_image,None)

            matches = self.bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)

            good = []
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            result_image = cv2.drawMatchesKnn(
                reference_image,
                kp1,
                sample_image,
                kp2,
                good,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

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
            return result_image, homography_image

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

            if is_inlay_activated:

                frame_ref_sample_difference = dict()
                for i in range(len(ref_coords)):
                    frame_ref_sample_difference[i] = dict()

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
                flag_is_succesfull = False

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
                    sample_frame_for_detection_stains = sample_image.copy()

                blind_points = list()
                drawed_points = list()

                # all_points = [*pre_c, *last_c]
                # stdo(1, "feature matching - all_points: {}".format(all_points))

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

                        if ( abs(pre_c[i][0]-last_c[j][0]) <= coord_thresh_1 ) and ( abs(pre_c[i][1]-last_c[j][1]) <= coord_thresh_2 ):

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

                            if termination_durty_points_elim:

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

                            ref_cropped_frame = pre_image[ norm_p_sy:norm_p_ey, norm_p_sx:norm_p_ex ]
                            sample_cropped_frame = last_image[ norm_s_sy:norm_s_ey, norm_s_sx:norm_s_ex ]

                            if detect_stains_and_eliminate_matched:

                                startx = 0 if sample[0]-40 < 0 else sample[0]-40
                                starty = 0 if sample[1]-40 < 0 else sample[1]-40
                                endx = sample_frame_for_detection_stains.shape[1] if sample[2]+80 > sample_frame_for_detection_stains.shape[1] else sample[2]+80
                                endy = sample_frame_for_detection_stains.shape[0] if sample[3]+80 > sample_frame_for_detection_stains.shape[0] else sample[3]+80

                                sample_frame_for_detection_stains =  draw_Rectangle(
                                    sample_frame_for_detection_stains,
                                    start_point=( startx, starty ),
                                    end_point=( endx, endy ),
                                    color=(255, 255, 255),
                                    thickness=-1
                                )

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

                            if is_switched:
                                temp = ref_cropped_frame.copy()
                                ref_cropped_frame = sample_cropped_frame.copy()
                                sample_cropped_frame = temp

                                sample_start_x = norm_p_sx
                                sample_start_y = norm_p_sy

                            else:
                                sample_start_x = norm_s_sx
                                sample_start_y = norm_s_sy


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
                        drawed_points.append(pre_c[i])


                if detect_stains_and_eliminate_matched:

                    frame_preprocess_detect_stain = preprocess_of_Detect_Stain(
                        sample_frame_for_detection_stains.copy(),
                        is_middle_object=is_middle_object,
                        middle_object_roi=middle_object_roi,
                        threshold_config=threshold_config,
                        is_line=is_line,
                        line_frame_preprocessed=line_frame_preprocessed,
                        is_stain_threshold=True,
                        stain_threshold_config=stain_threshold,
                        sector_index=sector_index,
                        activate_debug_images=flag_activate_debug_images
                    )

                    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
                    # dilate = cv2.dilate(frame_s, kernel, iterations=1)

                    stain_ratio_px = stain_ratio * 11.35 # (454px -> 40mm | 1px -> 0.088mm | 1mm -> 11.35px) mm to px

                    _, contours, all_contour_area, will_be_removed_buffer, not_removed_buffer = remove_Small_Object(
                        frame_preprocess_detect_stain.copy(),
                        is_chosen_max_area=False,
                        is_contour_number_for_area=False,
                        ratio=int(stain_ratio_px*stain_ratio_px),
                        counter=not_found_index
                    )

                    centroid_contour = contour_Centroids(not_removed_buffer)

                    stdo(1, "[{}]: Unmatched Area-1|0: - stain_area_rso(px2):{:.3f} | centroid_contour:{}".format(not_found_index, stain_ratio_px*stain_ratio_px, centroid_contour))

                    for id, coords in enumerate(centroid_contour):
                        frame_not_found.append(dict())

                        centroids_x = int(coords[0])
                        centroids_y = int(coords[1])

                        frame_not_found[not_found_index]['ref_frame'] = -1
                        frame_not_found[not_found_index]['sample_frame'] = -1
                        frame_not_found[not_found_index]['crop'] = centroids_x, centroids_y

                        not_found_index += 1


                """
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

                # import pdb; pdb.set_trace()

                for point in blind_points:
                    frame_not_found.append(dict())

                    frame_not_found[not_found_index]['ref_frame'] = -1
                    frame_not_found[not_found_index]['sample_frame'] = -1
                    frame_not_found[not_found_index]['crop'] = point[0], point[1]
                    not_found_index += 1

                    stdo(1, "[{}]: Unmatched Area-1|1: - stain_coords:({})".format(not_found_index, point))

                    # cv2.circle(display_r, (point[0], point[1]) , 7, (255,20,200), -1)
                """

            return frame_ref_sample_difference, frame_not_found
