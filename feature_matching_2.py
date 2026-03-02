# cython: language_level=3, boundscheck=False
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
import time
from ordered_enum import OrderedEnum

from image_manipulation import contour_Centroids, draw_Circle, draw_Rectangle, draw_Text, remove_Small_Object
from image_tools import save_image
from stdo import stdo
# import extractor_difference
# import extractor_centroid
# from object_contour import PROCESS_TOOLS
from detect_stains import preprocess_of_Detect_Stain
from math_tools import is_Bbox_Inside_Other_Bbox

# xfeat_path = os.path.join(os.path.expanduser('~'), 'Workspace/lera-developer-framework/third_party_imports/accelerated_features')
# sys.path.append(xfeat_path)
# from modules.xfeat import XFeat


class APPLIABLE_ALGORITHMS(OrderedEnum):
    ORB = 0
    SIFT = 1
    FLANN_based_SIFT = 2
    CWCFT = 3  # FOOTNOT: Counter Weighted Centroid Feature Transform#
    FLANN_based_SIFT_2 = 4
    XFEAT = 5
    XFeat_resize = 6
    BRISK = 7
    AKAZE = 8


class FEATURE_MATCHING:
    def __init__(self):
        # self.orb = cv2.ORB_create(nfeatures=500, edgeThreshold=50)
        self.orb_m = cv2.ORB_create(nfeatures=10000, edgeThreshold=50)
        self.orb = cv2.ORB_create(nfeatures=100000, edgeThreshold=50)
        self.sift = cv2.SIFT_create(contrastThreshold=0.10)

        FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        self.bf_for_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf_for_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.bf_for_brisk = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf_for_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # self.xfeat = XFeat(top_k=4096)

        self.akaze = cv2.AKAZE_create()

        self.brisk = cv2.BRISK_create()

        # self._obj_process_tool = PROCESS_TOOLS()

    def finder(
        self,
        reference_image,
        sample_image,
        algorithm=APPLIABLE_ALGORITHMS.ORB,
        min_match_count=5,
        coord_thresh=5,
        ref_coords=[],
        sample_coords=[],
        ref_bb=[],
        sample_bb=[],
        is_inlay_activated=False,
        crop_ratio=10,
        detect_stains_and_eliminate_matched=True,
        is_middle_object=False,
        middle_object_roi=[],
        stain_threshold=[100, 255, cv2.THRESH_BINARY_INV],
        stain_ratio=0.1,
        stain_frame_preprocessed=None,
        kernel=[4,5],
        is_line=False,
        line_frame_preprocessed=None,
        threshold_config=[100, 255, cv2.THRESH_BINARY_INV],
        sector_index='',
        background_color='white',
        ref_bbox_for_detect_stain=[],
        sample_bbox_for_detect_stain=[],
        flag_activate_debug_images=False,
        counter=0
    ):

        if algorithm == APPLIABLE_ALGORITHMS.FLANN_based_SIFT:
            kp1, des1 = self.sift.detectAndCompute(reference_image, None)
            kp2, des2 = self.sift.detectAndCompute(sample_image, None)

            flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                #if m.distance < 0.7*n.distance:
                good.append(m)

            if len(good)>min_match_count:
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
                return -1

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

            result_image = cv2.drawMatches(reference_image,kp1,sample_image,kp2,good,None,**draw_params)
            return result_image, homography_image

        elif algorithm == APPLIABLE_ALGORITHMS.FLANN_based_SIFT_2:

            start_seq = time.time()

            start_detect = time.time()
            kp1, des1 = self.sift.detectAndCompute(reference_image, None)
            kp2, des2 = self.sift.detectAndCompute(sample_image, None)
            points = None
            stop_detect = time.time() - start_detect

            if des1 is None or des2 is None:
                return sample_image, reference_image, points

            start_knn_match = time.time()
            try:
                if len(des1) == len(des2):
                    matches = self.flann.knnMatch(des1, des2, k=2)
                    print("Equal")

                else:
                    print("Not Equal")
                    matches = self.flann.knnMatch(des1, des2, k=2)
                    # return sample_image, reference_image

            except:
                print("Except")
                return sample_image, reference_image, points
            stop_knn_match = time.time() - start_knn_match

            # print("::::matches", matches, type(matches))
            for match in matches:
                if len(match) < 2:
                    return sample_image, reference_image, points

            good = []
            for m,n in matches:
                #if m.distance < 0.7*n.distance:
                good.append(m)

            start_homography = time.time()

            if len(good)>min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                try:
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                except:
                    return sample_image, reference_image, points
                """
                if sample_image.dtype != np.uint8:
                    sample_image = sample_image.astype(np.uint8)
                """

                if M is None:
                    print("M is None")
                    return sample_image, reference_image, points


                # print("::::sample_image", sample_image.shape, type(sample_image), sample_image.dtype)
                # print("::::M", M, M.shape, type(M), M.dtype)

                h,w = reference_image.shape[:2]
                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

                h,w = sample_image.shape[:2]
                corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
                transformed_corners = cv2.perspectiveTransform(corners, M)
                # drawed_image = cv2.polylines(homography_sample_image.copy(), [np.int32(transformed_corners)], True, (0,255,0), 1, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
                elif background_color == 'black' or background_color == 'piano-black':
                    mask = np.ones((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8) * 255

                points = np.array([
                    [round(transformed_corners[0][0][0]), round(transformed_corners[0][0][1])],
                    [round(transformed_corners[1][0][0]), round(transformed_corners[1][0][1])],
                    [round(transformed_corners[2][0][0]), round(transformed_corners[2][0][1])],
                    [round(transformed_corners[3][0][0]), round(transformed_corners[3][0][1])],
                ], dtype=np.int32)

                mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                mask_inv = cv2.bitwise_not(mask)

                first_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
                # add_image = cv2.add(mask_image, mask_inv)
                mask_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                masked_reference_image = cv2.add(first_image, mask_rgb)


                if flag_activate_debug_images:
                    """
                    drawed_image = homography_sample_image.copy()
                    for i in range(4):
                        p1 = tuple(list(transformed_corners[i][0]))
                        p2 = tuple(list(transformed_corners[(i + 1) % 4][0]))
                        cv2.line(drawed_image, p1, p2, (0, 255, 0), 3)
                    """
                    draw_params = dict(
                        matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2
                    )
                    result_image = cv2.drawMatches(reference_image,kp1,sample_image,kp2,good,None,**draw_params)

                    list_temp_save_image = [result_image, homography_sample_image, masked_reference_image, reference_image, sample_image]
                    filename = [
                        str(sector_index) + "_0_result_image",
                        str(sector_index) + "_1_homography_sample_image",
                        str(sector_index) + "_2_masked_reference_image",
                        str(sector_index) + "_3_reference_image",
                        str(sector_index) + "_4_sample_image"
                    ]
                    save_image(list_temp_save_image, path="temp_files/feature_matching_2/FLANN_based_SIFT_2", filename=filename, format="jpg")

            else:
                return sample_image, reference_image, points

            stop_homography = time.time() - start_homography

            stop_seq = time.time() - start_seq

            stdo(1, "[{}] T:{:.2f} - Detect:{:.2f} | knn-Match:{:.2f} | Homo:{:.2f}".format
                (
                    "FLANN_based_SIFT",
                    stop_seq,
                    stop_detect,
                    stop_knn_match,
                    stop_homography,
                )
            )

            return homography_sample_image, masked_reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.SIFT:
            kp1, des1 = self.sift.detectAndCompute(reference_image,None)
            kp2, des2 = self.sift.detectAndCompute(sample_image,None)
            points = None

            if des1 is None or des2 is None:
                return sample_image, reference_image, points

            des1 = np.asarray(des1, dtype=np.float32)
            des2 = np.asarray(des2, dtype=np.float32)

            matches = self.bf_for_sift.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            good = []
            # for m, n in matches:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)
            for m in matches:
                if m.distance < 50:
                    good.append([m])

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
                # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

            return homography_sample_image, reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.ORB:

            start_seq = time.time()

            start_detect = time.time()
            if sector_index == 'M':
                kp1, des1 = self.orb_m.detectAndCompute(reference_image,None)
                kp2, des2 = self.orb_m.detectAndCompute(sample_image,None)
            else:
                kp1, des1 = self.orb.detectAndCompute(reference_image,None)
                kp2, des2 = self.orb.detectAndCompute(sample_image,None)
            points = None
            stop_detect = time.time() - start_detect

            if des1 is None or des2 is None:
                return sample_image, reference_image, points

            start_knn_match = time.time()
            matches = self.bf_for_orb.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)

            if flag_activate_debug_images:
                result_image = cv2.drawMatches(reference_image,kp1,sample_image,kp2,matches[:],None,flags=2)

            stop_knn_match = time.time() - start_knn_match

            start_homography = time.time()
            if len(matches)>min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if M is None:
                    print("M is None")
                    return sample_image, reference_image, points

                # matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

                h,w = sample_image.shape[:2]
                corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
                transformed_corners = cv2.perspectiveTransform(corners, M)
                # drawed_image = cv2.polylines(homography_sample_image.copy(), [np.int32(transformed_corners)], True, (0,255,0), 1, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
                elif background_color == 'black' or background_color == 'piano-black':
                    mask = np.ones((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8) * 255

                points = np.array([
                    [round(transformed_corners[0][0][0]), round(transformed_corners[0][0][1])],
                    [round(transformed_corners[1][0][0]), round(transformed_corners[1][0][1])],
                    [round(transformed_corners[2][0][0]), round(transformed_corners[2][0][1])],
                    [round(transformed_corners[3][0][0]), round(transformed_corners[3][0][1])],
                ], dtype=np.int32)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                    mask_inv = cv2.bitwise_not(mask)
                elif background_color == 'black' or background_color == 'piano-black':
                    mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                    mask_inv = cv2.bitwise_not(mask)

                first_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
                # add_image = cv2.add(mask_image, mask_inv)
                mask_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                masked_reference_image = cv2.add(first_image, mask_rgb)

                if flag_activate_debug_images:
                    list_temp_save_image = [result_image, homography_sample_image, masked_reference_image, reference_image, sample_image]
                    filename = [
                        str(sector_index) + "_0_result_image",
                        str(sector_index) + "_1_homography_sample_image",
                        str(sector_index) + "_2_masked_reference_image",
                        str(sector_index) + "_3_reference_image",
                        str(sector_index) + "_4_sample_image"
                    ]
                    save_image(list_temp_save_image, path="temp_files/feature_matching_2/ORB", filename=filename, format="jpg")

            else:
                return sample_image, reference_image, points

            stop_homography = time.time() - start_homography
            stop_seq = time.time() - start_seq

            # stdo(1, "[{}] T:{:.2f} - Detect:{:.2f} | knn-Match:{:.2f} | Homo:{:.2f}".format
            #     (
            #         "ORB",
            #         stop_seq,
            #         stop_detect,
            #         stop_knn_match,
            #         stop_homography,
            #     )
            # )

            return homography_sample_image, masked_reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.BRISK:
            kp1, des1 = self.brisk.detectAndCompute(reference_image,None)
            kp2, des2 = self.brisk.detectAndCompute(sample_image,None)
            points = None

            if des1 is None or des2 is None:
                return sample_image, reference_image, points

            matches = self.bf_for_brisk.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            good = []
            # for m, n in matches:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)
            for m in matches:
                if m.distance < 50:
                    good.append([m])

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
                # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

            return homography_sample_image, reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.AKAZE:
            kp1, des1 = self.akaze.detectAndCompute(reference_image,None)
            kp2, des2 = self.akaze.detectAndCompute(sample_image,None)
            points = None

            if des1 is None or des2 is None:
                return sample_image, reference_image, points

            matches = self.bf_for_akaze.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            good = []
            # for m, n in matches:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)
            for m in matches:
                if m.distance < 50:
                    good.append([m])

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
                # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                #dst = cv2.perspectiveTransform(pts,M)
                #drawed_image = cv2.polylines(sample_image.copy(), [np.int32(dst)], True, (255,0,0), 30, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

            return homography_sample_image, reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.XFEAT:

            start_seq = time.time()

            start_detect = time.time()
            kp1 = self.xfeat.detectAndCompute(reference_image, top_k=4096)[0]
            kp2 = self.xfeat.detectAndCompute(sample_image, top_k=4096)[0]
            points = None
            if kp1 is None or kp2 is None:
                return sample_image, reference_image, points
            stop_detect = time.time() - start_detect


            start_knn_match = time.time()
            kp1.update({'image_size': (reference_image.shape[1], reference_image.shape[0])})
            kp2.update({'image_size': (sample_image.shape[1], sample_image.shape[0])})
            try:
                mkpts = self.xfeat.match_lighterglue(kp1, kp2)
                _, mask = cv2.findHomography(mkpts[0], mkpts[1], cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
                mask = mask.flatten()
                keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts[0]]
                keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts[1]]
                matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

            except Exception as e:
                print("Except", e)
                return sample_image, reference_image, points
            stop_knn_match = time.time() - start_knn_match


            start_homography = time.time()
            if len(matches) > min_match_count:
                src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h,w = reference_image.shape[:2]
                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M, (w, h), borderValue=(0,0,0))

                h,w = sample_image.shape[:2]
                corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
                transformed_corners = cv2.perspectiveTransform(corners, M)
                # drawed_image = cv2.polylines(homography_sample_image.copy(), [np.int32(transformed_corners)], True, (0,255,0), 1, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
                elif background_color == 'black' or background_color == 'piano-black':
                    mask = np.ones((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8) * 255

                points = np.array([
                    [round(transformed_corners[0][0][0]), round(transformed_corners[0][0][1])],
                    [round(transformed_corners[1][0][0]), round(transformed_corners[1][0][1])],
                    [round(transformed_corners[2][0][0]), round(transformed_corners[2][0][1])],
                    [round(transformed_corners[3][0][0]), round(transformed_corners[3][0][1])],
                ], dtype=np.int32)

                mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                mask_inv = cv2.bitwise_not(mask)

                first_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
                # add_image = cv2.add(mask_image, mask_inv)
                mask_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                masked_reference_image = cv2.add(first_image, mask_rgb)

                if flag_activate_debug_images:

                    result_image = cv2.drawMatches(reference_image,keypoints1,sample_image,keypoints2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                    list_temp_save_image = [result_image, homography_sample_image, masked_reference_image, reference_image, sample_image]
                    filename = [
                        sector_index + "_" + str(counter) + "_0_result_image",
                        sector_index + "_" + str(counter) + "_1_homography_sample_image",
                        sector_index + "_" + str(counter) + "_2_masked_reference_image",
                        sector_index + "_" + str(counter) + "_3_reference_image",
                        sector_index + "_" + str(counter) + "_4_sample_image",
                    ]
                    save_image(list_temp_save_image, path="temp_files/feature_matching_2/XFeat", filename=filename, format="jpg")

            else:
                return sample_image, reference_image, points

            stop_homography = time.time() - start_homography

            stop_seq = time.time() - start_seq
            # stdo(1, "[{}] T:{:.2f} - Detect:{:.2f} | knn-Match:{:.2f} | Homo:{:.2f}".format
            #     (
            #         "XFeat",
            #         stop_seq,
            #         stop_detect,
            #         stop_knn_match,
            #         stop_homography,
            #     )
            # )

            return homography_sample_image, masked_reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.XFeat_resize:
            start_seq = time.time()

            resized_w, resized_h = 1185, 675  # Küçük boyut, hız için ayarlanabilir
            ref_h, ref_w = reference_image.shape[:2]
            smp_h, smp_w = sample_image.shape[:2]

            # Ölçek oranları
            scale_ref = (ref_w / resized_w, ref_h / resized_h)
            scale_smp = (smp_w / resized_w, smp_h / resized_h)

            # Resize işlemleri
            reference_resized = cv2.resize(reference_image, (resized_w, resized_h))
            sample_resized = cv2.resize(sample_image, (resized_w, resized_h))



            start_detect = time.time()
            kp1 = self.xfeat.detectAndCompute(reference_resized, top_k=4096)[0]
            kp2 = self.xfeat.detectAndCompute(sample_resized, top_k=4096)[0]
            points = None
            if kp1 is None or kp2 is None:
                return sample_image, reference_image, points
            stop_detect = time.time() - start_detect


            start_knn_match = time.time()
            kp1.update({'image_size': (reference_resized.shape[1], reference_resized.shape[0])})
            kp2.update({'image_size': (sample_resized.shape[1], sample_resized.shape[0])})

            try:
                mkpts = self.xfeat.match_lighterglue(kp1, kp2)

                _, mask = cv2.findHomography(mkpts[0], mkpts[1], cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
                mask = mask.flatten()
                keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts[0]]
                keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts[1]]
                matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

            except:
                print("Except")
                return sample_image, reference_image, points
            stop_knn_match = time.time() - start_knn_match


            start_homography = time.time()
            if len(matches) > min_match_count:
                src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

                start_find_homography = time.time()
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                stop_find_homography = time.time() - start_find_homography

                # Ölçekleme matrisleri
                scale_up_ref = np.diag([scale_ref[0], scale_ref[1], 1.0])
                scale_up_smp = np.diag([1.0 / scale_smp[0], 1.0 / scale_smp[1], 1.0])

                # Resize edilmiş M matrisini orijinal boyuta adapte et
                M_scaled = scale_up_ref @ M @ scale_up_smp

                start_warp_homography = time.time()
                h,w = reference_image.shape[:2]
                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    homography_sample_image = cv2.warpPerspective(sample_image, M_scaled, (w, h), borderValue=(255,255,255))
                elif background_color == 'black' or background_color == 'piano-black':
                    homography_sample_image = cv2.warpPerspective(sample_image, M_scaled, (w, h), borderValue=(0,0,0))
                stop_warp_homography = time.time() - start_warp_homography

                h,w = sample_image.shape[:2]
                corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(-1,1,2)
                transformed_corners = cv2.perspectiveTransform(corners, M_scaled)
                # drawed_image = cv2.polylines(homography_sample_image.copy(), [np.int32(transformed_corners)], True, (0,255,0), 1, cv2.LINE_AA)

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
                elif background_color == 'black' or background_color == 'piano-black':
                    mask = np.ones((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8) * 255

                points = np.array([
                    [round(transformed_corners[0][0][0]), round(transformed_corners[0][0][1])],
                    [round(transformed_corners[1][0][0]), round(transformed_corners[1][0][1])],
                    [round(transformed_corners[2][0][0]), round(transformed_corners[2][0][1])],
                    [round(transformed_corners[3][0][0]), round(transformed_corners[3][0][1])],
                ], dtype=np.int32)

                mask = cv2.fillPoly(mask, [points], color=(255), lineType=cv2.LINE_AA)
                mask_inv = cv2.bitwise_not(mask)

                first_image = cv2.bitwise_and(reference_image, reference_image, mask=mask)
                # add_image = cv2.add(mask_image, mask_inv)
                mask_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                masked_reference_image = cv2.add(first_image, mask_rgb)

                if flag_activate_debug_images:

                    result_image = cv2.drawMatches(reference_resized,keypoints1,sample_resized,keypoints2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                    list_temp_save_image = [result_image, homography_sample_image, masked_reference_image, reference_image, sample_image]
                    filename = [
                        sector_index + "_" + str(counter) + "_0_result_image",
                        sector_index + "_" + str(counter) + "_1_homography_sample_image",
                        sector_index + "_" + str(counter) + "_2_masked_reference_image",
                        sector_index + "_" + str(counter) + "_3_reference_image",
                        sector_index + "_" + str(counter) + "_4_sample_image",
                    ]
                    save_image(list_temp_save_image, path="temp_files/feature_matching_2/XFeat_resize", filename=filename, format="jpg")

            else:
                return sample_image, reference_image, points

            stop_homography = time.time() - start_homography

            stop_seq = time.time() - start_seq
            # stdo(1, "[{}] T:{:.2f} - Detect:{:.2f} | knn-Match:{:.2f} | Homo:{:.2f} - f-homo:{:.2f} - w-homo:{:.2f}".format
            #     (
            #         "XFeat",
            #         stop_seq,
            #         stop_detect,
            #         stop_knn_match,
            #         stop_homography,
            #         stop_find_homography,
            #         stop_warp_homography
            #     )
            # )

            return homography_sample_image, masked_reference_image, points

        elif algorithm == APPLIABLE_ALGORITHMS.CWCFT:

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

            if flag_activate_debug_images:
                display_r = pre_image.copy()
                display_s = last_image.copy()

                # temp_r = pre_image.copy()
                # for i, ref in enumerate(pre_bb):
                #     draw_Rectangle(temp_r, start_point=(int(ref[0]), int(ref[1])), end_point=(int(ref[2]), int(ref[3])), color=(24,132,255), thickness=1)
                #     draw_Text(temp_r, text=[str(i)], center_point=(int(ref[0]), int(ref[1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)
                # temp_s = sample_image.copy()
                # for i, sample in enumerate(last_bb):
                #     draw_Rectangle(temp_s, start_point=(int(sample[0]), int(sample[1])), end_point=(int(sample[2]), int(sample[3])), color=(24,132,255), thickness=1)
                #     draw_Text(temp_s, text=[str(i)], center_point=(int(sample[0]), int(sample[1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)
                # list_frame = [temp_r, temp_s]
                # title_pack = ["temp_r", "temp_s",]
                # save_image(list_frame, path="temp_files/feature_matching_2/CWCFT/", filename=title_pack, format="jpg")

            if detect_stains_and_eliminate_matched:
                sample_frame_for_detection_stains = sample_image.copy()

                if background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol':
                    color = (255,255,255)
                elif background_color == 'black' or background_color == 'piano-black':
                    color = (0,0,0)

                if sector_index == 'R' and (background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol'): # 15.07.2025
                    sample_bbox_for_detect_stain[0][0] = sample_bbox_for_detect_stain[0][0] - 20 # 15.07.2025
                    sample_bbox_for_detect_stain[0][1] = sample_bbox_for_detect_stain[0][1] - 20 # 15.07.2025
                    sample_bbox_for_detect_stain[1][0] = sample_bbox_for_detect_stain[1][0] + 40 # 15.07.2025
                    sample_bbox_for_detect_stain[1][1] = sample_bbox_for_detect_stain[1][1] + 40 # 15.07.2025
                    x, y, w, h = sample_bbox_for_detect_stain[0][0], sample_bbox_for_detect_stain[0][1], sample_bbox_for_detect_stain[1][0] - sample_bbox_for_detect_stain[0][0], sample_bbox_for_detect_stain[2][1] - sample_bbox_for_detect_stain[0][1] # 15.07.2025
                    x, y = x-20, y-20
                    w, h = w+80, h+80
                    box = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ], dtype=np.int64)
                    sample_frame_for_detection_stains = cv2.drawContours(
                        sample_frame_for_detection_stains,
                        [box],
                        -1,
                        color,
                        thickness=-1
                    ) # 15.07.2025

                if isinstance(sample_bb, list) and len(sample_bb) > 0:
                    sample_bb = np.array(sample_bb, dtype=np.int64)
                    for bbox in sample_bb:
                        x, y, w, h = bbox  # Eğer bbox (x, y, w, h) formatındaysa
                        x, y = x-10, y-10
                        w, h = w+20, h+20
                        box = np.array([
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ], dtype=np.int64)

                        if sector_index == 'R' and (background_color == 'white' or background_color == 'gray - siyah sembol' or background_color == 'gray - beyaz sembol'): # 15.07.2025
                            symbol_bbox = [x, y, w, h] # 15.07.2025
                            touch_screen_bbox = [sample_bbox_for_detect_stain[0][0], sample_bbox_for_detect_stain[0][1], sample_bbox_for_detect_stain[1][0] - sample_bbox_for_detect_stain[0][0], sample_bbox_for_detect_stain[2][1] - sample_bbox_for_detect_stain[0][1]] # 15.07.2025
                            if is_Bbox_Inside_Other_Bbox(touch_screen_bbox, symbol_bbox): # 15.07.2025
                                continue # 15.07.2025

                        sample_frame_for_detection_stains = cv2.drawContours(
                            sample_frame_for_detection_stains,
                            [box],
                            -1,
                            color,
                            thickness=-1
                        )

            blind_points = list()
            drawed_points_pre = list()
            drawed_points_last = list()

            crop_length_w, crop_length_h = crop_ratio, crop_ratio

            all_points = [*pre_c, *last_c]
            all_points_pre = [*pre_c]
            all_points_last = [*last_c]
            all_bbox = [*pre_bb, *last_bb]
            all_bbox_pre = [*pre_bb]
            all_bbox_last = [*last_bb]
            # stdo(1, "feature matching - all_points: {}".format(all_points))

            for i, ref in enumerate(pre_bb):

                flag_is_succesfull = False

                for j, sample in enumerate(last_bb):

                    if (pre_c[i][0] == -1 or pre_c[i][1] == -1) or (last_c[j][0] == -1 or last_c[j][1] == -1):
                        # flag_is_succesfull = True
                        # cv2.circle(display_r, (pre_c[i][0], pre_c[i][1]) , 10, (0,0,255), -1)
                        stdo(1, "[{}] Unmatched Area-Negative: pre_c:{} | last_c:{}".format(pre_c[i], last_c[j]))
                        continue

                    if isinstance(coord_thresh, float):
                        coord_thresh_1 = coord_thresh
                        coord_thresh_2 = coord_thresh
                    else:
                        coord_thresh_1 = coord_thresh[0]
                        coord_thresh_2 = coord_thresh[1]

                    coord_thresh_1 = int(coord_thresh_1 * 20.244) # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
                    coord_thresh_2 = int(coord_thresh_2 * 20.244) # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
                    if  (
                            ( abs(pre_c[i][0]-last_c[j][0]) <= coord_thresh_1 ) and ( abs(pre_c[i][1]-last_c[j][1]) <= coord_thresh_2 )
                        ) and (
                            ( abs(ref[2]-sample[2]) <= 15 ) and ( abs(ref[3]-sample[3]) <= 15 )
                        ):

                        if flag_activate_debug_images:
                            # stdo(1, "[{}] Matched Area: pre_c:{} | last_c:{} | ref:{} | sample:{}".format(sum_index, pre_c[i], last_c[j], ref, sample))

                            draw_Circle(display_r, (pre_c[i][0], pre_c[i][1]), radius=1, color=(0, 255, 0), thickness=-1)
                            draw_Rectangle(display_r, start_point=(int(ref[0]), int(ref[1])), end_point=(int(ref[2]), int(ref[3])), color=(24,132,255), thickness=1)
                            draw_Text(display_r, text=[str(i)], center_point=(int(ref[0]), int(ref[1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)

                            draw_Circle(display_s, (last_c[j][0], last_c[j][1]), radius=1, color=(0, 255, 0), thickness=-1)
                            draw_Rectangle(display_s, start_point=(int(sample[0]), int(sample[1])), end_point=(int(sample[2]), int(sample[3])), color=(24,132,255), thickness=1)
                            draw_Text(display_s, text=[str(j)], center_point=(int(sample[0]), int(sample[1])+15), fontscale=0.3, color=(0,0,255), thickness=1, plain=False)


                        flag_is_succesfull = True

                        # crop_length_w = crop_ratio
                        # crop_length_h = crop_ratio
                        if ref[2] / ref[3] > 3:
                            crop_length_w = 5
                            crop_length_h = 5

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

                        if (
                                ref_cropped_frame.shape != sample_cropped_frame.shape
                            ) or (
                                ref_cropped_frame.shape[0] <= 1 or sample_cropped_frame.shape[0] <= 1
                            ) or (
                                ref_cropped_frame.shape[1] <= 1 or sample_cropped_frame.shape[1] <= 1
                            ):
                            stdo(2, "[{}] Shapes are not equal: Ref:{} | Sample:{}".format(sum_index, ref_cropped_frame.shape, sample_cropped_frame.shape))
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
                        # if not is_switched:
                        #     startx_dc = l_cx - (p_cx - p_sx) - (crop_length_w)
                        #     starty_dc = l_cy - (p_cy - p_sy) - (crop_length_h)
                        #     endx_dc = l_cx + (p_ex - p_cx) + (crop_length_w)
                        #     endy_dc = l_cy + (p_ey - p_cy ) + (crop_length_h)
                        #     cropped_image_data_collector = last_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                        #     """
                        #     temp_image = last_image.copy()
                        #     cv2.rectangle(temp_image, (startx_dc, starty_dc, endx_dc-startx_dc, endy_dc-starty_dc), (24,132,255), 2)
                        #     show_image(temp_image)
                        #     """

                        # else:
                        #     startx_dc = p_cx - (l_cx - l_sx) - (crop_length_w)
                        #     starty_dc = p_cy - (l_cy - l_sy) - (crop_length_h)
                        #     endx_dc = p_cx + (l_ex - l_cx) + (crop_length_w)
                        #     endy_dc = p_cy + (l_ey - l_cy ) + (crop_length_h)
                        #     cropped_image_data_collector = pre_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                        #     """
                        #     temp_image = pre_image.copy()
                        #     cv2.rectangle(temp_image, (startx_dc, starty_dc, endx_dc-startx_dc, endy_dc-starty_dc), (24,132,255), 2)
                        #     show_image(temp_image)
                        #     """
                        ###############################################

                        frame_ref_sample_difference.append(dict())
                        frame_ref_sample_difference[sum_index]['ref_frame'] = ref_cropped_frame
                        frame_ref_sample_difference[sum_index]['sample_frame'] = sample_cropped_frame
                        frame_ref_sample_difference[sum_index]['crop'] = (sample_start_x, sample_start_y) #for coordinate rescalling#
                        frame_ref_sample_difference[sum_index]['sample_frame_bbox'] = [norm_s_sy, norm_s_ey, norm_s_sx, norm_s_ex] #for redecision learning#

                        # frame_ref_sample_difference[sum_index]['data_collector'] = cropped_image_data_collector #collecting data for training deep learning model#
                        # frame_ref_sample_difference[sum_index]['ref_coordinates'] = [norm_p_sy, norm_p_ey, norm_p_sx, norm_p_ex, p_cx, p_cy] #for paper-diagram
                        # frame_ref_sample_difference[sum_index]['sample_coordinates'] = [norm_s_sy, norm_s_ey, norm_s_sx, norm_s_ex, l_cx, l_cy] #for paper-diagram

                        sum_index += 1
                        #last_bb.pop(j)
                        drawed_points_pre.append(pre_c[i])
                        drawed_points_last.append(last_c[j])
                        # drawed_points.append(last_c[j])
                        last_bb.pop(j)
                        last_c.pop(j)
                        break

                if not flag_is_succesfull:
                    # drawed_points.append(pre_c[i])
                    pass


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
                    stain_frame_preprocessed=stain_frame_preprocessed,
                    crop_ratio=crop_ratio,
                    sector_index=sector_index,
                    background_color=background_color,
                    activate_debug_images=flag_activate_debug_images
                )

                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel[0], kernel[1]))
                # dilate = cv2.dilate(frame_s, kernel, iterations=1)

                # stain_ratio_px = stain_ratio * 11.35 # (454px -> 40mm | 1px -> 0.088mm | 1mm -> 11.35px) mm to px # Close 21.07.2025
                stain_ratio_px = stain_ratio * 20.244 # (83px -> 4,10mm | 1px -> 0.0494mm | 1mm -> 20.244px) mm to px # Edit 21.07.2025
                frame_preprocess_detect_stain_rso, _, _, _, not_removed_buffer = remove_Small_Object(
                    frame_preprocess_detect_stain.copy(),
                    is_chosen_max_area=False,
                    is_contour_number_for_area=False,
                    ratio=int(stain_ratio_px*stain_ratio_px),
                    counter=not_found_index
                )
                if flag_activate_debug_images:
                    list_temp_save_image = [frame_preprocess_detect_stain_rso]
                    filename = [str(sector_index)+"_3_rso"]
                    save_image(list_temp_save_image, path="temp_files/detect_stains/preprocess_of_Detect_Stain", filename=filename, format="jpg")


                centroid_contour = contour_Centroids(not_removed_buffer)
                if flag_activate_debug_images:
                    stdo(1, "[{}] Unmatched Area-Stain: stain_area_rso(px2):{:.3f} | centroid_contour:{}".format(not_found_index, stain_ratio_px*stain_ratio_px, centroid_contour))

                for _, coords in enumerate(centroid_contour):

                    # if is_middle_object:
                    #     if (
                    #         middle_object_roi[2] < coords[0] < middle_object_roi[3]
                    #     ) and (
                    #         middle_object_roi[0] < coords[1] < middle_object_roi[1]
                    #     ):
                    #         # if flag_activate_debug_images:
                    #             # stdo(1, "[{}] Unmatched Area-Stain-Middle Object Roi: {} | centroid_contour:{}".format(not_found_index, middle_object_roi, coords))
                    #         continue

                    frame_not_found.append(dict())
                    frame_not_found[not_found_index]['ref_frame'] = -1
                    frame_not_found[not_found_index]['sample_frame'] = -1
                    frame_not_found[not_found_index]['crop'] = int(coords[0]), int(coords[1])
                    frame_not_found[not_found_index]['ref_blind_frame'] = None
                    frame_not_found[not_found_index]['sample_blind_frame'] = None
                    not_found_index += 1

                    # if flag_activate_debug_images:
                    #    draw_Circle(display_r, (coords[0], coords[1]), radius=1, color=(255, 0, 0), thickness=-1)
                    #    draw_Circle(display_s, (coords[0], coords[1]), radius=1, color=(255, 0, 0), thickness=-1)


            """
            mask = list()
            flag_is_draw_eq = False

            for i in range(len(all_points)):
                flag_is_draw_eq = False

                if is_middle_object:
                    if (
                        middle_object_roi[2] < all_points[i][0] < middle_object_roi[3]
                    ) and (
                        middle_object_roi[0] < all_points[i][1] < middle_object_roi[1]
                    ):
                        stdo(1, "[{}] Unmatched Area-Blind-Middle Object Roi: {}".format(not_found_index, middle_object_roi))
                        continue

                for j in range(len(drawed_points)):
                    if (all_points[i][0] == drawed_points[j][0]) and (all_points[i][1] == drawed_points[j][1]):
                        flag_is_draw_eq = True
                        break

                if not flag_is_draw_eq:
                    mask.append(i)

            all_points = np.array(all_points)
            blind_points = all_points[mask]

            all_bbox = np.array(all_bbox)
            blind_bbox = all_bbox[mask]

            for index, point in enumerate(blind_points_pre):
                frame_not_found.append(dict())
                frame_not_found[not_found_index]['ref_frame'] = -1
                frame_not_found[not_found_index]['sample_frame'] = -1
                frame_not_found[not_found_index]['crop'] = point[0], point[1]

                #
                blind_points_index = np.where(all_points == point)[0]
                current_bbox = all_bbox[blind_points_index]

                startx_dc = current_bbox[0][0] - (crop_length_w)
                starty_dc = current_bbox[0][1] - (crop_length_h)
                endx_dc = current_bbox[0][0] + current_bbox[0][2] + (crop_length_w)
                endy_dc = current_bbox[0][1] + current_bbox[0][3] + (crop_length_h)
                frame_not_found[not_found_index]['ref_blind_frame'] = pre_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                startx_dc = current_bbox[1][0] - (crop_length_w)
                starty_dc = current_bbox[1][1] - (crop_length_h)
                endx_dc = current_bbox[1][0] + current_bbox[1][2] + (crop_length_w)
                endy_dc = current_bbox[1][1] + current_bbox[1][3] + (crop_length_h)
                frame_not_found[not_found_index]['sample_blind_frame'] = last_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                not_found_index += 1
                stdo(1, "[{}] Unmatched Area-Blind: blind_coords:{} | ref-blind_bbox:{} | sample-blind_bbox:{}".format(not_found_index, point, current_bbox[0], current_bbox[1]))

                if flag_activate_debug_images:
                    draw_Rectangle(display_s, start_point=(int(current_bbox[1][0]), int(current_bbox[1][1])), end_point=(int(current_bbox[1][2]), int(current_bbox[1][3])), color=(255,0,0), thickness=1)
                    draw_Circle(display_s, (point[0], point[1]), radius=1, color=(255, 0, 0), thickness=-1)
                    draw_Text(display_s, text=[str(not_found_index)+":"+str(current_bbox[1][0])+","+str(current_bbox[1][1])], center_point=(int(current_bbox[1][0]), int(current_bbox[1][1]+15)), fontscale=0.3, color=(255,0,0), thickness=1, plain=False)

            """

            mask_pre = []
            blind_points_pre = []
            for i, point in enumerate(all_points_pre):
                # if is_middle_object and middle_object_roi[2] < point[0] < middle_object_roi[3] and middle_object_roi[0] < point[1] < middle_object_roi[1]:
                #     stdo(1, "[{}] Unmatched Area-Blind-Middle Object Roi-Ref: {}".format(i, point))
                #     continue

                if not any(np.array_equal(point, dp) for dp in drawed_points_pre):
                    mask_pre.append(i)
                    blind_points_pre.append(point)
                    # stdo(1, "[{}] Unmatched Area-blind_points_pre: {}".format(i, point))
            blind_points_pre = np.array(blind_points_pre)

            mask_last = []
            blind_points_last = []
            for i, point in enumerate(all_points_last):
                # if is_middle_object and middle_object_roi[2] < point[0] < middle_object_roi[3] and middle_object_roi[0] < point[1] < middle_object_roi[1]:
                #     stdo(1, "[{}] Unmatched Area-Blind-Middle Object Roi-Sample: {}".format(i, point))
                #     continue

                if not any(np.array_equal(point, dp) for dp in drawed_points_last):
                    mask_last.append(i)
                    blind_points_last.append(point)
                    # stdo(1, "[{}] Unmatched Area-blind_points_last: {}".format(i, point))
            blind_points_last = np.array(blind_points_last)


            all_bbox_pre = np.array(all_bbox_pre)
            blind_bbox_pre = all_bbox_pre[mask_pre]

            all_bbox_last = np.array(all_bbox_last)
            blind_bbox_last = all_bbox_last[mask_last]

            for index, point in enumerate(blind_points_pre):
                frame_not_found.append(dict())
                frame_not_found[not_found_index]['ref_frame'] = -1
                frame_not_found[not_found_index]['sample_frame'] = -1
                frame_not_found[not_found_index]['crop'] = point[0], point[1]

                current_bbox = blind_bbox_pre[index]

                startx_dc = current_bbox[0] - (crop_length_w)
                starty_dc = current_bbox[1] - (crop_length_h)
                endx_dc = current_bbox[0] + current_bbox[2] + (crop_length_w)
                endy_dc = current_bbox[1] + current_bbox[3] + (crop_length_h)
                frame_not_found[not_found_index]['ref_blind_frame'] = pre_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                not_found_index += 1

                if flag_activate_debug_images:
                    stdo(1, "[{}] Unmatched Area-Blind: blind_coords:{} | ref-blind_bbox:{}".format(not_found_index, point, current_bbox))

                    draw_Rectangle(display_r, start_point=(int(current_bbox[0]), int(current_bbox[1])), end_point=(int(current_bbox[2]), int(current_bbox[3])), color=(0,255,0), thickness=1)
                    draw_Circle(display_r, (point[0], point[1]), radius=1, color=(0,2550, 0), thickness=-1)
                    # draw_Text(display_r, text=[str(not_found_index)+":"+str(current_bbox[0])+","+str(current_bbox[1])], center_point=(int(current_bbox[0]), int(current_bbox[1]+15)), fontscale=0.3, color=(0,255,0), thickness=1, plain=False)
                    draw_Text(display_r, text=[str(not_found_index)], center_point=(int(current_bbox[0]), int(current_bbox[1]+15)), fontscale=0.3, color=(0,255,0), thickness=1, plain=False)

            for index, point in enumerate(blind_points_last):
                frame_not_found.append(dict())
                frame_not_found[not_found_index]['ref_frame'] = -1
                frame_not_found[not_found_index]['sample_frame'] = -1
                frame_not_found[not_found_index]['crop'] = point[0], point[1]

                current_bbox = blind_bbox_last[index]

                startx_dc = current_bbox[0] - (crop_length_w)
                starty_dc = current_bbox[1] - (crop_length_h)
                endx_dc = current_bbox[0] + current_bbox[2] + (crop_length_w)
                endy_dc = current_bbox[1] + current_bbox[3] + (crop_length_h)
                frame_not_found[not_found_index]['sample_blind_frame'] = last_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

                not_found_index += 1

                if flag_activate_debug_images:
                    stdo(1, "[{}] Unmatched Area-Blind: blind_coords:{} | sample-blind_bbox:{}".format(not_found_index, point, current_bbox))

                    draw_Rectangle(display_s, start_point=(int(current_bbox[0]), int(current_bbox[1])), end_point=(int(current_bbox[2]), int(current_bbox[3])), color=(0,255,0), thickness=1)
                    draw_Circle(display_s, (point[0], point[1]), radius=1, color=(0,2550, 0), thickness=-1)
                    # draw_Text(display_s, text=[str(not_found_index)+":"+str(current_bbox[0])+","+str(current_bbox[1])], center_point=(int(current_bbox[0]), int(current_bbox[1]+15)), fontscale=0.3, color=(0,255,0), thickness=1, plain=False)
                    draw_Text(display_s, text=[str(not_found_index)], center_point=(int(current_bbox[0]), int(current_bbox[1]+15)), fontscale=0.3, color=(0,255,0), thickness=1, plain=False)


            # collecting data for training deep learning model#
            # for bbox in blind_bbox:

            #     stdo(1, "[{}]: Unmatched Area-Blind: blind_bbox:({})".format(not_found_index, bbox))

            #     startx_dc = bbox[0] - (crop_length_w)
            #     starty_dc = bbox[1] - (crop_length_h)
            #     endx_dc = bbox[0] + bbox[2] + (crop_length_w)
            #     endy_dc = bbox[1] + bbox[3] + (crop_length_h)
            #     ref_data_collector = pre_image[ starty_dc:endy_dc, startx_dc:endx_dc ]
            #     sample_data_collector = last_image[ starty_dc:endy_dc, startx_dc:endx_dc ]

            #     frame_not_found.append(dict())
            #     frame_not_found[not_found_index]['ref_blind_frame'] = ref_data_collector
            #     frame_not_found[not_found_index]['sample_blind_frame'] = sample_data_collector
            #     not_found_index += 1

            if flag_activate_debug_images:
                list_frame = [display_r, display_s]
                title_pack = ["display_r", "display_s",]
                save_image(list_frame, path="temp_files/feature_matching_2/CWCFT/", filename=title_pack, format="jpg")

            return frame_ref_sample_difference, frame_not_found