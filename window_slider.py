import time
import numpy as np
import cv2
from colorama import init, Fore, Style

from tools import time_log, time_list
from image_tools import show_image, save_image
from stdo import stdo


def operation_Window_Slider(
        refImage,
        sampleImage,
        window_step_size=10,
        scale_percent=10,
        process_methods=None,
        process_params=[],
        ratio_threshold=0,
        configuration_id='N/A',
        pattern_id='0',
        show_result=False,
        show_specified_component=0
    ):

    result_dashboard = ""

    ref_org_h, ref_org_w = refImage.shape
    sample_org_h, sample_org_w = sampleImage.shape

    scale_width = int(ref_org_w * (scale_percent / 100))
    scale_height = int(ref_org_h * (scale_percent / 100))
    scale_dim = (scale_width, scale_height)
    # refImage = cv2.resize(refImage, scale_dim, interpolation=cv2.INTER_NEAREST)
    refImage = cv2.resize(refImage, None, fx=(scale_percent / 100), fy=(scale_percent / 100), interpolation=cv2.INTER_NEAREST)

    scale_width = int(sample_org_w * (scale_percent / 100))
    scale_height = int(sample_org_h * (scale_percent / 100))
    scale_dim = (scale_width, scale_height)
    # sampleImage = cv2.resize(sampleImage, scale_dim, interpolation=cv2.INTER_NEAREST)
    sampleImage = cv2.resize(sampleImage, None, fx=(scale_percent / 100), fy=(scale_percent / 100), interpolation=cv2.INTER_NEAREST)

    ref_h, ref_w = refImage.shape
    sample_h, sample_w = sampleImage.shape


    list_timer = list()
    list_ratio = list()
    list_coords = list()

    if show_result:
        crop_frame_list = list()
    # if (show_result == True) and (show_specified_component == int(pattern_id)):
    #    crop_frame_list = list()

    #stdo(1, "[ORG: ({},{}), ({},{})] [RESIZE: ({},{}), ({},{})]".format(ref_org_h, ref_org_w,sample_org_h, sample_org_w,ref_h, ref_w,sample_h, sample_w))

    if ref_h < sample_h and ref_w < sample_w:

        """
        for j in range(0, sample_h, window_step_size):
            for k in range (0, sample_w, window_step_size):
                if (ref_w + k <= sample_w) and (ref_h + j <= sample_h):

                    window_sampleImage = sampleImage[j:ref_h+j, k:ref_w+k]
                    if show_result:
                        crop_frame_list.append(window_sampleImage)
                    # if (show_result == True) and (show_specified_component == int(pattern_id)):
                    #    crop_frame_list.append(window_sampleImage)

                    start_time = time.time()
                    similarity_match_ratio = process_methods(refImage, window_sampleImage, *process_params)
                    stop_time = (time.time() - start_time) * 1000
                    list_timer.append(stop_time)
                    list_ratio.append(similarity_match_ratio)
                    list_coords.append((j,k))
        """

        positions = [
            (j, k) for j in range(0, sample_h - ref_h + 1, window_step_size)
            for k in range(0, sample_w - ref_w + 1, window_step_size)
        ]
        windows = np.array([sampleImage[j:j+ref_h, k:k+ref_w] for j, k in positions])
        for window in windows:
            if show_result:
                crop_frame_list.append(window)
            start_time = time.time()
            similarity_match_ratio = process_methods(refImage, window, *process_params)
            stop_time = (time.time() - start_time) * 1000
            list_timer.append(stop_time)
            list_ratio.append(similarity_match_ratio)


        # TODO: @msyasak disable pattern matching while cropping area re-drawing by user
        list_ratio = np.array(list_ratio)
        max_matched_ratio = np.max(list_ratio)
        max_matched_frame_index = np.where(list_ratio == np.amax(list_ratio))

        list_coords = np.array(positions)

        x = int(list_coords[max_matched_frame_index[0][0]][1])
        y = int(list_coords[max_matched_frame_index[0][0]][0])
        w = ref_w
        h = ref_h
        max_matched_frame_coords = np.array([x, y, w, h])

        list_timer = np.array(list_timer)
        cvtl = np.sum(list_timer)

        if show_result:
            # stdo(1, "[{}][{}]: Ratio:{:.2f} - R.Thr:{:.2f} | T:{:.2f} ms".format(pattern_id, configuration_id, max_matched_ratio, ratio_threshold, cvtl))
            # separator = Fore.BLUE + "[{}][{}]".format(pattern_id, configuration_id) + Style.RESET_ALL
            # result_dashboard = "[{}][{}] \n-Ratio:{:.2f} \n-R.Thr:{:.2f} \n-T:{:.2f}ms".format(pattern_id, configuration_id, max_matched_ratio, ratio_threshold, cvtl)

            temp_pattern_id = str(pattern_id)
            temp_configuration_id = str(configuration_id)
            temp_max_matched_ratio = str(np.round(max_matched_ratio, 2))
            temp_ratio_threshold = str(np.round(ratio_threshold, 2))
            temp_cvtl = str(np.round(cvtl, 2))

            # result_dashboard = "[" + temp_pattern_id + "]" + "[" + temp_configuration_id + "]" + "\n-Ratio:" + temp_max_matched_ratio + "\n-R.Thr:" + temp_ratio_threshold + "\n-T:" + temp_cvtl + "ms"
            result_dashboard = "[" + temp_pattern_id + "]" + "[" + temp_configuration_id + "]" + " Ratio:" + temp_max_matched_ratio + " R.Thr:" + temp_ratio_threshold + " T:" + temp_cvtl + "ms"

            crop_frame_list = np.array(crop_frame_list)
            max_matched_frame = crop_frame_list[max_matched_frame_index[0][0]]

            list_name = list()
            list_frame = list()

            list_frame.append(refImage)
            list_frame.append(sampleImage)
            list_frame.append(max_matched_frame)

            #show_image(list_frame, open_order=1, figsize=(5,5))
            list_name=[str(pattern_id)+"_refImage", str(pattern_id)+"_sampleImage", str(pattern_id)+"_max_matched_frame"]
            save_image(list_frame, path="temp_files/Window_Slider", filename=list_name, format="png")

        if (show_result == True) and (show_specified_component is not None) and (show_specified_component == int(pattern_id)):
            crop_frame_list = np.array(crop_frame_list)
            max_matched_frame = crop_frame_list[max_matched_frame_index[0][0]]

            #max_matched_frame = max_matched_frame.transpose((-2, -1, 0))

            list_name = list()
            list_frame = list()

            list_frame.append(refImage)
            list_frame.append(sampleImage)
            list_frame.append(max_matched_frame)

            #show_image(list_frame, open_order=1, figsize=(5,5))
            list_name=[str(pattern_id)+"_refImage", str(pattern_id)+"_sampleImage", str(pattern_id)+"_max_matched_frame"]
            save_image(list_frame, path="temp_files/Window_Slider", filename=list_name, format="png")

    else:
        max_matched_ratio = -1
        max_matched_frame_coords = np.array([-1, -1, -1, -1])

    return max_matched_ratio, max_matched_frame_coords, result_dashboard