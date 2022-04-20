import time
import numpy as np
import cv2

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
    
    ref_org_h, ref_org_w = refImage.shape
    sample_org_h, sample_org_w = sampleImage.shape
    
    scale_width = int(ref_org_w * (scale_percent / 100))
    scale_height = int(ref_org_h * (scale_percent / 100))
    scale_dim = (scale_width, scale_height)
    refImage = cv2.resize(refImage, scale_dim, interpolation=cv2.INTER_NEAREST)
    
    scale_width = int(sample_org_w * (scale_percent / 100))
    scale_height = int(sample_org_h * (scale_percent / 100))
    scale_dim = (scale_width, scale_height)
    sampleImage = cv2.resize(sampleImage, scale_dim, interpolation=cv2.INTER_NEAREST)
    
    ref_h, ref_w = refImage.shape
    sample_h, sample_w = sampleImage.shape
    
    
    list_timer = list()
    list_ratio = list()
    list_coords = list()
    
    if (show_result == True) and (show_specified_component == int(pattern_id)):
        crop_frame_list = list()
    
    #stdo(1, "[ORG: ({},{}), ({},{})] [RESIZE: ({},{}), ({},{})]".format(ref_org_h, ref_org_w,sample_org_h, sample_org_w,ref_h, ref_w,sample_h, sample_w))
    
    for j in range(0, sample_h, window_step_size):
        for k in range (0, sample_w, window_step_size):
            if (ref_w + k <= sample_w) and (ref_h + j <= sample_h):
                
                window_sampleImage = sampleImage[j:ref_h+j, k:ref_w+k]
                if (show_result == True) and (show_specified_component == int(pattern_id)):
                    crop_frame_list.append(window_sampleImage)
                
                start_time = time.time()
                similarity_match_ratio = process_methods(refImage, window_sampleImage, *process_params)
                stop_time = (time.time() - start_time) * 1000
                list_timer.append(stop_time)
                list_ratio.append(similarity_match_ratio)
                list_coords.append((j,k))
        
    list_ratio = np.array(list_ratio)
    max_mathed_ratio = np.max(list_ratio)
    max_matched_frame_index = np.where(list_ratio == np.amax(list_ratio))
    
    list_coords = np.array(list_coords)
    
    x = int(list_coords[max_matched_frame_index[0][0]][1])
    y = int(list_coords[max_matched_frame_index[0][0]][0])
    w = ref_w
    h = ref_h
    max_matched_frame_coords = np.array([x, y, w, h])
    
    list_timer = np.array(list_timer)
    cvtl = np.sum(list_timer)
    
    if show_result:
        stdo(1, "[{}][{}]: Time:{:.4f} ms | Ratio:{:.5f} | Ratio-Threshold:{:.2f}".format(pattern_id, configuration_id, cvtl, max_mathed_ratio, ratio_threshold))
    
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
        list_name=["refImage", "sampleImage", "max_matched_frame"]
        save_image(list_frame, path="temp_files/Window_Slider", filename=list_name, format="png")
        
    return max_mathed_ratio, max_matched_frame_coords