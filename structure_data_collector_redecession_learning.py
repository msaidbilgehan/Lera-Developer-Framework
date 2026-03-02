# BUILT-IN LIBRARIES
#import threading
import logging
import os

import pandas as pd
import numpy as np
import cv2
from skimage import measure


# CUSTOM LIBRARIES
import libs 
from stdo import get_time, stdo
# from structure_data import structure_buffer
from structure_data import Structure_Buffer
from structure_threading import Thread_Object
from image_tools import save_image
from tools import path_control

class Data_Collector_Redecession_Learning():
    def __init__(
            self, 
            logger_level=logging.INFO, 
            dataset_collection_path="dataset", 
            trigger_quit=None, 
            max_buffer_limit=0,
            column_names=[]
        ):
        self.is_initialized = False
        self.logger_level = logger_level

        self.logger = logging.getLogger("Data_Collector_Redecession_Learning")

        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(name)s : %(message)s',
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)
        
        self.buffer_data_list = Structure_Buffer(max_limit=max_buffer_limit)
        self.time = get_time(level=3)
        self.path = dataset_collection_path # + "/" + self.time
        self.column_names = column_names
        
        self.initialize(
            trigger_quit=trigger_quit,
            logger_level=logger_level
        )


    def initialize(
        self, 
        trigger_quit=None, 
        logger_level=logging.INFO,
    ):
        self.logger.info("Data Collector Initializing...")
        
        
        ### ### ### ### ### ### ### #
        # Dataset Collector Thread  #
        ### ### ### ### ### ### ### #
        self.thread_dataset_collector = Thread_Object(
            name="thread_dataset_collector",
            delay=0.01,
            logger_level=logger_level,
            set_Daemon=True,
            run_number=None,
            quit_trigger=trigger_quit
        )
        self.thread_dataset_collector.logger.disabled = True  # is_logger_disabled
        # self.thread_dataset_collector.logger.propagate = not is_logger_disabled

        self.thread_dataset_collector.init(
            params=None,
            connection=self.thread_collector_task
        )

        # To start Camera Steam
        self.thread_dataset_collector.start()


    def buffer_add(self, element):
        # print("[Dataset Collector] Element Added:", element)
        # print("[Dataset Collector] Element len:", len(element))
        self.buffer_data_list.nts_Append(element)


    def buffer_Clear(self):
        self.logger.debug("buffer_Clear Buffer is cleaning...")
        return self.buffer_data_list.clear()


    def quit(self):
        self.logger.info("Exiting from Data Collector ...")
        self.logger.info("Buffer cleaned. {} number of buffer element removed.".format(len(self.buffer_Clear())))
        if hasattr(self, "thread_dataset_collector"):
            self.thread_dataset_collector.statement_quit = True



    def trigger_pause(self):
        return False


    def buffer_Connector(self, connector, custom_buffer = None):
        self.logger.debug(
            "Buffer Connector '{}' is started with custom_buffer: {}".format(
                connector,
                custom_buffer
            )
        )
        if custom_buffer != () or custom_buffer is not None:
            connector(custom_buffer)
        else:
            connector(self.get_information())
        self.logger.debug("Buffer Connector is ended")


    def buffer_information_text_Connector(self, connector):
        connector(self.get_information())
    

    def thread_collector_task(self):
        if len(self.buffer_data_list) > 0:
            # print("self.buffer_data_list:", len(self.buffer_data_list))
            self.write_buffer_to_local()


    def write_buffer_to_local(self):
        

        if not self.buffer_data_list:
            stdo(1, "[WARN] Buffer boş, işlenecek resim yok.")
            return -1

        element = self.buffer_data_list.pop(0)

        # DEBUG: element tipi ve boyutu
        # print(f"[DEBUG] element type: {type(element)}, len: {len(element) if hasattr(element, '__len__') else 'N/A'}")
        # print(f"[DEBUG] element content (kısaltılmış): {str(element)[:200]}...")

        # element doğrudan [original, mask, edge] geliyor mu?
        if not isinstance(element, (list, tuple)) or len(element) < 3:
            stdo(1, f"[WARN] Beklenmeyen element formatı: {element}")
            return -1

        data = element  # [original, mask, edge]

        # boyutlar
        h, w = data[1].shape[:2]

        # white_pixels
        white_pixels = np.sum(data[1] > 0)

        # moments
        m = cv2.moments(data[1])
        centroid_x = m["m10"]/m["m00"] if m["m00"] != 0 else 0
        centroid_y = m["m01"]/m["m00"] if m["m00"] != 0 else 0

        # hu_moments
        hu = cv2.HuMoments(m).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        # edge_white_pixels
        edges_bin = (data[2] > 0).astype(np.uint8)

        # contour özellikleri
        contour_perimeter_sum = np.sum(edges_bin)
        labeled = measure.label(edges_bin, connectivity=2)
        contour_count = labeled.max()
        contour_area_sum = sum(np.sum(labeled == i) for i in range(1, labeled.max()+1))

        features = {
            "id": data[4],
            "start_x": data[5][0],
            "start_y": data[5][1],
            "width": w,
            "height": h,
            "white_pixels": white_pixels,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "hu1_log": hu_log[0],
            "hu2_log": hu_log[1],
            "hu3_log": hu_log[2],
            "hu4_log": hu_log[3],
            "hu5_log": hu_log[4],
            "hu6_log": hu_log[5],
            "hu7_log": hu_log[6],
            "contour_count": contour_count,
            "contour_area_sum": contour_area_sum,
            "contour_perimeter_sum": contour_perimeter_sum,
            "Label": data[3]
        }

        # DataFrame’e çevir
        features_df = pd.DataFrame([features])

        # csv_path = os.path.join(self.path, "features.csv")
        # if os.path.exists(csv_path):
        #     df_existing = pd.read_csv(csv_path)
        #     df_combined = pd.concat([df_existing, features_df], ignore_index=True)
        # else:
        #     df_combined = features_df

        #df_combined.to_csv(csv_path, index=False)
        
        features_df.to_csv(self.path+'/features.csv', mode='a', header=not os.path.exists(self.path+'/features.csv'), index=False)

    @staticmethod
    def buffer_element_parser(path, element):
        """
        str(self.counter_data_collector_pano_index) + "_" + str(counter_data_collector_symbol_index),
        self.com_get_current_object_id, 
        self.pano_sector_index, 
        self.object_background_color, 
        frames['data_collector'],
        "OK"
        """
        name, id, sector, bg_color, frame, status = element

        # full_path = path + "/" + str(bg_color) + "_" + str(id) + "/" + str(sector) + "/" + str(status) # for-pano
        full_path = path + "/" + str(sector) + "/" + str(bg_color) + "/" + str(id) + "/" + str(status)

        return full_path, name, frame


    def get_information(self, filter_keys=None, rename=None, return_only_text=True, return_only_dict=True, text_format="{}:{}", ends="\n"):
        dict_information = dict()
    
        dict_information["verbose_level"] = self.logger_level
        dict_information["is_initialized"] = int(self.is_initialized)
        dict_information["len_max_buffer_stream"] = self.buffer_data_list.max_limit
        dict_information["len_buffer_stream"] = len(self.buffer_data_list)
        
        if filter_keys is not None:
            swap_dict_information = dict()

            if rename is not None:
                for index, filter_key in enumerate(filter_keys):
                    swap_dict_information[rename[index]] = dict_information[filter_key]
            else:
                for filter_key in filter_keys:
                    swap_dict_information[filter_key] = dict_information[filter_key]
                
            dict_information = swap_dict_information

        temp_text = ""
        if return_only_text or not return_only_dict:
            for key, value in dict_information.items():
                temp_text += text_format.format(str(key), str(value)) + ends


        if return_only_text:
            return temp_text
        else:
            if return_only_dict:
                return dict_information
            else:
                return temp_text, dict_information


