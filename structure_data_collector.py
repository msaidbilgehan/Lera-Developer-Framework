# BUILT-IN LIBRARIES
#import threading
import logging

# CUSTOM LIBRARIES
import libs 
from stdo import get_time
# from structure_data import structure_buffer
from structure_data import Structure_Buffer
from structure_threading import Thread_Object
from image_tools import save_image


class Data_Collector():
    def __init__(
            self, 
            logger_level=logging.INFO, 
            dataset_collection_path="dataset", 
            trigger_quit=None, 
            max_buffer_limit=0
        ):
        self.is_initialized = False
        self.logger_level = logger_level

        self.logger = logging.getLogger("Data_Collector")

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
        self.path = dataset_collection_path + "/" + self.time

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
            set_Deamon=True,
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
        full_path, name, frame = self.buffer_element_parser(self.path, self.buffer_data_list.pop(0))

        save_image(frame, path=full_path, filename=[name], format="png")


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

        full_path = path + "/" + str(bg_color) + "_" + str(id) + "/" + str(sector) + "/" + str(status)

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


