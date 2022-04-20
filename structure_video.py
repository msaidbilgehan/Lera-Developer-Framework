
# BUILT-IN LIBRARIES
#import threading
from enum import Enum
from time import sleep
import logging

# EXTERNAL LIBRARIES
import cv2
#import numpy as np
# from ordered_enum import OrderedEnum

# CUSTOM LIBRARIES
from stdo import stdo
# from structure_data import structure_buffer
from structure_threading import Thread_Object
from structure_data import Structure_Buffer
from tools import time_log, time_list, TIME_FLAGS

# Camera Instance Flags
class VIDEO_SAVE_ALGORITHM_FLAGS(Enum):
    CV2 = 0
    MPEG = 1

class Video_Object():
    __thread_list = dict()
    __camera_object_counter = 0
    __instance_exit_statement = False
    is_auto_detecting = False
    instance_camera = None
    converter = None
    flag_camera = None
    __buffer_Stream = None

    verbose_level = None
    is_initialized = False
    last_snapshot = None
    camera_Last_Statement = False

    def __init__(
            self, 
            camera_flag=VIDEO_SAVE_ALGORITHM_FLAGS.MPEG, 
            logger_level=logging.INFO, 
            auto_configure=True, 
            trigger_pause=None, 
            trigger_quit=None, 
            lock_until_done=True, 
            max_buffer_limit=1, 
            exposure_time=60000, 
            acquisition_framerate_enable=True, 
            acquisition_framerate=25,
            is_auto_detect=False
        ):
        Video_Object.__camera_object_counter += 1

        self.logger = logging.getLogger(
            "Camera_Object {}-{}".format(
                camera_flag.name, 
                self.__camera_object_counter
            )
        )

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
        
        self.__buffer_Stream = Structure_Buffer(max_limit=max_buffer_limit)

        if is_auto_detect:
            self.flag_camera = self.auto_detect(
                trigger_quit=trigger_quit, 
                trigger_pause=trigger_pause, 
                delay=0.1
            )
        
        if camera_flag.value not in VIDEO_SAVE_ALGORITHM_FLAGS._value2member_map_:
            self.logger.error("Non Compatible Video Save Algorithm Flag Error: {}".format(self.flag_camera))
            self.quit()
            # self.__del__()
            return -1
        self.flag_camera = camera_flag

        # Decide if auto init or not
        self.initialize(
            auto_configure=auto_configure, 
            trigger_quit=trigger_quit, 
            trigger_pause=trigger_pause,
            lock_until_done=lock_until_done, 
            exposure_time=exposure_time, 
            acquisition_framerate_enable=acquisition_framerate_enable, 
            acquisition_framerate=acquisition_framerate
        )

    @classmethod
    def __len__(cls):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return cls.__camera_object_counter

    def __del__(self):
        self.logger.info(
            "Camera Object {}-{} Destroyed.".format(
                self.flag_camera,
                self.__camera_object_counter
            )
    )

    # ### ## ### #
    # CLASS APIs #
    # ### ## ### #

    def initialize(
        self, 
        auto_configure=True, 
        trigger_pause=None, 
        trigger_quit=None, 
        lock_until_done=True, 
        exposure_time=60000, 
        acquisition_framerate_enable=True, 
        acquisition_framerate=25
    ):
        self.logger.info("Video Object Initializing...")

        if trigger_pause is None:
            trigger_pause = self.trigger_pause()
                
        if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
            self.api_Baumer_Camera_Create_Instance()

        elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
            self.api_CV2_Camera_Create_Instance()
            #self.api_CV2_Camera_Create_Instance(self.__camera_object_counter)

        # is_initialized = (False or True) is done in initialize function
        # Do Not handle not initialized scenario!
        if self.connect() is True:
            if auto_configure:
                if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
                    self.api_Baumer_Camera_Is_Colorfull()
                    self.api_Baumer_Camera_Configurations(
                        exposure_time=exposure_time, 
                        acquisition_framerate_enable=acquisition_framerate_enable, 
                        acquisition_framerate=acquisition_framerate
                    )
                    
                elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
                    self.logger.warning("No configuration appointed for CV2 camera instance. Nothing done...")

        if lock_until_done and not self.is_Camera_Active():
            self.logger.info("Camera Initializing failed! Calling recovery...")
            self.camera_Recovery(trigger_quit=trigger_quit, trigger_pause=trigger_pause)
            self.camera_Last_Statement = True

    def auto_detect(self, trigger_quit=None, trigger_pause=None, delay=0.1):
        stdo(2, "Still at Development...")
        """
        stdo(1, "Auto Detecting...")
        
        if trigger_pause is None:
            trigger_pause = self.trigger_pause()
            
        self.is_auto_detecting = True
        while not trigger_quit() and trigger_pause() and self.is_auto_detecting:
            sleep(delay)
            if trigger_pause:
                continue
            
            for camera_flag in VIDEO_SAVE_ALGORITHM_FLAGS:
                self.flag_camera = camera_flag
                self.is_initialized = self.connect()
                
            if self.is_initialized:
                self.is_auto_detecting = False
        stdo(1, "Auto Detected.")
        """

    def quit(self):
        self.logger.info("Exiting from camera {}-{} ...".format(self.flag_camera, self.__camera_object_counter))
        
        camera_release_result = self.camera_Releaser()
        if camera_release_result == 0:
            self.logger.info("Camera released.")
        else:
            self.logger.error("Camera Release Error {}".format(camera_release_result))
        self.__instance_exit_statement = True
        self.logger.info("Buffer cleaned. {} number of buffer element removed.".format(len(self.buffer_Clear())))
        
        for thread_id, thread in self.__thread_list.items():
            thread.statement_quit = True
            self.logger.info("Thread {}:{} stopped.".format(thread_id, thread.name))
        self.__thread_list.clear()
        
        self.__del__()

    def trigger_pause(self):
        return False

    def undistort(self):
        self.logger.info("Still at development...")

    # ### ## ### #
    # ### ## ### #
    # ### ## ### #

    # ### ### ### #
    # CAMERA APIs #
    # ### ### ### #

    def connect(self):
        try:
            if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
                self.instance_camera.Connect()
                self.logger.info("Camera Instance is Successful, The Instance is {}".format(
                    self.instance_camera))
            elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
                self.logger.warning(
                    "No need to connect CV2 camera instance. Nothing done...")
            self.is_initialized = True
        except neoapi.neoapi.NotConnectedException as error:
            self.logger.error(
                "BAUMER Camera No Device NotConnectedException: {}".format(error)
            )
            self.is_initialized = False
        except neoapi.neoapi.FeatureAccessException as error:
            self.logger.error(
                "BAUMER Camera No Device FeatureAccessException: {}".format(error)
            )
            self.is_initialized = False
        except Exception as error:
            self.logger.error("Camera Connect Error: {}".format(error))
            self.is_initialized = False

        return self.is_initialized

    def snapshot(self, is_buffer_enabled=True):
        is_successfull = False
        local_last_snapshot = None

        if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
            try:
                time_log(id=repr(VIDEO_SAVE_ALGORITHM_FLAGS.MPEG))
                local_last_snapshot = self.instance_camera.GetImage().GetNPArray()
                time_log(id=repr(VIDEO_SAVE_ALGORITHM_FLAGS.MPEG))
                self.logger.debug("{} Time passed: {:.3f} ms".format(
                    repr(VIDEO_SAVE_ALGORITHM_FLAGS.MPEG),
                    time_list[repr(VIDEO_SAVE_ALGORITHM_FLAGS.MPEG)][TIME_FLAGS.PASSED]
                )
                )
                if local_last_snapshot is not None:
                    if len(local_last_snapshot) != 0:
                        self.logger.debug("snapshot get successful frame!")
                        is_successfull = True
                    else:
                        self.logger.debug(
                            "snapshot get unsuccessful frame (0 length)!")
                        is_successfull = False
                else:
                    self.logger.debug(
                        "snapshot get unsuccessful frame (None)!")
                    is_successfull = False

            except neoapi.NotConnectedException as error:
                self.logger.error(
                    "BAUMER Camera NotConnectedException: {}".format(error)
                )
                is_successfull = False
            except Exception as error:
                self.logger.error(
                    "BAUMER Camera Get Snapshot Error: {}".format(error)
                )
                is_successfull = False

            #if self.is_initialized == False:
            #    is_successfull = False

        elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
            # cap.open(0, cv::CAP_DSHOW)
            # Returns (False, None) in fail, (True, image) otherwise
            is_successfull, local_last_snapshot = self.instance_camera.read()

            # TODO: At connection lost, re-initialize (camera recovery) is not working. Someone sad to use grab+retrieve instead of read which read does both and gives decoded frame rather than 2 step. Just try it!
            # https://answers.opencv.org/question/21069/cant-grab-frame-from-webcamera/
            # https://stackoverflow.com/questions/57716962/difference-between-video-capture-read-and-grab
            # cap.grab();
            # cap.retrieve(&frame, 0);

        if is_buffer_enabled and is_successfull:
            #print(local_last_snapshot.shape, type(local_last_snapshot), self.camera_pack)
            #cv2.imshow("TP", cv2.resize(local_last_snapshot,None,fx=0.2,fy=0.2))
            self.__buffer_Stream.append(local_last_snapshot)

        self.last_snapshot = local_last_snapshot
        self.last_is_successfull = is_successfull
        return is_successfull, local_last_snapshot

    def stream_Start_Thread(self, trigger_pause=None, trigger_quit=None, number_of_snapshot=-1, delay=0.001):
        self.__thread_list["stream_Start_Thread"] = Thread_Object(
            name="Camera_Object.stream_Start_Thread",
            delay=0.0001,
            logger_level=self.logger.getEffectiveLevel(),
            set_Deamon=True,
            run_number=1,
            quit_trigger=trigger_quit
        )
        self.__thread_list["stream_Start_Thread"].init(
            params=[
                trigger_pause,
                trigger_quit,
                number_of_snapshot,
                delay
            ],
            task=self.stream_Start
        )
        self.__thread_list["stream_Start_Thread"].start()

        return self.__thread_list["stream_Start_Thread"]

    def stream_Start(self, trigger_pause=None, trigger_quit=None, number_of_snapshot=-1, delay=0.001):
        if trigger_pause is None:
            trigger_pause = self.trigger_pause()

        while number_of_snapshot != 0:
            if trigger_pause():
                if not self.is_Camera_Active():
                    self.logger.debug(
                        "stream_Start get Camera NOT Active. 'camera_Recovery' calling...")
                    self.camera_Recovery(
                        trigger_quit=trigger_quit, trigger_pause=trigger_pause)
                # (self.instance_camera is None) Means we got quit trigger from camera recovery function, thus quit from thread.
                if self.instance_camera is None:
                    self.logger.debug(
                        "stream_Start get instance_camera as None caused by active quit trigger. Quiting...")
                    return 0  # break # Lets make it clear at quit statement

                is_successfull, _ = self.snapshot(is_buffer_enabled=True)
                # print(is_successfull, type(frame), frame.shape)

                if not is_successfull:
                    self.logger.debug(
                        "stream_Start get NOT successfull frame. 'camera_Recovery' calling...")
                    self.camera_Recovery(
                        trigger_quit=trigger_quit, trigger_pause=trigger_pause)

                if number_of_snapshot > 0:
                    number_of_snapshot -= 1
            else:
                pause_delay = 1
                self.logger.debug(
                    "stream_Start get trigger_pause. Waiting {} sec.".format(pause_delay))
                sleep(pause_delay)

            if trigger_quit is not None:
                if trigger_quit():
                    self.logger.debug(
                        "stream_Start quit trigger activated! Quiting...")
                    self.quit()
                    return 0  # break # Lets make it clear at quit statement
            elif self.__instance_exit_statement:
                self.logger.debug("{}-{} Camera Instance stream_Start Exit Statement activated! Quiting...".format(
                    self.flag_camera,
                    self.__camera_object_counter
                )
                )
                return 0  # break # Lets make it clear at quit statement
            # stdo(1, "Buffer: {}".format(len(self.__buffer_Stream)))
            sleep(delay)
        return 0

    def stream_Connector(self, connection, trigger_pause=None, trigger_quit=None, number_of_snapshot=-1, auto_pop=True, pass_broken=True, delay=0.001):
        if trigger_pause is None:
            trigger_pause = self.trigger_pause()

        while number_of_snapshot != 0 and trigger_pause():
            if len(self.__buffer_Stream) > 0:
                # call ui.set_display function that given frame-buffer as parameter

                frame = self.__buffer_Stream.pop() if auto_pop else self.__buffer_Stream.get_Last()

                if not pass_broken:
                    self.logger.debug("stream_Connector get frame!")
                    connection(frame)
                else:
                    self.logger.debug("stream_Connector get broken frame!")
                    if len(frame.shape) == 3:
                        if frame.shape[2] == 3:
                            self.logger.debug(
                                "stream_Connector Broken frame has 3 channel, sending to connection")
                            connection(frame)
                        else:
                            self.logger.debug(
                                "stream_Connector Broken frame passing...")
                    else:
                        self.logger.debug(
                            "stream_Connector Broken frame passing...")

                if number_of_snapshot > 0:
                    number_of_snapshot -= 1

            if trigger_quit is not None:
                if trigger_quit():
                    self.logger.debug(
                        "stream_Connector quit trigger activated! Breaking the loop...")
                    # self.quit()
                    break
            elif self.__instance_exit_statement:
                self.logger.debug("{}-{} Camera Instance stream_Start Exit Statement activated! Quiting...".format(
                    self.flag_camera,
                    self.__camera_object_counter
                )
                )
                return 0  # break # Lets make it clear at quit statement
            sleep(delay)
        else:
            if trigger_pause():
                pause_delay = 1
                self.logger.debug(
                    "stream_Connector get trigger_pause. Waiting {} sec.".format(pause_delay))
                # sleep(pause_delay)
        return 0

    def stream_Returner(self, auto_pop=True, pass_broken=True):
        if len(self.__buffer_Stream) > 0:
            frame = self.__buffer_Stream.pop() if auto_pop else self.__buffer_Stream.get_Last()
            if pass_broken:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 3:
                        return frame
                    else:
                        self.logger.debug(
                            "stream_Connector Broken frame has less than 3 channel, passing..."
                        )
                else:
                    self.logger.debug(
                        "stream_Connector Broken frame passing..."
                    )
            else:
                return frame
        return None

    def is_Camera_Active(self):
        if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
            self.camera_Last_Statement, _ = self.snapshot(True)

            if self.camera_Last_Statement:
                self.logger.debug(
                    "is_Camera_Active get successfull frame from BAUMER Camera. Returning True...")
                return self.camera_Last_Statement
            else:
                self.logger.debug(
                    "is_Camera_Active get NOT successfull frame from BAUMER Camera. Returning False...")
                return self.camera_Last_Statement
        
        self.logger.error(
            "is_Camera_Active returning False at the end of control. Unplanned situation!")
        self.camera_Last_Statement = False
        return self.camera_Last_Statement

    def camera_Recovery(self, trigger_quit=None, trigger_pause=None, delay=1):
        if trigger_pause is None:
            trigger_pause = self.trigger_pause()

        while not trigger_pause():
            self.camera_Releaser()
            # Call create instance with lock_until_done = True for handeling infinite loop problem
            self.initialize(lock_until_done=False)
            if self.is_Camera_Active():
                self.logger.debug(
                    "camera_Recovery get Camera active. Breaking the loop")
                break

            if trigger_quit is not None:
                if trigger_quit():
                    self.logger.debug(
                        "camera_Recovery quit trigger activated! Returning None...")
                    #self.quit()
                    return None
            elif self.__instance_exit_statement:
                self.logger.debug("{}-{} Camera Instance stream_Start Exit Statement activated! Quiting...".format(
                    self.flag_camera,
                    self.__camera_object_counter
                )
                )
                return 0  # break # Lets make it clear at quit statement
            sleep(delay)

    def camera_Releaser(self):
        #self.logger.debug("camera_Releaser is releasing the camera...")
        if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
            self.instance_camera.release()
            self.logger.debug("camera_Releaser CV2 Camera is released...")
        elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
            if self.instance_camera.IsGrabbing():
                self.instance_camera.StopGrabbing()
            self.instance_camera.Close()
            self.logger.debug("camera_Releaser MPEG Camera is released...")
        else:
            self.logger.debug("camera_Releaser No need action...")
        return 0

    def set_Exposure_Time(self, value):
        if self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.MPEG:
            self.baumer_Set_Camera_Exposure(value)
        elif self.flag_camera is VIDEO_SAVE_ALGORITHM_FLAGS.CV2:
            self.cv2_Set_Camera_Exposure(value)
    
    def api_Baumer_Camera_Configurations(self, exposure_time=60000, acquisition_framerate_enable=True, acquisition_framerate=25, crash_protection = False):
        self.baumer_Camera_Configurations_Protected(
            exposure_time = exposure_time, 
            acquisition_framerate_enable= acquisition_framerate_enable, 
            acquisition_framerate=acquisition_framerate
        ) if crash_protection else self.baumer_Camera_Configurations(
            exposure_time=exposure_time,
            acquisition_framerate_enable=acquisition_framerate_enable,
            acquisition_framerate=acquisition_framerate
        )

    # ### ### ### #
    # ### ### ### #
    # ### ### ### #

    # ### ### ### ### ### ### #
    # THIRD PARTY CAMERA APIs #
    # ### ### ### ### ### ### #
    
    def api_CV2_Camera_Create_Instance(self, index_device = 0):
        try:
            # self.instance_camera = cv2.VideoCapture(index_device)
            # TODO: Recovery is not working for CV2 camera, check what we need for re-initialization and what CAP_DSHOW flag stands for
            self.instance_camera = cv2.VideoCapture(
                index_device, 
                cv2.CAP_DSHOW
            )
            # self.is_initialized = True
        except Exception as error:
            self.logger.error("CV2 Camera Instance Creation Error: {}".format(error))
            self.is_initialized = False

    def api_Baumer_Camera_Create_Instance(self):
        global neoapi
        try:
            import neoapi
            self.instance_camera = neoapi.Cam()
        except Exception as error:
            self.logger.error("BAUMER Camera Instance Creation Error: {}".format(error))
            self.is_initialized = False

    def api_Baumer_Camera_Is_Colorfull(self):
        self.logger.info("Pixel Format list is {}".format(self.instance_camera.f.PixelFormat.GetString()))

        try:
            if self.instance_camera.f.PixelFormat.GetEnumValueList().IsReadable("BGR8"):
                self.is_colored = True
                self.instance_camera.f.PixelFormat.SetString("BGR8")
            elif self.instance_camera.f.PixelFormat.GetEnumValueList().IsReadable("Mono8"):
                self.instance_camera.f.PixelFormat.SetString("Mono8")
                self.is_colored = False
            else:
                self.logger.error("BAUMER Camera: Error, no supported pixelformat")
                return -1
            return self.is_colored
        except neoapi.neoapi.FeatureAccessException as error:
            self.logger.error("BAUMER Camera api_Baumer_Camera_Is_Colorfull FeatureAccessException: {}".format(error))
            self.is_initialized = False
        except Exception as error:
            self.logger.error("BAUMER Camera api_Baumer_Camera_Is_Colorfull Exception: {}".format(error))
            self.is_initialized = False
 
    def baumer_Camera_Configurations_Protected(self, exposure_time=60000, acquisition_framerate_enable=True, acquisition_framerate=25):
        #self.instance_camera.f.ExposureAuto.Set(True)
        try:
            self.instance_camera.f.ColorTransformationAuto.SetString("Continuous")
            self.set_Exposure_Time(exposure_time)
            #self.instance_camera.f.ExposureTime.Set(exposure_time)
            self.instance_camera.f.AcquisitionFrameRateEnable.value = acquisition_framerate_enable
            self.instance_camera.f.AcquisitionFrameRate.value = acquisition_framerate
            self.is_initialized = True
        except neoapi.neoapi.FeatureAccessException as error:
            self.logger.error("BAUMER Camera api_Baumer_Camera_Configurations FeatureAccessException: {}".format(error))
            self.is_initialized = False
        except Exception as error:
            self.logger.error("BAUMER Camera api_Baumer_Camera_Configurations Exception: {}".format(error))
            self.is_initialized = False

        self.logger.info(
            "BAUMER Camera: Exposure Time is '{}', Acquisition Frame Rate Enable is '{}', Acquisition Frame Rate is '{}'".format(
                self.instance_camera.f.ExposureTime.Get(),  # exposure_time,
                self.instance_camera.f.AcquisitionFrameRateEnable.value,
                self.instance_camera.f.AcquisitionFrameRate.value
            )
        )
        sleep(3)
        return 0
        
    def baumer_Camera_Configurations(self, exposure_time=60000, acquisition_framerate_enable=True, acquisition_framerate=25):
        #self.instance_camera.f.ExposureAuto.Set(True)
        self.instance_camera.f.ColorTransformationAuto.SetString("Continuous")
        self.set_Exposure_Time(exposure_time)
        # self.instance_camera.f.ExposureTime.Set(exposure_time)
        self.instance_camera.f.AcquisitionFrameRateEnable.value = acquisition_framerate_enable
        self.instance_camera.f.AcquisitionFrameRate.value = acquisition_framerate

        self.logger.info(
            "BAUMER Camera: Exposure Time is '{}', Acquisition Frame Rate Enable is '{}', Acquisition Frame Rate is '{}'".format(
                self.instance_camera.f.ExposureTime.Get(),  # exposure_time,
                self.instance_camera.f.AcquisitionFrameRateEnable.value,
                self.instance_camera.f.AcquisitionFrameRate.value
            )
        )
        sleep(3)
        self.is_initialized = True
        return 0

    def baumer_Set_Camera_Exposure(self, value):
        self.instance_camera.f.ExposureTime.Set(value)

    def cv2_Set_Camera_Exposure(self, value):
        # self.instance_camera.set(cv2.CAP_PROP_FPS, )
        self.instance_camera.set(cv2.CAP_PROP_EXPOSURE, value)
        
    # ### ### ### ### ### ### #
    # ### ### ### ### ### ### #
    # ### ### ### ### ### ### #

    
    def set_Buffer_Size(self, size):
        self.__buffer_Stream.update_Buffer_Size(size)
    
    def get_Buffer_Size(self):
        return len(self.__buffer_Stream)
        
    def get_Buffered_Image(self, index=0):
        return self.__buffer_Stream.get(index) if len(self.__buffer_Stream) > index else None
    
    def buffer_Clear(self):
        self.logger.debug("buffer_Clear Buffer is cleaning...")
        return self.__buffer_Stream.clear()

    def buffer_Connector(self, connector, custom_buffer=None):
        """
        self.logger.debug(
            "Buffer Connector '{}' is started with custom_buffer: {}".format(
                connector,
                custom_buffer
            )
        )
        """
        connector(custom_buffer) \
            if custom_buffer != () or custom_buffer is not None \
            else connector(self.get_information())
        #self.logger.debug("Buffer Connector is ended")

    def buffer_information_text_Connector(self, connector):
        connector(self.get_information())
    
    # ### ### ### #
    # ### ### ### #
    # ### ### ### #
    
    # ### ## ### #
    # OTHER APIs #
    # ### ## ### #

    def get_information(self, filter_keys=None, rename=None, return_only_text=True, return_only_dict=True, text_format="{}:{}", ends="\n"):
        dict_information = dict()
    
        dict_information["flag_camera"] = self.flag_camera
        dict_information["verbose_level"] = self.verbose_level
        dict_information["is_initialized"] = int(self.is_initialized)
        dict_information["is_Camera_Active"] = int(self.camera_Last_Statement)
        dict_information["buffer_max"] = self.__buffer_Stream.max_limit
        dict_information["buffer_len"] = len(self.__buffer_Stream)
        
        if filter_keys is not None:
            swap_dict_information = dict()

            if rename is not None:
                for index, filter_key in enumerate(filter_keys):
                    swap_dict_information[rename[index]] = dict_information[filter_key]
            else:
                for filter_key in filter_keys:
                    swap_dict_information[filter_key] = dict_information[filter_key]
                
            dict_information = swap_dict_information

        if return_only_text or not return_only_dict:
            temp_text = ""
            for key, value in dict_information.items():
                temp_text += text_format.format(str(key), str(value)) + ends

        if return_only_text:
            return temp_text
        else:
            if return_only_dict:
                return dict_information
            else:
                return temp_text, dict_information

    # ### ## ### #
    # ### ## ### #
    # ### ## ### #
