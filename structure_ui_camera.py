
### ### ### ### ### ## ### ### ### ###
### ### ### BUILT-IN LIBRARIES ### ###
### ### ### ### ### ## ### ### ### ###

import logging



### ### ### ### ## ## ## ### ### ###
### ### ### CUSTOM LIBRARIES ### ###
### ### ### ### ## ## ## ### ### ###

from stdo import stdo
from structure_ui import Structure_UI, init_and_run_UI
from structure_camera import Camera_Object, CAMERA_FLAGS
from structure_data import Structure_Buffer
from image_tools import save_image, open_image


### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### CAMERA UI CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###

class Structure_Ui_Camera(Structure_UI):

    def __init__(self, *args, obj=None, logger_level=logging.INFO, **kwargs):
        super(Structure_Ui_Camera, self).__init__(*args, **kwargs)
        self.exposure_Time = 40000

        self.UI_File_Path = ""
        self.themes_list = {
            "default": "default.qss"
        }
        self.is_Camera_Stream_Active = False
        self.mouse_Positions = dict()

        ### ### ### ### ###
        ### Constractor ###
        ### ### ### ### ###
        
        self.logger_level = logger_level

        ### ### ### ### ###
        ### ### ### ### ###
        ### ### ### ### ###
        
        self.camera_Instance = None
        self.__buffer_Stream = Structure_Buffer(max_limit=1)

    ### ### ### ### ###
    ### CAMERA APIs ###
    ### ### ### ### ###

    def connect_to_Camera(self, camera_flag, camera_index, buffer_size=10, exposure_time=40000, auto_configure=True):
        #print(f"{camera_flag}, {buffer_size}, {exposure_time}")
        #print("self.is_Camera_Instance_Exist()", self.is_Camera_Instance_Exist())
        if self.is_Camera_Instance_Exist():
            self.camera_Remove()
            self.connect_to_Camera(
                camera_flag,
                camera_index,
                buffer_size,
                exposure_time
            )

        else:
            self.camera_Instance = self.get_Camera(
                camera_flag,
                camera_index,
                buffer_size=buffer_size,
                exposure_time=exposure_time,
                auto_configure=auto_configure
            )

            if self.is_Camera_Instance_Exist():
                self.camera_Instance.stream_Start_Thread(
                    trigger_pause=self.is_Stream_Active,
                    trigger_quit=self.is_Quit_App,
                    number_of_snapshot=-1,
                    delay=0.001
                )
                self.__buffer_Stream.update_Buffer_Size(buffer_size)
                self.camera_Instance.buffer_Connector(self.__buffer_Stream)
                self.stream_Switch(True)

    def get_Camera(self, camera_flag=CAMERA_FLAGS.CV2, camera_index=0, buffer_size=10, exposure_time=40000, auto_configure=True):
        return Camera_Object(
            camera_flag=camera_flag,
            index_device=camera_index,
            auto_configure=auto_configure,
            trigger_quit=self.is_Quit_App,
            trigger_pause=self.is_Stream_Active,
            lock_until_done=False,
            acquisition_framerate=30,
            exposure_time=exposure_time,
            max_buffer_limit=buffer_size,
            logger_level=self.logger_level
        )

    def snapshot(self):
        if self.is_Camera_Instance_Exist():
            return self.camera_Instance.snapshot(is_buffer_enabled=True)
        else:
            return False, None

    def get_Last_Snapshot(self):
        if self.is_Camera_Instance_Exist():
            return self.camera_Instance.last_Snapshot
        else:
            return None

    def is_Camera_Instance_Exist(self):
        return False if self.camera_Instance is None else False if not self.camera_Instance.get_Is_Object_Initialized() else True

    def camera_Remove(self):
        if self.is_Camera_Instance_Exist():
            self.stream_Switch(False)
            #self.camera_Instance.camera_Releaser()
            self.camera_Instance.quit() if self.camera_Instance is not None else None
            # delattr(self, "camera_Instance")
            self.camera_Instance = None
        #return self.is_Stream_Active()

    def is_Stream_Active(self):
        return self.is_Camera_Stream_Active

    def stream_Switch(self, bool=None):
        self.is_Camera_Stream_Active = \
            not self.is_Camera_Stream_Active \
            if bool is None else bool
        return self.is_Camera_Stream_Active

    def set_Camera_Exposure(self, exposure_time):
        self.exposure_Time = exposure_time
        None if self.camera_Instance is None else self.camera_Instance.set_Exposure_Time(
            self.exposure_Time
        )

    ### ### ### ### ###
    ### ### ### ### ###
    ### ### ### ### ###
    
    ### ### ## ### ###
    ### IMAGE APIs ###
    ### ### ## ### ###
    
    def save_Image_Action(self, img, path=None, filename=[], format="png"):
        save_image(img, path=path, filename=filename, format=format)
        
    def load_Image_Action(self, path=None, format="png"):
        if path:
            self.__buffer_Stream.append(
                open_image(path=path, option="cv2-rgb")
            )
            return self.__buffer_Stream.get_Last()
        else:
            return None
    
    ### ### ## ### ###
    ### ### ## ### ###
    ### ### ## ### ###
    
    ### ### ### ### ###
    ### BUFFER APIs ###
    ### ### ### ### ###
    
    def api_Get_Buffered_Image(self, index=-1):
        return self.__buffer_Stream.get_API(index) # if len(self.__buffer_Stream) > index else None
    
    def api_Append_Image_To_Buffer(self, data):
        # if len(self.__buffer_Stream) > index else None
        return self.__buffer_Stream.append(data)
    
    ### ### ## ### ###
    ### ### ## ### ###
    ### ### ## ### ###

### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###

if __name__ == "__main__":
    import sys
    
    # title, Class_UI, run=True, UI_File_Path= "test.ui", qss_File_Path = ""
    stdo(1, "Running {}...".format(__name__))
    
    app, ui = init_and_run_UI(
        title="Camera Developer UI",
        Class_UI = Structure_Ui_Camera,
        UI_File_Path = "" if len(sys.argv) < 2 else sys.argv[1],
        run=True, 
        show_UI=True, 
        is_Maximized=False
    )
