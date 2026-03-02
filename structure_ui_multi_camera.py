
### ### ### ### ### ## ### ### ### ###
### ### ### BUILT-IN LIBRARIES ### ###
### ### ### ### ### ## ### ### ### ###
import logging


### ### ### ### ## ## ## ### ### ###
### ### ### CUSTOM LIBRARIES ### ###
### ### ### ### ## ## ## ### ### ###
from stdo import stdo
from image_tools import save_image, open_image
from structure_ui import Structure_UI, init_and_run_UI
from structure_multi_camera import Camera_Object, CAMERA_FLAGS
from structure_data import Structure_Buffer
from image_manipulation import image_Flip, image_Rotate


### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### CAMERA UI CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###

class Structure_Ui_Multi_Camera(Structure_UI):

    def __init__(self, *args, obj=None, logger_level=logging.INFO, camera_count=3, **kwargs):
        super(Structure_Ui_Multi_Camera, self).__init__(*args, **kwargs)
        self.exposure_Time = 40000

        self.UI_File_Path = ""
        self.themes_list = {
            "default": "default.qss"
        }
        self.is_Camera_Stream_Active = False
        self.is_Camera_Stream_Active_Cam2 = False
        self.mouse_Positions = dict()

        ### ### ### ### ###
        ### Constructor ###
        ### ### ### ### ###
        
        self.logger_level = logger_level

        ### ### ### ### ###
        ### ### ### ### ###
        ### ### ### ### ###
        
        self.camera_count = camera_count
        self.camera_Instance_Array = None
        
        self.buffer_Stream = dict()
        self.buffer_Dict = dict()
        for id in range(self.camera_count):
            self.buffer_Stream['cam'+str(id)] = Structure_Buffer(max_limit=1)
            self.buffer_Dict['cam'+str(id)+"_is_Camera_Stream_Active"] = False

    ### ### ### ### ###
    ### CAMERA APIs ###
    ### ### ### ### ###

    def connect_to_Camera(self, camera_flag, camera_index, buffer_size=10, exposure_time=40000, auto_configure=True, extra_params=[]):
        #print(f"{camera_flag}, {buffer_size}, {exposure_time}")
        #print("self.is_Camera_Instance_Exist()", self.is_Camera_Instance_Exist())
        
        if self.is_Camera_Instance_Exist():
            self.camera_Remove()
            self.connect_to_Camera(
                camera_flag,
                camera_index,
                buffer_size,
                exposure_time,
                extra_params
            )

        else:
            self.camera_Instance_Array = self.get_Camera(
                camera_flag,
                camera_index,
                buffer_size=buffer_size,
                exposure_time=exposure_time,
                auto_configure=auto_configure,
                extra_params=extra_params
            )

            if self.is_Camera_Instance_Exist():
                
                # self.camera_Instance_Array.stream_Start_Thread(
                #     trigger_pause=self.is_Stream_Active,
                #     trigger_quit=self.is_Quit_App,
                #     number_of_snapshot=-1,
                #     delay=0.001,
                #     cam_id=0
                # )
                
                # self.camera_Instance_Array.stream_Start_Thread_2(
                #     trigger_pause=self.is_Stream_Active_Cam2,
                #     trigger_quit=self.is_Quit_App,
                #     number_of_snapshot=-1,
                #     delay=0.001,
                #     cam_id=1
                # )
                
                for id in range(camera_index):
                    
                    self.camera_Instance_Array.stream_Start_Multiple_Thread(
                        trigger_pause=self.is_Multiple_Stream_Active(id),
                        trigger_quit=self.is_Quit_App,
                        number_of_snapshot=-1,
                        delay=0.001,
                        cam_id=id
                    )
                    
                    self.buffer_Stream['cam'+str(id)].update_Buffer_Size(buffer_size)
                    self.camera_Instance_Array.buffer_Connector(
                        custom_buffer=self.buffer_Stream['cam'+str(id)], 
                        custom_buffer_id=id
                    )
                    self.stream_Multiple_Switch(True, cam_id=id)
                    
                # self.stream_Switch_Cam1(True)
                # self.stream_Switch_Cam2(True)
                    
    def get_Camera(self, camera_flag=CAMERA_FLAGS.CV2, camera_index=0, buffer_size=10, exposure_time=40000, auto_configure=True, extra_params=[]):
        return Camera_Object(
            camera_flag=camera_flag,
            index_device=camera_index,
            auto_configure=auto_configure,
            extra_params=extra_params,  # cv2.CAP_DSHOW == 700
            trigger_quit=self.is_Quit_App,
            trigger_pause=self.is_Multiple_Stream_Active(cam_id=0),
            lock_until_done=False,
            acquisition_framerate=30,
            exposure_time=exposure_time,
            max_buffer_limit=buffer_size,
            logger_level=self.logger_level
        )

    def snapshot(self, cam_id=0):
        if self.is_Camera_Instance_Exist():
            return self.camera_Instance_Array.snapshot(cam_id=cam_id, is_buffer_enabled=True)
        else:
            return False, None

    def get_Last_Snapshot(self):
        if self.is_Camera_Instance_Exist():
            return self.camera_Instance_Array.last_Snapshot
        else:
            return None

    def is_Camera_Instance_Exist(self):
        return False if self.camera_Instance_Array is None else False if not self.camera_Instance_Array.get_Is_Object_Initialized() else True

    def camera_Remove(self):
        if self.is_Camera_Instance_Exist():
            # self.stream_Switch_Cam1(False)
            # self.stream_Switch_Cam2(False)
            for id in range(self.camera_count):
                self.stream_Multiple_Switch(False, cam_id=id)
            #self.camera_Instance_Array.camera_Releaser()
            self.camera_Instance_Array.quit() if self.camera_Instance_Array is not None else None
            # delattr(self, "camera_Instance_Array")
            self.camera_Instance_Array = None
        #return self.is_Stream_Active()

    def is_Stream_Active(self):
        return self.is_Camera_Stream_Active
    
    def stream_Switch_Cam1(self, bool=None):
        self.is_Camera_Stream_Active = \
            not self.is_Camera_Stream_Active \
            if bool is None else bool
        return self.is_Camera_Stream_Active
    
    def is_Stream_Active_Cam2(self):
        return self.is_Camera_Stream_Active_Cam2
    
    def stream_Switch_Cam2(self, bool=None):
        self.is_Camera_Stream_Active_Cam2 = \
            not self.is_Camera_Stream_Active_Cam2 \
            if bool is None else bool
        return self.is_Camera_Stream_Active_Cam2
    
    def is_Multiple_Stream_Active(self, cam_id=0):
        return self.buffer_Dict['cam'+str(cam_id)+"_is_Camera_Stream_Active"]
    
    def stream_Multiple_Switch(self, bool=None, cam_id=0):
        self.buffer_Dict['cam'+str(cam_id)+"_is_Camera_Stream_Active"] = not self.buffer_Dict['cam'+str(cam_id)+"_is_Camera_Stream_Active"] if bool is None else bool
        return self.buffer_Dict['cam'+str(cam_id)+"_is_Camera_Stream_Active"]
    

    def set_Camera_Exposure(self, cam_id=0, exposure_time=20000):
        
        self.exposure_Time = exposure_time
        
        if self.camera_Instance_Array is None:
            self.exposure_Time = None
        
        else:
            self.camera_Instance_Array.set_Exposure_Time(
                cam=self.camera_Instance_Array.buffer_Camera.get(cam_id),
                exposure_time=self.exposure_Time
            )
        
    def get_Camera_Exposure(self, cam_id=0):
        
        if self.camera_Instance_Array is None:
            return self.exposure_Time
            
        else:
            return self.camera_Instance_Array.get_Exposure_Time(
                cam=self.camera_Instance_Array.buffer_Camera.get(cam_id)
            )

    ### ### ### ### ###
    ### ### ### ### ###
    ### ### ### ### ###
    
    ### ### ### ### ###
    ### BUFFER APIs ###
    ### ### ### ### ###
    
    def api_Get_Buffered_Image(self, index=-1, buffer_id=0):
        return self.buffer_Stream['cam'+str(buffer_id)].get_API(index)
    
    def api_Append_Image_To_Buffer(self, data, buffer_id=0):
        return self.buffer_Stream['cam'+str(buffer_id)].append(data)
    
    ### ### ## ### ###
    ### ### ## ### ###
    ### ### ## ### ###

    ### ### ## ### ###
    ### IMAGE APIs ###
    ### ### ## ### ###

    def save_Image_Action(self, img, path=None, filename=[], format="png"):
        save_image(img, path=path, filename=filename, format=format)

    def load_Image_Action(self, buffer_id=0, path=None, format="png"):
        if path:
            self.buffer_Stream['cam'+str(buffer_id)].append(
                open_image(path=path, option="cv2-rgb")
            )
            return self.buffer_Stream['cam'+str(buffer_id)].get_Last()
        else:
            return None

    def flip_Image_Action(self, img=None, buffer_id=0, task='horizontaly'):
        self.buffer_Stream['cam'+str(buffer_id)].append(
            image_Flip(img, task)
        )
        return self.buffer_Stream['cam'+str(buffer_id)].get_Last()
    
    def rotate_Image_Action(self, img=None, buffer_id=0,):
        self.buffer_Stream['cam'+str(buffer_id)].append(
            image_Rotate(img)
        )
        return self.buffer_Stream['cam'+str(buffer_id)].get_Last()
        
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
        Class_UI = Structure_Ui_Multi_Camera,
        UI_File_Path = "" if len(sys.argv) < 2 else sys.argv[1],
        run=True, 
        show_UI=True, 
        is_Maximized=False
    )
