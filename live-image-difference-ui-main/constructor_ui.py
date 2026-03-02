from collections import deque
import cv2
import numpy as np
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PyQt5 import QtCore

import libs
from stdo import stdo
from structure_ui import init_and_run_UI, Graphics_View, ROI_Rectangle  # , Structure_UI, init_UI
from structure_camera import CAMERA_FLAGS
from structure_ui_camera import Structure_Ui_Camera
from structure_threading import Thread_Object
from qt_tools import qtimer_Create_And_Run


class Ui_Image_Difference(Structure_Ui_Camera):
    def __init__(self, *args, obj=None, **kwargs):
        super(Ui_Image_Difference, self).__init__(*args, **kwargs)

        ### ### ### ### ###
        ### Constructor ###
        ### ### ### ### ###
        self.__thread_Dict = dict()
        self.history_image = deque(maxlen=2)
        self.buffer_graphicsView_Mask = None
        self.buffer_graphicsView_Difference = None
        self.time_counter = 0
        self.history_pixel = {
            "time": deque(maxlen=100),
            "pixel": deque(maxlen=100),
        }

        ### ### ### ### ###
        ### ### Init ### ##
        ### ### ### ### ###
        self.init()

        self.configure_Button_Connections()
        self.configure_Other_Settings()
        self.init_QTimers()

    ### ### ## ### ###
    ### OVERWRITES ###
    ### ### ## ### ###

    def init(self):
        self.connect_to_Camera(
            CAMERA_FLAGS.CV2,
            # self.spinBox_Buffer_Size.value(),
            0,
            10,
            self.exposure_Time
        )
        self.graphicsView_Camera.init_Render_QTimer(
            connector_stream=self.camera_Instance.stream_Returner,
            delay=10
        )
        self.graphicsView_Camera_Mask.init_Render_QTimer(
            connector_stream=lambda: self.buffer_graphicsView_Mask,
            delay=75
        )
        self.graphicsView_Difference.init_Render_QTimer(
            connector_stream=lambda: self.buffer_graphicsView_Difference,
            delay=75
        )

        self.qtimer_stream_Flow = qtimer_Create_And_Run(
            self,
            self.stream_Flow,
            100
        )
        self.qtimer_plot_history_graph = qtimer_Create_And_Run(
            self,
            self.plot_history_graph,
            100
        )

        self.camera_Instance.api_Set_Camera_Size(
            resolution=(1920, 1080) # 5 px
            # resolution=(720, 1280) # 4 px
            # resolution=(3040, 4056)  # 10 px
        ) if self.camera_Instance is not None else None

        self.graph_pen = pg.mkPen(
            color=(255, 0, 0),
            width=3,
            style=QtCore.Qt.DashLine
        )
        self.widget_Historical_Graph_CH1.setBackground("black")

    def init_QTimers(self, *args, **kwargs):
        super(Ui_Image_Difference, self).init_QTimers(*args, **kwargs)

    def configure_Button_Connections(self):
        self.pushButton_Set_Exposure.clicked.connect(
            lambda: self.set_Camera_Exposure(
                self.spinBox_Exposure_Time.value()
            )
        )
        self.pushButton_Load_Image.clicked.connect(
            lambda: [
                self.stream_Switch(False),
                self.graphicsView_Renderer(
                    self.graphicsView_Camera,
                    self.load_Image_Action(
                        path=self.QFileDialog_Event(
                            "getOpenFileName",
                            [
                                "Open file",
                                "",
                                "Image files (*.png *.jpg *.jpeg)"
                            ]
                        )[0]
                    )
                ),
            ]
        )
        self.pushButton_Save_Image.clicked.connect(
            lambda: self.save_Image_Action(
                # self.camera_Instance.stream_Returner(auto_pop=False),
                img=self.api_Get_Buffered_Image(),
                path=None,
                filename=[],
                format="png"
            )
        )
        self.pushButton_Connect_to_Camera.clicked.connect(
            lambda: self.connect_to_Camera(
                CAMERA_FLAGS.CV2,
                # self.spinBox_Buffer_Size.value(),
                10,
                self.exposure_Time
            )
        )
        self.pushButton_Remove_the_Camera.clicked.connect(
            self.camera_Remove
        )
        self.pushButton_Stream_Switch.clicked.connect(
            lambda: self.stream_Switch()
        )
        
    def stream_Flow(self):
        if self.camera_Instance is not None:
            cache = self.camera_Instance.stream_Returner(
                auto_pop=False,
                pass_broken=True
            )
            if cache is not None:
                self.history_image.append(
                    cache
                )
            # if len(self.history_image) == 2:
            #     print(self.history_image[0])
            #     print(self.history_image[1])

            if len(self.history_image) == 2:
                diff = cv2.absdiff(
                    self.history_image[0], 
                    self.history_image[1]
                )

                mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                if self.spinBox_Erosion.value() != 0:
                    kernel = np.ones(
                        (
                            self.spinBox_Erosion.value(),  # type: ignore
                            self.spinBox_Erosion.value()  # type: ignore
                        ), np.uint8)
                    mask = cv2.erode(
                        mask,
                        kernel, 
                        iterations=1
                    )
                
                th = 1
                inv_mask = mask > th
                # inv_mask = mask > th  # type: ignore
                # self.buffer_graphicsView_Mask = diff
                self.buffer_graphicsView_Mask = mask
                
                self.buffer_graphicsView_Difference = np.zeros_like(self.history_image[1], np.uint8)
                self.buffer_graphicsView_Difference[
                    inv_mask
                ] = self.history_image[1][
                    inv_mask
                ]
                # print("inv_mask", len(inv_mask[inv_mask != 0]))

                if self.spinBox_Log_Resolution.value() != 0:  # type: ignore
                    cache_history_pixel = len(inv_mask[inv_mask != 0]) / self.spinBox_Log_Resolution.value() if len(  # type: ignore
                        inv_mask[inv_mask != 0]) != 0 else 0
                else:
                    cache_history_pixel = len(inv_mask[inv_mask != 0])
                self.history_pixel["pixel"].append(
                    cache_history_pixel
                ) 
                return self.buffer_graphicsView_Difference
        return None

    def plot_history_graph(self):
        if len(self.history_pixel["pixel"]) > 0:
            self.history_pixel["time"].append(
                self.time_counter
            )
            self.time_counter += 1
            self.widget_Historical_Graph_CH1.clear()  # type: ignore
            self.widget_Historical_Graph_CH1.plot(  # type: ignore
                list(self.history_pixel["time"]),
                list(self.history_pixel["pixel"]),
                # symbol='*'
                pen=self.graph_pen,
                symbolSize=7,
            )

### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###

if __name__ == "__main__":
    # title, Class_UI, run=True, UI_File_Path= "test.ui", qss_File_Path = ""
    stdo(1, "Running {}...".format(__name__))
    app, ui = init_and_run_UI(
        "Process Test",
        Ui_Image_Difference,
        UI_File_Path="image_difference.ui"
    )
