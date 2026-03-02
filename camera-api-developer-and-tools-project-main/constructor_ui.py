
# ## ### ### ### ### ## ### ### ### ###
# ## ### ### BUILT-IN LIBRARIES ### ###
# ## ### ### ### ### ## ### ### ### ###
import logging
from time import sleep

# ## ### ### ### ## ## ## ### ### ###
# ## ### ### CUSTOM LIBRARIES ### ###
# ## ### ### ### ## ## ## ### ### ###
import libs

from stdo import stdo
from qt_tools import lcdNumber_Set, qtimer_Create_And_Run, hide_elements, show_elements
from structure_ui import Structure_UI, init_and_run_UI, init_UI, Graphics_View
from structure_camera import CAMERA_FLAGS
from structure_ui_camera import Structure_Ui_Camera

# CONFIGURATIONS
MAX_SUBWINDOW = 20


class Graphics_View_LCD(Graphics_View):
    def __init__(self, *args, ** kwargs):
        super(Graphics_View_LCD, self).__init__(*args, **kwargs)

        # ## ### ### ## ## ## ### ###
        # ## INITIALIZE VARIABLES ###
        # ## ### ### ## ## ## ### ###

        self.LCD_Number_List = None
        self.list_coords_double_clicked = list()

        # ## ### ### ## ## ## ### ###
        # ## ### ### ## ## ## ### ###
        # ## ### ### ## ## ## ### ###

    # ## ### ### ### ### ### ### ###
    # ## ###  EVENT HANDLERS ### ###
    # ## ### ### ### ### ### ### ###

    def mouseMoveEvent(self, event):
        # Graphics_View.mouseMoveEvent(self, event)
        super(Graphics_View_LCD, self).mouseMoveEvent(event)

        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsPixmapItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsPixmapItem.pixmap
        # if type(self.mouse_Events["mouseMove_current_item"]) is QGraphicsPixmapItem and self.LCD_Number_List is not None:
        if self.mouse_Events["mouseMove_current_item"] is not None and self.LCD_Number_List is not None:
            qt_color_red, qt_color_green, qt_color_blue = self.get_Color(
                # self.mouse_Events["mouseMove_current_item"].pixmap(),
                self.get_Background_Item().pixmap(),
                self.mouse_Events["mouseMove_position_scene"].x(),
                self.mouse_Events["mouseMove_position_scene"].y(),
                is_QT_Type=True
            )
            lcdNumber_Set(
                self.LCD_Number_List,
                [
                    qt_color_red,
                    qt_color_green,
                    qt_color_blue,
                    int((qt_color_red + qt_color_green + qt_color_blue) / 3)
                    if qt_color_red + qt_color_green + qt_color_blue != 0
                    else 0,
                    int(255 - (qt_color_red + qt_color_green + qt_color_blue) / 3)
                    if qt_color_red + qt_color_green + qt_color_blue != 0
                    else 0
                ]
            )

    # ## ### ### ### ### ### ### ###
    # ## ### ### ### ### ### ### ###
    # ## ### ### ### ### ### ### ###


# ## ### ### ### ### ## ## ## ### ### ### ### ###
# ## ### ### CAMERA UI CONFIGURATIONS ### ### ###
# ## ### ### ### ### ## ## ## ### ### ### ### ###
class Ui_Camera_API_Main(Structure_UI):
    logger_level = logging.INFO

    def __init__(self, *args, obj=None, logger_level=logging.INFO, **kwargs):
        super(Ui_Camera_API_Main, self).__init__(*args, **kwargs)

        # ## ### ### ### ###
        # ## Constructor ###
        # ## ### ### ### ###
        self.logger_level = logger_level

        # ## ### ### ### ###
        # ## ### Init ### ##
        # ## ### ### ### ###
        self.init()

    # ## ### ## ### ###
    # ## OVERWRITES ###
    # ## ### ## ### ###

    def init(self):

        self.configure_Other_Settings()
        self.configure_Button_Connections()
        self.init_QTimers()

    def configure_Other_Settings(self):
        self.action_Settings_Page.triggered.connect(
            self.action_Settings_Page_Triggered
        )
        self.action_Add_Camera.triggered.connect(
            self.action_Add_Camera_Triggered
        )

    # ## ### ## ### ###
    # ## ### ## ### ###
    # ## ### ## ### ###

    # ## ### ### ## ### ###
    # ## MDI SUB WINDOW ###
    # ## ### ### ## ### ###
    """
    def destroy_Sub_Window_Overwrite(self, sub_window):
        self.mdiArea_Camera_Area.removeSubWindow(sub_window)
        self.action_Add_Camera.setEnabled(True) \
            if len(self.mdiArea_Camera_Area.subWindowList()) < MAX_SUBWINDOW \
            else self.action_Add_Camera.setEnabled(False)
    """

    def action_Add_Camera_Triggered(self):
        if len(self.mdiArea_Camera_Area.subWindowList()) < MAX_SUBWINDOW:
            # self.action_Add_Camera.setEnabled(True)
            self.create_Sub_Window(
                parent=self,
                mdiArea=self.mdiArea_Camera_Area,
                UI_Class=Ui_Camera_API_Developer,
                title="Camera",
                UI_File_Path="camera_api_developer_UI.ui"
            ).show()
            self.mdiArea_Camera_Area.tileSubWindows()
        """
        else:
            self.action_Add_Camera.setEnabled(False)
        """

    # ## ### ### ## ### ###
    # ## ### ### ## ### ###
    # ## ### ### ## ### ###

    def action_Settings_Page_Triggered(self):
        app_Ui_Settings_Page, ui_Settings_Page = init_UI(
            Ui_Settings,
            UI_File_Path="Settings_UI.ui"
        )
        ui_Settings_Page.setWindowTitle("Settings")
        ui_Settings_Page.show()
        self.garbage_Collector_Add([app_Ui_Settings_Page, ui_Settings_Page])
        """
        ui_Settings_Page = Structure_UI(
            UI_File_Path="Ui_Settings_Page.ui"
        )
        ui_Settings_Page.show()
        self.garbage_Collector_Add(ui_Settings_Page)
        """


# ## ### ### ### ### ### ## ### ### ### ### ### ###
# ## ### ### SETTINGS UI CONFIGURATIONS ### ### ###
# ## ### ### ### ### ### ## ### ### ### ### ### ###

class Ui_Settings(Structure_UI):

    UI_File_Path = ""
    themes_list = {
        "default": "default.qss"
    }

    def __init__(self, *args, obj=None, **kwargs):
        super(Ui_Settings, self).__init__(*args, **kwargs)
        """
        self.UI_File_Path = UI_File_Path
        self.load_UI(self, self.UI_File_Path)
        # self.configure_Button_Connections()
        """

        self.init()

    # ## ### ## ### ###
    # ## OVERWRITES ###
    # ## ### ## ### ###

    def init(self):
        self.themes_list = self.load_themes_to_combobox(
            self.comboBox_theme_chooser,
            "themes"
        )
        self.configure_Other_Settings()
        self.configure_Button_Connections()

    def configure_Other_Settings(self):
        self.comboBox_theme_chooser.currentIndexChanged.connect(
            self.comboBox_theme_chooser_currentIndexChanged
        )
        self.pushButton_theme_chooser_refresher.clicked.connect(
            lambda: self.pushButton_theme_chooser_refresher_Clicked(
                self.comboBox_theme_chooser
            )
        )

    # ## ### ## ### ###
    # ## ### ## ### ###
    # ## ### ## ### ###

    # ## ### ### ### ###
    # # Theme Chooser ##
    # ## ### ### ### ###

    def comboBox_theme_chooser_currentIndexChanged(self):
        self.set_Style_Sheet_Globally(
            self.themes_list[self.comboBox_theme_chooser.currentText(
            )] if self.comboBox_theme_chooser.currentText() in self.themes_list else ""
        )

    def pushButton_theme_chooser_refresher_Clicked(self, comboBox):
        self.themes_list = self.load_themes_to_combobox(
            comboBox,
            "themes",
            True
        )

    # ## ### ## ### ###
    # ## ### ## ### ###
    # ## ### ## ### ###

# ## ### ### ### ### ### ## ### ### ### ### ### ###
# ## ### ### ### ### ### ## ### ### ### ### ### ###
# ## ### ### ### ### ### ## ### ### ### ### ### ###

# ## ### ### ### ### ## ## ## ### ### ### ### ###
# ## ### ### CAMERA UI CONFIGURATIONS ### ### ###
# ## ### ### ### ### ## ## ## ### ### ### ### ###


class Ui_Camera_API_Developer(Structure_Ui_Camera):
    logger_level = logging.INFO
    # __Threads = dict()

    mdiArea = None

    def __init__(self, *args, obj=None, logger_level=logging.INFO, **kwargs):
        super(Ui_Camera_API_Developer, self).__init__(*args, **kwargs)

        # ## ### ### ### ###
        # ## Constructor ###
        # ## ### ### ### ###
        self.logger_level = logger_level

        # ## ### ### ### ###
        # ## ### Init ### ##
        # ## ### ### ### ###
        self.init()

    # ## ### ## ### ###
    # ## OVERWRITES ###
    # ## ### ## ### ###

    def init(self):

        self.graphicsView_Page_1_Camera.LCD_Number_List = [
            self.lcdNumber_Pointer_Color_Red,
            self.lcdNumber_Pointer_Color_Green,
            self.lcdNumber_Pointer_Color_Blue,
            self.lcdNumber_Pointer_Color_Grayscale,
            self.lcdNumber_Pointer_Color_Grayscale_Inverted
        ]
        self.configure_Button_Connections()
        self.configure_Other_Settings()
        self.init_QTimers()

        self.comboBox_Camera_API_Selection.addItems(
            [flag.name for flag in CAMERA_FLAGS]
        )
        self.is_Video_Saving_Stopped = True
        # self.init_Buffers()
        # self.init_Threads()
        # self.connect_Threads()

    """
    def init_Buffers(self, *args, **kwargs):
        super(Ui_Camera_API_Developer, self).init_Buffers(*args, **kwargs)

        self.Buffer_Dict["graphicsView_Page_1_Camera"] = Structure_Buffer(
            max_limit=240
        )
    """

    def init_QTimers(self, *args, **kwargs):
        super(Ui_Camera_API_Developer, self).init_QTimers(*args, **kwargs)

        """
        self.QTimer_Dict["graphicsView_Page_1_Camera_Buffer_Connector"] = qtimer_Create_And_Run(
            self,
            lambda: self.Buffer_Dict["graphicsView_Page_1_Camera"].append(self.camera_Instance.stream_Returner())
            ),
            100
        )
        """
        self.QTimer_Dict["graphicsView_Page_1_Camera_Renderer"] = qtimer_Create_And_Run(
            self,
            lambda: self.graphicsView_Renderer(
                self.graphicsView_Page_1_Camera,
                None if self.camera_Instance is None else self.camera_Instance.stream_Returner(auto_pop=False)
            ) if self.is_Stream_Active() else None,
            10
        )

    def configure_Button_Connections(self):
        self.pushButton_Page_1_Connect_to_Camera.clicked.connect(
            lambda: self.connect_to_Camera(
                [
                    flag for
                    flag in CAMERA_FLAGS if flag.name == self.comboBox_Camera_API_Selection.currentText()
                ][0],
                self.spinBox_Camera_Index.value(),
                self.spinBox_Buffer_Size.value(),
                self.exposure_Time,
                True,
                30
            )
        )
        self.pushButton_Page_1_Remove_the_Camera.clicked.connect(
            self.camera_Remove
        )
        self.pushButton_Page_1_Stream_Switch.clicked.connect(
            # lambda: print("self.stream_Switch", self.stream_Switch())
            lambda: self.stream_Switch()
        )
        self.pushButton_Buffer_Clear.clicked.connect(
            lambda: None if self.camera_Instance is None \
                else self.camera_Instance.buffer_Clear()
        )
        self.pushButton_Set_Exposure.clicked.connect(
            lambda: self.set_Camera_Exposure(
                self.spinBox_Exposure_Time.value()
            )
        )
        self.pushButton_Page_1_Video.clicked.connect(
            self.video_Saving_Process
        )
        # self.pushButton_Page_1_Snapshot.clicked.connect()

    def configure_Other_Settings(self):
        # Event Position Initializes
        self.mouse_Positions["mouseMove_graphicsView_Pos"] = None
        self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"] = None

        # self.init_qt_graphicsView(
        #    self.graphicsView_Page_1_Camera
        # )

        self.init_qt_graphicsView_Scene(
            self.graphicsView_Page_1_Camera,
        )
        self.spinBox_Exposure_Time.valueChanged.connect(
            lambda: self.dial_Exposure_Time.setValue(
                int(
                    float(
                        self.spinBox_Exposure_Time.value()
                    ) * 10000
                )
            )
        )

        self.dial_Exposure_Time.valueChanged.connect(
            lambda: self.spinBox_Exposure_Time.setValue(
                int(
                    self.dial_Exposure_Time.value() / 10000
                )
            )
        )

        # ## ### ## ###
        # ## BUFFER ###
        # ## ### ## ###
        self.horizontalSlider_Page_1_Buffer_Step_Bar.setMinimum(0)
        self.spinBox_Buffer_Size.valueChanged.connect(
            lambda:(
                self.horizontalSlider_Page_1_Buffer_Step_Bar.setMaximum(
                    self.spinBox_Buffer_Size.value()
                ),
                None if self.camera_Instance is None \
                else self.camera_Instance.set_Buffer_Size(
                    self.spinBox_Buffer_Size.value()
                )
            )
        )

        self.horizontalSlider_Page_1_Buffer_Step_Bar.valueChanged.connect(
            lambda: self.graphicsView_Renderer(
                self.graphicsView_Page_1_Camera,
                None if self.camera_Instance is None else self.camera_Instance.get_Buffered_Image(
                    self.horizontalSlider_Page_1_Buffer_Step_Bar.value()
                )
            )
        )

        self.checkBox_Show_Only_Stream.stateChanged.connect(
            lambda:
            hide_elements([
                self.groupBox_Page_1_Camera_General,
                self.widget_Real_Time_Properties,
                self.widget_Page_1_Tools
            ]) \
                if self.checkBox_Show_Only_Stream.isChecked() else \
            show_elements([
                self.groupBox_Page_1_Camera_General,
                self.widget_Real_Time_Properties,
                self.widget_Page_1_Tools
            ])
        )
        hide_elements([self.dial_Exposure_Time])

    def closeEvent(self, *args, **kwargs):
        super(Ui_Camera_API_Developer, self).closeEvent(*args, **kwargs)

        self.camera_Remove()
        if self.mdiArea is not None and self.Parent is not None:
            # self.Parent.destroy_Sub_Window_Overwrite(self)
            self.Parent.destroy_Sub_Window(self.mdiArea, self)

    # ## ### ## ### ###
    # ## ### ## ### ###
    # ## ### ## ### ###

    # ## ### ### ### ###
    # ## THREAD APIs ###
    # ## ### ### ### ###

        """
    def init_Threads(self):
        return
        logger_level = logging.INFO
        self.__Threads["camera_Listener"] = Thread_Object(
            name="camera_Listener",
            delay=0.1,
            logger_level=logger_level,
            set_Deamon=True,
            run_number=1,
            quit_trigger=self.is_Quit_App
        )
        """
        """
        # ## ### ### ### ### ### ### #
        #  Camera Streamer Thread   #
        # ## ### ### ### ### ### ### #
        self.__Threads["camera_Listener"] = Thread_Object(
            name="camera_Listener",
            delay=0.1,
            logger_level=logger_level,
            set_Deamon=True,
            run_number=1,
            quit_trigger=self.is_Quit_App
        )
        # self.__Threads["camera_Listener"].logger.disabled = True  # is_logger_disabled
        # self.__Threads["camera_Listener"].logger.propagate = not is_logger_disabled
        """

    # ## ### ### ### ###
    # ## ### ### ### ###
    # ## ### ### ### ###

    # ## ### ### ### ### ### ### ###
    # ## ###  EVENT HANDLERS ### ###
    # ## ### ### ### ### ### ### ###

    def mouseMove_Event_Handler_graphicsView(self, event):
        self.mouse_Positions["mouseMove_graphicsView_Pos"] = event.pos()
        self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"] = self.graphicsView_Page_1_Camera.mapToScene(
            self.mouse_Positions["mouseMove_graphicsView_Pos"]
        )
        self.lcdNumber_Set(
            [
                self.lcdNumber_Pointer_X,
                self.lcdNumber_Pointer_Y
            ],
            [
                self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"].x(),
                self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"].y()
            ]
        )

        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsPixmapItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsPixmapItem.pixmap
        background_item = self.get_Background_Item(
            self.graphicsView_Page_1_Camera
        )
        if background_item is not None:
            qt_color_red, qt_color_green, qt_color_blue = self.get_Color(
                background_item.pixmap(),
                self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"].x(),
                self.mouse_Positions["mouseMove_graphicsView_Pos_To_Scene"].y(),
                is_QT_Type=True
            )
            self.lcdNumber_Set(
                [
                    self.lcdNumber_Pointer_Color_Red,
                    self.lcdNumber_Pointer_Color_Green,
                    self.lcdNumber_Pointer_Color_Blue,
                    self.lcdNumber_Pointer_Color_Grayscale,
                    self.lcdNumber_Pointer_Color_Grayscale_Inverted
                ],
                [
                    qt_color_red,
                    qt_color_green,
                    qt_color_blue,
                    int((qt_color_red + qt_color_green + qt_color_blue) / 3)
                    if qt_color_red + qt_color_green + qt_color_blue != 0
                    else 0,
                    int(255 - (qt_color_red + qt_color_green + qt_color_blue) / 3)
                    if qt_color_red + qt_color_green + qt_color_blue != 0
                    else 0
                ]
            )

    # ## ### ### ### ### ### ### ###
    # ## ### ### ### ### ### ### ###
    # ## ### ### ### ### ### ### ###

    def video_Saving_Process(self):
        if self.is_Video_Save_Stop():
            self.switch_Video_Save_Stop(False)
            if self.camera_Instance is not None:
                self.camera_Instance.save_Video_From_Buffer_Thread(
                    path="",
                    name=f"camera_{self.spinBox_Save_Video_FPS.value()}fps",
                    extension="avi",
                    fps=self.spinBox_Save_Video_FPS.value(),
                    trigger_pause=self.is_Stream_Active,
                    trigger_quit=self.is_Video_Save_Stop,
                    number_of_snapshot=-1,
                    delay=0.001
                )
        else:
            self.switch_Video_Save_Stop(True)

    def switch_Video_Save_Stop(self, bool=None):
        self.is_Video_Saving_Stopped = bool if bool is not None else not self.is_Video_Saving_Stopped
        return self.is_Video_Saving_Stopped

    def is_Video_Save_Stop(self):
        return self.is_Quit_App() if self.is_Quit_App() else self.is_Video_Saving_Stopped


# ## ### ### ### ### ## ## ## ### ### ### ### ###
# ## ### ### ### ### ## ## ## ### ### ### ### ###
# ## ### ### ### ### ## ## ## ### ### ### ### ###

if __name__ == "__main__":
    # title, Class_UI, run=True, UI_File_Path= "test.ui", qss_File_Path = ""
    stdo(1, "Running {}...".format(__name__))
    """app, ui = init_and_run_UI(
        "Camera Developer UI",
        Ui_Camera_API_Main,
        UI_File_Path="camera_api_developer_MDI_UI.ui"
    )"""
    app, ui = init_and_run_UI(
        "Camera Developer UI",
        Ui_Camera_API_Developer,
        UI_File_Path="camera_api_developer_UI.ui"
    )
