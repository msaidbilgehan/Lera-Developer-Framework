# REFs;
# - https://stackoverflow.com/questions/60782496/where-do-i-write-the-class-for-a-single-promoted-qwidget-from-qt-designer

import libs
from structure_ui import init_and_run_UI, Graphics_View
from structure_ui_camera import Structure_Ui_Camera

from qt_tools import get_Color, lcdNumber_Set
#from tools import list_files
# , QGraphicsPixmapItem, QMdiSubWindow


### ### ### ### ### # ### ### ### ### ###
### ### ### UI CONFIGURATIONS ### ### ###
### ### ### ### ### # ### ### ### ### ###
class Ui_Modified(Structure_Ui_Camera):
    
    Main_Window = None
    Developer_Settings_PopUp = None
    
    UI_File_Path = ""
    themes_list = {
        "default": "default.qss"
    }

    def __init__(self, *args, obj=None, **kwargs):
        # Call super for built-in initialize
        super(Ui_Modified, self).__init__(*args, **kwargs)

        # Go for other setups
        # self.graphicsView_Main = Graphics_View()
        self.graphicsView_Main.mouseMoveEvent = self.Graphics_View_MouseMove_Event_Handler_LCDNumber

        #self.pushButton_Draw_ROI.clicked.connect()
        self.pushButton_Load_Image.clicked.connect(
            lambda: self.graphicsView_Renderer(
                self.graphicsView_Main,
                self.load_Image_Action(
                    path=self.QFileDialog_Event(
                        "getOpenFileName",
                        [
                            "Open file",  # Title
                            "",  # Path
                            # Filter Information
                            "Image files (*.png *.jpg *.jpeg)"
                        ]
                    )[0]  # Get the file Path
                )
            )
        )

    def Graphics_View_MouseMove_Event_Handler_LCDNumber(self, event):
        Graphics_View.mouseMoveEvent(self.graphicsView_Main, event)
        # super(type(self.graphicsView_Main), self.graphicsView_Main).mouseMoveEvent(event)
        # super(Graphics_View, self).mouseMoveEvent(event)
        # super(Graphics_View, self.graphicsView_Main).mouseMoveEvent(event)
        # super(Graphics_View).mouseMoveEvent(event)  # Not working
        
        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsPixmapItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QGraphicsPixmapItem.pixmap
        """
        background_item = self.get_Background_Item(
            self.graphicsView_Main
        )
        """
        if self.graphicsView_Main.mouse_Events["mouseMove_current_item"] is not None: 
            qt_color_red, qt_color_green, qt_color_blue = get_Color(
                self.graphicsView_Main.mouse_Events["mouseMove_current_item"].pixmap(),
                self.graphicsView_Main.mouse_Events["mouseMove_position_scene"].x(),
                self.graphicsView_Main.mouse_Events["mouseMove_position_scene"].y(),
                is_QT_Type=True
            )
            lcdNumber_Set(
                [
                    self.lcdNumber_Pointer_X,
                    self.lcdNumber_Pointer_Y,
                    self.lcdNumber_Pointer_Color_Red,
                    self.lcdNumber_Pointer_Color_Green,
                    self.lcdNumber_Pointer_Color_Blue,
                    self.lcdNumber_Pointer_Color_Grayscale,
                    self.lcdNumber_Pointer_Color_Grayscale_Inverted
                ],
                [
                    self.graphicsView_Main.mouse_Events["mouseMove_position_scene"]
                        .x(),
                    self.graphicsView_Main.mouse_Events["mouseMove_position_scene"]
                        .y(),
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
    
    def load_Image(self):
        pass


if __name__ == "__main__":
    # import sys
    
    # title, Class_UI, run=True, UI_File_Path= "test.ui", qss_File_Path = ""
    app, ui = init_and_run_UI(
        "QT Graphics View",
        Ui_Modified,
        #UI_File_Path="test.ui" if len(sys.argv) == 1 else sys.argv[1]
        UI_File_Path="main.ui"
    )
