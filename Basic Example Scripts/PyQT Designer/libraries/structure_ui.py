import sys

from PyQt6 import uic, QtWidgets
from PyQt6.QtGui import QStandardItemModel, QPen, QBrush, QColor
from PyQt6.QtCore import QTimer, QPoint
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem

import libs
from tools import list_files


### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### UI OBJECT CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###
class ROI_Rect(QGraphicsRectItem):
    
    def __init__(self, scene=None, x=0, y=0, w=10, h=10, init=True, keep_in_scene=True, create_holders=True, hold_point_offset = 15, color=None, setAlphaF=0.25, setAlphaF_holders=0.25, pen_width=0):

        self.keep_in_scene = keep_in_scene
        self.history = list()
        
        self.creation_offset_x = x
        self.creation_offset_y = y

        self.mid_roi = None
        """
        self.hold_up = None
        self.hold_down = None
        self.hold_left = None
        self.hold_right = None
        """
        self.hold_point_offset = hold_point_offset

        super().__init__(x, y, w, h)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        if init:
            self.init(scene=scene, create_holders=create_holders, color=color, setAlphaF=setAlphaF, setAlphaF_holders=setAlphaF_holders, pen_width=pen_width)

    def init(self, scene=None, create_holders=False, color=None, setAlphaF=0.25, setAlphaF_holders=0.1, pen_width=0):
        if scene is not None:
            scene.addItem(self)
            self.scenerect = self.scene().sceneRect()
        
        if self.parentItem() is None and create_holders:
            self.holding_parts(color="white", setAlphaF=setAlphaF_holders)
        
        if color is not None:
            self.color(color=color, setAlphaF=setAlphaF, pen_width=pen_width)

    def resize_rect(self, rect):
        rect = self.itemChange(QGraphicsItem.ItemScaleChange, rect)
        self.setRect(rect)
        
    def resize_pos(self, lock_dimesion, pos):
        start_x, start_y, _, _, width, height = self.info()
        rect = None

        # print("resize_pos", start_x, start_y, "| width, height", width, height)
        if lock_dimesion == "up":
            rect = QRectF(
                start_x,
                start_y + pos.y(),
                width,
                height - pos.y()
            )
            """
            hold_up_rect = self.hold_up.rect()
            hold_up_rect
            self.hold_up.setRect()
            """

        elif lock_dimesion == "down":
            rect = QRectF(
                start_x,
                start_y,
                width,
                height + pos.y()
            )
            
        elif lock_dimesion == "left":
            rect = QRectF(
                start_x + pos.x(),
                start_y,
                width - pos.x(),
                height
            )
            
        elif lock_dimesion == "right":
            rect = QRectF(
                start_x,
                start_y,
                width + pos.x(),
                height
            )
            
        # print("lock_dimesion", lock_dimesion, rect)
        if rect is not None:
            rect = self.itemChange(QGraphicsItem.ItemScaleChange, rect)
            # rect = self.itemChange(lock_dimesion, rect)
            self.setRect(rect)
            self.scene().update()
            

    def holding_parts(self, color="red", setAlphaF=0.15):
        """
        self.hold_up = ROI_Rect_Hold()
        self.hold_down = ROI_Rect_Hold()
        self.hold_left = ROI_Rect_Hold()
        self.hold_right = ROI_Rect_Hold()
        
        self.hold_up.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_down.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_left.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_right.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        
        self.hold_up.setParentItem(self)
        self.hold_down.setParentItem(self)
        self.hold_left.setParentItem(self)
        self.hold_right.setParentItem(self)
        """
        self.mid_roi = ROI_Rect_Hold()

        self.mid_roi.color(color=color, setAlphaF=setAlphaF, pen_width=1)
        # self.mid_roi.opaqueArea(10)

        self.mid_roi.setParentItem(self)
        
        self.resize_holding_points()

    def move_holding_points(self):
        start_x = self.boundingRect().topLeft().x()
        start_y = self.boundingRect().topLeft().y()
        
        
        if self.mid_roi is not None:
            self.mid_roi.setPos(
                QPointF(
                    start_x - self.creation_offset_x,
                    start_y - self.creation_offset_y,
                )
            )

    def resize_holding_points(self):
        start_x, start_y, _, _, width, height = self.info()
        
        if self.mid_roi is not None:
            self.mid_roi.setRect(
                QRectF(
                    start_x + self.hold_point_offset,
                    start_y + self.hold_point_offset,
                    width - 2 * self.hold_point_offset,
                    height - 2 * self.hold_point_offset
                )
            )
        """
        if self.hold_up is not None:
            self.hold_up.setRect(
                QRectF(
                    start_x,
                    start_y,
                    width,
                    self.hold_point_offset,
                )
            )
            
        if self.hold_down is not None:
            self.hold_down.setRect(
                QRectF(
                    start_x,
                    end_y - self.hold_point_offset,
                    width,
                    self.hold_point_offset,
                )
            )
            
        if self.hold_left is not None:
            self.hold_left.setRect(
                QRectF(
                    start_x,
                    start_y,
                    self.hold_point_offset,
                    height,
                )
            )
            
        if self.hold_right is not None:
            self.hold_right.setRect(
                QRectF(
                    end_x - self.hold_point_offset,
                    start_y,
                    self.hold_point_offset,
                    height,
                )
            )
        """

    def itemChange(self, change, value):
        # print("itemChange in  | change:", change, "value:", value)
        
        if change == QGraphicsItem.ItemPositionChange and self.keep_in_scene:
            # print("QGraphicsItem.ItemPositionChange")
            # (WRONG) https://stackoverflow.com/questions/47216468/restraining-a-qgraphicsitem-using-itemchange
            # (WRONG) https://doc.qt.io/archives/qt-4.8/qgraphicsitem.html#itemChange
            # https://stackoverflow.com/questions/34379511/restrict-item-in-pyqt4-using-itemchange
            # value is the new position.
            left_top_value = QPointF(
                value.x() + self.creation_offset_x,
                value.y() + self.creation_offset_y
            )
            right_bottom_value = QPointF(
                value.x() + self.creation_offset_x + self.boundingRect().right(),
                value.y() + self.creation_offset_y + self.boundingRect().bottom()
            )

            if not ( self.scenerect.contains(left_top_value) and self.scenerect.contains(right_bottom_value) ):
                # Keep the item inside the scene rect.
                # print("value_old:", value)
                value.setX(
                    min(
                        self.scenerect.right() - self.boundingRect().right(),
                        max(
                            value.x(),
                            self.scenerect.left() - self.creation_offset_x,
                        )
                    )
                )
                value.setY(
                    min(
                        self.scenerect.bottom() - self.boundingRect().bottom(),
                        max(
                            value.y(),
                            self.scenerect.top() - self.creation_offset_y,
                        )
                    )
                )
            self.move_holding_points()
                
        elif change == QGraphicsItem.ItemScaleChange and self.keep_in_scene:
            # print("QGraphicsItem.ItemScaleChange")
            # print("value_old", value)
            # https://stackoverflow.com/questions/34379511/restrict-item-in-pyqt4-using-itemchange
            # value is the new position.
            left_top_value = QPointF(
                value.x(),
                value.y()
            )
            
            right_bottom_value = QPointF(
                value.x() + value.width(),
                value.y() + value.height()
            )
            
            # if not ( self.scenerect.contains(left_top_value) and self.scenerect.contains(right_bottom_value) ):
            """
            print("self.scenerect:", self.scenerect)
            print("left_top_value:", left_top_value)
            print("right_bottom_value:", right_bottom_value)
            print("self.scenerect.contains(left_top_value):", self.scenerect.contains(left_top_value))
            print("self.scenerect.contains(right_bottom_value):", self.scenerect.contains(right_bottom_value))
            """
            
            # Keep the item inside the scene rect.
            value.setX(
                min(
                    self.scenerect.right(),
                    max(
                        value.x(),
                        self.scenerect.left()
                    )
                )
            )
            value.setY(
                min(
                    self.scenerect.bottom(),
                    max(
                        value.y(),
                        self.scenerect.top()
                    )
                )
            )
            
            
            value.setHeight(
                min(
                    self.scenerect.bottom() - value.y(),
                    max(
                        value.height(),
                        self.scenerect.top()  # - value.y() - hold_point_offset
                    )
                )
            )
            value.setWidth(
                min(
                    self.scenerect.right() - value.x(),
                    max(
                        value.width(),
                        self.scenerect.left()  # - value.x() - hold_point_offset
                    )
                )
            )
            
            self.resize_holding_points()
            # print("value_new", value)

        # print("itemChange out | change:", change, "value:", value)
        self.history.append([change, value])
        if self.scene() is not None:
            self.scene().update()
        # return QGraphicsItem.itemChange(self, change, value)
        return super().itemChange(change, value) # Call super

    def info(self):
        start_pos = self.mapToScene(self.boundingRect().topLeft())
        start_x, start_y = start_pos.x(), start_pos.y()
        
        end_pos = self.mapToScene(self.boundingRect().bottomRight())
        end_x, end_y = end_pos.x(), end_pos.y()
        
        width, height = self.rect().width(), self.rect().height()

        return start_x, start_y, end_x, end_y, width, height

    def color(self, color=None, setAlphaF=1, pen_width=0):
        if type(color) is str:
            self.current_color = color
            self.current_setAlphaF = setAlphaF
            self.current_pen_width = pen_width
            
            
            qcolor = QColor(self.current_color)
            qcolor.setAlphaF(self.current_setAlphaF)
            
            self.setBrush(QBrush(qcolor))
            self.setPen(QPen(QColor(self.current_color), pen_width, Qt.SolidLine))
            
            if self.scene() is not None:
                self.scene().update()
        else:
            return self.brush, self.pen

    def move(self, value_pos):
        self.setPos(value_pos)
        
    def move_to_mouse(self, mouseMove_pos, position_offset=QPoint(0, 0)):
        # mouseMove_pos = self.mapToScene(mouse_pos)
        # print("mouseMove_pos:", mouseMove_pos)
        
        move_x = mouseMove_pos.x() - self.creation_offset_x - position_offset.x()
        move_y = mouseMove_pos.y() - self.creation_offset_y - position_offset.y()
        
        # print("move_x, move_y", move_x, move_y)
        self.move(QPoint(move_x, move_y))
    
    def mouseReleaseEvent(self, event):
        self.mouseRelease_pos = event.pos()
        self.history.append([self.MouseReleaseEvent, event])
        self.mouseRelease_pos_scene = self.graphicsView_msb_test.mapToScene(event.pos())
    
    """
    def mouseRelease_control(self, set_bool=None):
        if set_bool is not None:
            self.is_mouseRelease = set_bool
        return self.is_mouseRelease
    """
    
    def mouseDoubleClickEvent(self, event):
        self.doubleClick_control(True)
        self.doubleClick_pos = event.pos()

    def doubleClick_control(self, set_bool=None):
        if set_bool is not None:
            self.is_doubleClick = set_bool
        return self.is_doubleClick

    def grabMouse(self):
        self.is_grabMouse = True
        print("Item Grabbed.")
        self.grabMouse_graphics_pos_scene = self.mouseMove_graphics_pos_scene
        while self.is_grabMouse:
            self.qt_priority()
            self.grabMouse_graphics_pos_scene = self.mouseMove_graphics_pos_scene
    
    def doubleClick_control(self, set_bool=None):
        if set_bool is not None:
            self.is_doubleClick = set_bool
        return self.is_doubleClick
    
    def ungrabMouse(self):
        print("Item UN-Grabbed.")
        self.is_grabMouse = False
    
    def qt_priority(self):
        # https://stackoverflow.com/questions/8766584/displayin-an-image-in-a-qgraphicsscene
        QtCore.QCoreApplication.processEvents()
                
    def destroy(self):
        self.scene().removeItem(self)

class ROI_Rect_Hold(QGraphicsRectItem):
    def __init__(self, scene=None, x=0, y=0, w=10, h=10, color=None, setAlphaF=0.25, pen_width=0):
        self.creation_offset_x = x
        self.creation_offset_y = y

        super().__init__(x, y, w, h)

        self.setFlag(QGraphicsItem.ItemStacksBehindParent, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsFocusable, False)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
        self.setAcceptHoverEvents(False)

        self.init(scene=scene, color=color, setAlphaF=setAlphaF, pen_width=pen_width)

    def init(self, scene=None, color=None, setAlphaF=0.25, pen_width=0):
        if scene is not None:
            scene.addItem(self)
            self.scenerect = self.scene().sceneRect()
            
        if self.scene() is not None:
            self.scenerect = self.scene().sceneRect()
            
        if color is not None:
            self.color(color=color, setAlphaF=setAlphaF, pen_width=pen_width)

    def info(self):
        start_pos = self.mapToScene(self.boundingRect().topLeft())
        start_x, start_y = start_pos.x(), start_pos.y()
        
        end_pos = self.mapToScene(self.boundingRect().bottomRight())
        end_x, end_y = end_pos.x(), end_pos.y()
        
        width, height = self.rect().width(), self.rect().height()

        return start_x, start_y, end_x, end_y, width, height

    def color(self, color=None, setAlphaF=1, pen_width=0):
        if color is not None:
            self.current_color = color
            self.current_setAlphaF = setAlphaF
            self.current_pen_width = pen_width
            
            qcolor = QColor(self.current_color)
            qcolor.setAlphaF(self.current_setAlphaF)
            
            self.setBrush(QBrush(qcolor))
            self.setPen(QPen(QColor(self.current_color), pen_width, Qt.SolidLine))
            
            if self.scene() is not None:
                self.scene().update()
        else:
            return self.brush, self.pen


### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### UI WINDOW CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###
class Ui_ImagePopup_Modified(QtWidgets.QMainWindow): #, Ui_ImagePopup):
    def __init__(self, *args, obj=None, **kwargs):
        super(Ui_ImagePopup_Modified, self).__init__(*args, **kwargs)
        # self.setupUi(self)
        uic.loadUi('image_popup.ui', self)  # Load the .ui file
        self.last_image = None
        
    def show_image(self, image=None, fixed_to_image_size=True, keep_aspect_ratio=True):
        if image is not None:
            self.set_image(image)
            
            if fixed_to_image_size:
                window_size = image.size()
                
                self.setMinimumSize(window_size)
                
                # rect (window_size)
                #self.setGeometry()
            
            if keep_aspect_ratio:
                stdo(2, "Keep Aspect Ratio Feature is still at development.")
                
            self.show()
        else:
            stdo(2, "Image parameter is None")
    
    def set_image(self, image):
        self.last_image = image
        self.qt_label_image.setPixmap(image)
    

### ### ### ### ### # ### ### ### ### ###
### ### ### UI CONFIGURATIONS ### ### ###
### ### ### ### ### # ### ### ### ### ###
class Structure_UI(QtWidgets.QMainWindow):
    
    Main_Window = None
    Developer_Settings_PopUp = None
    
    UI_File_Path = ""
    Garbage_Collection = list()

    def __init__(self, UI_File_Path=""):
        super(Structure_UI, self).__init__()

        self.UI_File_Path = UI_File_Path
        self.load_UI(self, self.UI_File_Path)
        # self.configure_Button_Connections()

    @staticmethod
    def load_UI(UI, UI_File_Path):
        # Load the .ui file
        uic.loadUi(UI_File_Path, UI) if UI_File_Path != "" else None

        # self.Main_Window.configure_Other_Settings()
        UI.configure_Button_Connections()
        
    @staticmethod
    def set_Style_Sheet(QObject, style_sheet_file_path):
        QObject.setStyleSheet("") if style_sheet_file_path == "" or style_sheet_file_path == "default.qss" else QObject.setStyleSheet(
            open(style_sheet_file_path, "r").read())
            
    @staticmethod
    def set_Style_Sheet_Globally(style_sheet_file_path):
        Structure_UI.set_Style_Sheet(
            QtWidgets.QApplication.instance(), 
            style_sheet_file_path
        )

    @staticmethod
    def load_themes_to_combobox(comboBox, themes_path, clear_before_update=True):
        themes_subtree = list_files(
            themes_path,
            extensions=[".qss"],
            recursive=True
        )
        
        themes_dict = dict()
        themes_dict["default"] = "" 
        if len(themes_subtree) != 0:
            if type(themes_subtree[0]) is list:
                for themes in themes_subtree:
                    for theme in themes:
                        key = theme.split("/")[-1].split(".")[0]
                        themes_dict[key] = theme
                        
            else:
                for theme in themes_subtree:
                    key = theme.split("/")[-1].split(".")[0]
                    themes_dict[key] = theme
        
        if clear_before_update:
            comboBox.clear()
        for key in themes_dict.keys():
            comboBox.addItem(key)
            
        return themes_dict
        
    @staticmethod
    def listView_init(listView):
        model_Question_Pack_List = QStandardItemModel()
        listView.setModel(
            model_Question_Pack_List
        )
        return model_Question_Pack_List
                   
    @staticmethod
    def qtimer_create_and_run(parent, connection, delay=100, is_needed_start=True, is_single_shot=False):
        timer = QTimer(parent)
        timer.setInterval(delay)
        timer.timeout.connect(connection)
        timer.setSingleShot(is_single_shot)
        if is_needed_start:
            timer.start()
        return timer

    @staticmethod
    def qtimer_stop_dict(qtimer_dict):
        for key, qtimer in qtimer_dict.items():
            qtimer.stop()
            #stdo(1, "{} qtimer is stopped".format(key))

    def configure_Button_Connections(self):
        print("Overwrite configure_Button_Connections Function!")

    def configure_Other_Settings(self):
        print("Overwrite configure_Other_Settings Function!")



### ### ### ### ## ## ## ### ### ### ###
### ### ### CUSTOM FUNCTIONS ### ### ###
### ### ### ### ## ## ## ### ### ### ###

def init_UI(Class_UI, UI_File_Path= "test.ui", is_Maximized=False):
    app = QtWidgets.QApplication(sys.argv)

    ui = Class_UI(UI_File_Path = UI_File_Path)

    # Need to be returned to run namespace, else it won't show window render and stuck
    return app, ui


def run_UI(app, UI, title, show_UI=True, is_Maximized=False):
    UI.setWindowTitle(title)

    if show_UI:
        if is_Maximized:
            UI.showMaximized()  # Show in fully fitted window size#
        else:
            UI.show()  # Show at default window size#
            
    sys.exit(app.exec())
    
def init_and_run_UI(title, Class_UI, run=True, UI_File_Path= "test.ui", show_UI = True, is_Maximized=False):
    app, ui = init_UI(
        Class_UI, 
        UI_File_Path = UI_File_Path, 
    )
    if run:
        run_UI(
            app = app, 
            UI = ui,
            title = title,
            show_UI = show_UI, 
            is_Maximized=is_Maximized
        )

if __name__ == "__main__":
    
    app, ui = init_and_run_UI(
        title = "Test", 
        Class_UI = Structure_UI, 
        UI_File_Path = sys.argv[1] if len(sys.argv) == 2 else "test.ui"
    )
