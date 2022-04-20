# References;
#   - Variable Existence Control / "Is variable created?" Control
#       - https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists
#   - MDI SubWindow Examples:
#       - https://stackoverflow.com/questions/62291103/how-do-i-dock-a-subwindow-in-an-mdi-area-in-PyQt5-using-qtdesigner
#       - https://codeloop.org/PyQt5-make-multi-document-interface-mdi-application/
#   - Mouse Events
#       - https://doc.qt.io/qtforpython-5/PySide2/QtGui/QMouseEvent.html#PySide2.QtGui.PySide2.QtGui.QMouseEvent.w

import logging
from time import sleep
import sys
from enum import Enum

### ### ### ### ## ### ###
### EXTERNAL LIBRARIES ###
### ### ### ### ## ### ###
from PyQt5 import uic #, QtWidgets
from PyQt5.QtGui import QStandardItemModel, QPen, QBrush, QColor, QCloseEvent, QTransform  # , QImage, QPixmap,

# QPixmap Conversation is giving QImage parameter type error
# from PyQt5.QtGui import QPixmap
# , QObject, QRect, QSize
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QCoreApplication
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QMainWindow, QApplication, QGraphicsScene, QGraphicsView, QMessageBox, QFileDialog #, QGraphicsPixmapItem, QMdiSubWindow

### ### ### ### ## ### ###
### ### ### ### ## ### ###
### ### ### ### ## ### ###

import libs
from tools import list_files #, time_log
from math_tools import coordinate_Scaling
from stdo import stdo
from structure_data import Structure_Buffer
from qt_tools import QT_Scene_Set_Item, QT_Scene_Add_Item, numpy_To_QT_Type_Converter, qtimer_Create_And_Run, qtimer_All_Stop, lcdNumber_Set, get_Color, QT_Scene_Set_Item_Background, QT_Scene_Set_Item_To_Index, QT_Scene_Add_Item_To_Index, QT_Scene_Add_Item_To_Foreground

class QT_MESSAGEBOX_FLAGS(Enum):
    INFORMATION = 0
    WARNING = 1
    QUESTION = 2
    CRITICAL = 3
    ABOUT = 4
    ABOUT_QT = 5
    WARNING_OK = 6

### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### UI OBJECT CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###


class Graphics_View(QGraphicsView):
    def __init__(self, *args, parent=None, is_Fixed=True, obj=None, **kwargs):
        super(Graphics_View, self).__init__(*args, **kwargs)

        self.is_focus_to_image = False
        self.is_drawing_now = False
        self.current_Frame = None
        self.qtimer_render = None
        self.background_image = None
        self.background_image_qt = None
        self.last_drawn_item = None
        self.scene_items = list() 
        
        ######################
        # Set Configurations #
        ######################
        
        self.setScene(QGraphicsScene(parent=parent))
        
        ######################
        ######################
        ######################
        
        ####################
        # Set Event Buffer #
        ####################
        
        self.mouse_Events = dict() 
          
        # Mouse Move
        self.mouse_Events["mouseMove"] = False
        self.mouse_Events["mouseMove_position"] = None
        self.mouse_Events["mouseMove_position_scene"] = None
        self.mouse_Events["mouseMove_current_item"] = None

        # Mouse Release
        self.mouse_Events["mouseRelease"] = False
        self.mouse_Events["mouseRelease_button"] = None
        self.mouse_Events["mouseRelease_position"] = None
        self.mouse_Events["mouseRelease_position_scene"] = None
        self.mouse_Events["mouseRelease_current_item"] = None
    
        # Mouse Click
        self.mouse_Events["mouseClick"] = False
        self.mouse_Events["mouseClick_button"] = None
        self.mouse_Events["mouseClick_position"] = None
        self.mouse_Events["mouseClick_position_scene"] = None
        self.mouse_Events["mouseClick_current_item"] = None
        
        # Mouse Double Click
        self.mouse_Events["mouseDoubleClick"] = False
        self.mouse_Events["mouseDoubleClick_button"] = None
        self.mouse_Events["mouseDoubleClick_position"] = None
        self.mouse_Events["mouseDoubleClick_position_scene"] = None
        self.mouse_Events["mouseDoubleClick_current_item"] = None
        
    ####################
    ####################
    ####################
        
    def set_Scene(self, parent=None, scene=None):
        self.setScene(QGraphicsScene(parent=parent) if scene is None else scene)

    def is_mouseEvent(self, event, bool=None):
        if bool is not None:
            self.mouse_Events[event] = bool 
        return self.mouse_Events[event]
    
    def is_Drawing_Now(self, bool=None):
        if bool is not None:
            self.is_drawing_now = bool 
        return self.is_drawing_now
    
    def mouseMoveEvent(self, event):

        # CURRENT POSITION
        self.mouse_Events["mouseMove_position"] = event.pos()
        self.mouse_Events["mouseMove"] = True
        # print("mouseMove", True)

        self.mouse_Events["mouseMove_position_scene"] = self.mapToScene(
            self.mouse_Events["mouseMove_position"]
        )

        # CURRENT ITEM
        self.mouse_Events["mouseMove_current_item"] = self.scene().itemAt(
            self.mouse_Events["mouseMove_position_scene"], QTransform()
        ) if self.mouse_Events["mouseMove_position_scene"] is not None else None

        # if self.is_active_focus_to_image():
        #     self.focus_to_image()
            
        self.connector_mouseMoveEvent()

    def connector_mouseMoveEvent(self):
        pass

    def mouseReleaseEvent(self, event):
        # CURRENT BUTTON
        self.mouse_Events["mouseRelease_button"] = event.buttons()
        self.mouse_Events["mouseRelease"] = True
        # print("mouseRelease", True)
        
        # CURRENT POSITION
        self.mouse_Events["mouseRelease_position"] = self.mouse_Events["mouseMove_position"]
        self.mouse_Events["mouseRelease_position_scene"] = self.mouse_Events["mouseMove_position_scene"]

        # CURRENT ITEM
        self.mouse_Events["mouseRelease_current_item"] = self.scene().itemAt(
            self.mouse_Events["mouseRelease_position_scene"], QTransform()
        ) if self.mouse_Events["mouseRelease_position_scene"] is not None else None

    def connector_mouseReleaseEvent(self, set_Params):
        pass
    
    def mouseClickEvent(self, event):
        # CURRENT BUTTON
        self.mouse_Events["mouseClick_button"] = event.buttons()
        self.mouse_Events["mouseClick"] = True
        # print("mouseClick", True)

        # CURRENT POSITION
        self.mouse_Events["mouseClick_position"] = self.mouse_Events["mouseMove_position"]
        self.mouse_Events["mouseClick_position_scene"] = self.mouse_Events["mouseMove_position_scene"]

        # CURRENT ITEM
        self.mouse_Events["mouseClick_current_item"] = self.scene().itemAt(
            self.mouse_Events["mouseClick_position_scene"], QTransform()
        ) if self.mouse_Events["mouseClick_position_scene"] is not None else None

    def connector_mouseClickEvent(self, set_Params):
        pass
    
    def mouseDoubleClickEvent(self, event):
        # CURRENT BUTTON
        self.mouse_Events["mouseDoubleClick_button"] = event.buttons()
        self.mouse_Events["mouseDoubleClick"] = True
        # print("mouseDouble", True)
        
        # CURRENT POSITION
        self.mouse_Events["mouseDoubleClick_position"] = self.mouse_Events["mouseMove_position"]
        self.mouse_Events["mouseDoubleClick_position_scene"] = self.mouse_Events["mouseMove_position_scene"]
        
        # CURRENT ITEM
        self.mouse_Events["mouseDoubleClick_current_item"] = self.scene().itemAt(
            self.mouse_Events["mouseDoubleClick_position_scene"], QTransform()
        ) if self.mouse_Events["mouseDoubleClick_position_scene"] is not None else None

    def connector_mouseDoubleClickEvent(self, set_Params):
        pass

    def color_Picker(self, index=0):
        # lcdNumber_Set(
        #     [self.graphicsView_Developer_Camera_Processed],
        #     [
        #         [
        #             self.lcdNumber_Pointer_X,
        #             self.lcdNumber_Pointer_Y
        #         ],
        #         [
        #             self.mouse_Positions["mouseMove_position_scene"].x(
        #             ),
        #             self.mouse_Positions["mouseMove_position_scene"].y(
        #             )
        #         ]
        #     ]
        # )
        item = self.get_Item(index)
        red, green, blue = 0, 0, 0
        if item is not None:
            red, green, blue = get_Color(
                item.pixmap(),
                self.mouse_Events["mouseMove_position_scene"].x(),
                self.mouse_Events["mouseMove_position_scene"].y(),
                is_QT_Type=True
            )
            # lcdNumber_Set(
            #     [self.graphicsView_Developer_Camera_Processed],
            #     [
            #         [
            #             self.lcdNumber_Pointer_Color_Red,
            #             self.lcdNumber_Pointer_Color_Green,
            #             self.lcdNumber_Pointer_Color_Blue,
            #             self.lcdNumber_Pointer_Color_Grayscale,
            #             self.lcdNumber_Pointer_Color_Grayscale_Inverted
            #         ],
            #         [
            #             red,
            #             green,
            #             blue,
            #             int((red + green + blue) / 3)
            #             if red + green + blue != 0
            #             else 0,
            #             int(255 - (red + green + blue) / 3)
            #             if red + green + blue != 0
            #             else 0
            #         ]
            #     ]
            # )
        return red, green, blue

    # TODO:
    def mouseGrabber(self, event):
        pass
        """
        if self.scene().mouseGrabberItem() is None:
            if self.doubleClick_scene_current_item is not None:
                self.doubleClick_scene_current_item.grabMouse()
        else:
            self.scene().mouseGrabberItem().ungrabMouse()
        """

        """
        if self.scene().mouseGrabberItem() == self.doubleClick_scene_current_item:
            self.doubleClick_scene_current_item.ungrabMouse()
        else:
            self.doubleClick_scene_current_item.grabMouse()
        """

    def set_Background_Image(self, numpy_image, width_padding=0, height_padding=0):
        if numpy_image is not None:
            # numpy_To_QT_Type_Converter(image, QType=QPixmap, keep_aspect_ratio=True, convert_bgr_to_rgb=True, width=None, height=None, cv2_resize_algorithm=None):
            self.background_image = numpy_image
            qt_image = numpy_To_QT_Type_Converter(
                image=numpy_image,
                width=self.width() - width_padding,
                height=self.height() - height_padding
            )
            self.background_image_qt = qt_image
            
            # QT_Scene_Set_Item_To_Index(
            #     self.scene(),
            #     qt_image,
            #     -1
            # )
            QT_Scene_Set_Item_Background(
                self.scene(),
                qt_image
            )
            
            self.scene().setSceneRect(
                # QRect(10, 30, 161, 31)
                0, 0,
                qt_image.width(),
                qt_image.height()
            )
            
            if self.is_active_focus_to_image():
                self.focus_to_image()
            
            self.scene().update()
    
    def clear_Scene_Foreground(self):
        if self.background_image_qt is not None:
            QT_Scene_Set_Item(
                self.scene(),
                self.background_image_qt
            )
                
            self.scene().setSceneRect(
                # QRect(10, 30, 161, 31)
                0, 0,
                self.background_image_qt.width(),
                self.background_image_qt.height()
            )
            self.scene().update()
    
    def update_Scene_Size(self, width_padding=0, height_padding=0):
        qt_image = numpy_To_QT_Type_Converter(
            image=self.background_image,
            width=self.width() - width_padding,
            height=self.height() - height_padding
        )

        QT_Scene_Set_Item(
            self.scene(),
            qt_image
        )

        self.scene().setSceneRect(
            # QRect(10, 30, 161, 31)
            0, 0,
            qt_image.width(),
            qt_image.height()
        )
        self.scene().update()
    
    def get_Background_Item(self):
        return self.get_Item()
            
    def get_Item(self, index=0):
        background_item = self.scene().items(
            order=Qt.SortOrder.AscendingOrder
        )
        # background_item = background_item[0] \
        #     if len(background_item) else None
        return background_item[index] \
            if len(background_item) >= index + 1 else None
    
    def coordinate_Scaling_From_Scene(self, x, y):
        return coordinate_Scaling(
            x=int(x),
            y=int(y),
            old_w=int(self.scene().width()),
            old_h=int(self.scene().height()),
            new_w=self.background_image.shape[1],
            new_h=self.background_image.shape[0],
            task='RESIZE',
            is_dual=True
        ) if self.background_image is not None else (-1, -1)
    
    def coordinate_Scaling_To_Scene(self, x, y):
        return coordinate_Scaling(
            x=int(x),
            y=int(y),
            old_w=self.background_image.shape[1],
            old_h=self.background_image.shape[0],
            new_w=int(self.scene().width()),
            new_h=int(self.scene().height()),
            task='RESIZE',
            is_dual=True
        ) if self.background_image is not None else (-1, -1)
    
    ### #### ###
    ### TODO ###
    ### #### ###
    
    def handler_Mouse_Event(self, mouse_Event = "mouseDoubleClick"):
        # print(f"self.mouse_Events[{mouse_Event}]: {self.mouse_Events[mouse_Event]}")
        last_response = self.mouse_Events[mouse_Event]
        self.mouse_Events[mouse_Event] = False
        return last_response
    
    def qtimer_Draw_ROI_Rectangle(self, trigger_exit, controller_start="mouseDoubleClick", controller_end="mouseDoubleClick", mouse_Event="mouseDoubleClick_position", color="green", setAlphaF=0.43):
        if controller_end in self.mouse_Events and mouse_Event in self.mouse_Events and callable(trigger_exit):
            self.qtimer_drawer = qtimer_Create_And_Run(
                self,
                lambda: self.event_Draw_ROI_Rectangle(
                    trigger_exit, 
                    controller_start,
                    controller_end, 
                    mouse_Event, 
                    color,
                    setAlphaF
                ),
                delay = 100,
                is_single_shot=True
            )
        
            # self.add_ROI_Rect_item_to_history(
            #     self.draw_ROI_Rectangle(
            #         start_pos=self.mouse_Events[mouse_Events],
            #         controller_end=controller_end,
            #         color=color
            #     )
            # )
            return self.qtimer_drawer
        return None

    def event_Draw_ROI_Rectangle(self, trigger_exit, controller_start="mouseDoubleClick", controller_end="mouseDoubleClick", mouse_Event="mouseDoubleClick_position", color="green", setAlphaF=0.43):
        item = None
        if not self.is_Drawing_Now():
            self.is_Drawing_Now(True)
            self.wait_for_ClickEvent(
                control_function=lambda bool=None: self.is_mouseEvent(
                    controller_start, bool
                ),
                event_loop_start=None, 
                event_loop_end=None
            )
            item = self.draw_ROI_Rectangle(
                start_pos=self.mouse_Events[mouse_Event],
                create_holders=True,
                controller_end=controller_end,
                trigger_exit=trigger_exit,
                color=color,
                setAlphaF=setAlphaF
            )
            self.is_Drawing_Now(False)
        return item

    def draw_ROI_Rectangle(self, start_pos, controller_end, trigger_exit, color="gray", setAlphaF=0.43, create_holders=True):
        x, y = start_pos.x(), start_pos.y()
        w, h = x, y
        roi_item = self.add_ROI_Rectangle(
            x, y,
            w, h,
            init=True,
            keep_in_scene=True,
            create_holders=create_holders,
            hold_point_offset=15,
            color=color,
            setAlphaF=setAlphaF,
            setAlphaF_holders=0,
            pen_width=0
        )
        # roi_item.color(color, 0.25, 0)

        min_w, min_h = -x, -y
        max_w, max_h = self.scene().width() - x, self.scene().height() - y

        # controller_end(False)
        # roi_item.doubleClick_control(False)
        self.handler_Mouse_Event(mouse_Event = controller_end)
        while self.handler_Mouse_Event(mouse_Event = controller_end) is False:
            w = self.mouse_Events["mouseMove_position_scene"].x() - x
            h = self.mouse_Events["mouseMove_position_scene"].y() - y

            if w > max_w:
                w = max_w
            if h > max_h:
                h = max_h

            if w < min_w:
                w = min_w
            if h < min_h:
                h = min_h

            roi_item.resize_rect(
                QRectF(x, y, w, h)
            )
            self.qt_Priority()
            if trigger_exit():
                break

        # controller_end(False)
        # roi_item.doubleClick_control(False)
        self.handler_Mouse_Event(mouse_Event = controller_end)
        
        # Fix Wrong(Switched) ROI Item Points
        if w < 0:
            x += w
            w = -w

        if h < 0:
            y += h
            h = -h

        roi_item.resize_rect(QRectF(x, y, w, h))
        # scene.update()

        self.set_Last_Drawn_Item(roi_item)

        # s_x, s_y, e_x, e_y, w, h = roi_item.info()
        # print("ROI Item Added position at", s_x, s_y, e_x, e_y, "| with", w, ": height", h)
        return roi_item
    
    def add_ROI_Rectangle(
        self, 
        x, y,
        w, h,
        init,
        keep_in_scene,
        create_holders,
        hold_point_offset,
        color,
        setAlphaF,
        setAlphaF_holders,
        pen_width
    ):
        roi_item = ROI_Rectangle(
            self.scene(),
            x, y,
            w, h,
            init=init,
            keep_in_scene=keep_in_scene,
            create_holders=create_holders,
            hold_point_offset=hold_point_offset,
            color=color,
            setAlphaF=setAlphaF,
            setAlphaF_holders=setAlphaF_holders,
            pen_width=pen_width
        )
        self.set_Last_Drawn_Item(roi_item)
        return roi_item
    
    def get_Last_Drawn_Item(self):
        return self.last_drawn_item
    
    def set_Last_Drawn_Item(self, roi_item):
        self.last_drawn_item = roi_item
    
    def clear_Last_Drawn_Item(self):
        if self.last_drawn_item is not None:
            self.last_drawn_item.destroy()
            self.last_drawn_item = None
    
    def wait_for_ClickEvent(self, control_function, event_loop_start=None, event_loop_end=None): # lock_element=None):
        control_function(False)
        # if lock_element is not None:
        #     self.__revert_enabled_element(lock_element)
            
        while not control_function():
            
            if callable(event_loop_start):
                event_loop_start()
                
            self.qt_Priority()
            self.scene().update()
            
            if callable(event_loop_end):
                event_loop_end()

        # if lock_element is not None:
        #     self.__revert_enabled_element(lock_element)
        control_function(False)
    
    def qt_Priority(self):
        # https://stackoverflow.com/questions/8766584/displayin-an-image-in-a-qgraphicsscene
        QCoreApplication.processEvents()
    
    ### #### ###
    ### #### ###
    ### #### ###
    
    def clear_Coords_Double_Clicked(self):
        self.list_coords_double_clicked = list()
    
    def switch_focus_to_image(self, bool=None):
        self.is_focus_to_image = not self.is_focus_to_image \
            if bool is None else bool
    
    def is_active_focus_to_image(self):
        return self.is_focus_to_image
    
    def initialize_focus_to_image(self, zoom_x_times = 1.5, width = 100, height = 100):
        self.set_focus_to_image(zoom_x_times, width, height)
        self.switch_focus_to_image(bool=True)
    
    def set_focus_to_image(self, zoom_x_times = None, width = None, height = None):
        self.zoom_x_times = zoom_x_times if zoom_x_times is not None else self.zoom_x_times
        self.zoom_width = width if width is not None else self.zoom_width
        self.zoom_height = height if height is not None else self.zoom_height
        
    def focus_to_image(self):
        image_focus = None
        if self.mouse_Events["mouseMove_position_scene"] is not None:
            background_image = self.get_Background_Item()
            if background_image is not None:
                # QRect( x, y, width, height)
                zoomed_height = self.zoom_width * self.zoom_x_times
                zoomed_width = self.zoom_width * self.zoom_x_times
                
                start_x = self.mouse_Events["mouseMove_position_scene"].x() \
                    - self.zoom_width / 2 
                start_y = self.mouse_Events["mouseMove_position_scene"].y() \
                    - self.zoom_height / 2

                QT_Scene_Set_Item(
                    self.scene(), 
                    background_image
                )
                image_focus = background_image.pixmap().copy(start_x, start_y, self.zoom_width, self.zoom_width).scaled(
                    zoomed_width, zoomed_height,
                    aspectRatioMode=Qt.KeepAspectRatio, # if keep_aspect_ratio else Qt.IgnoreAspectRatio,
                    transformMode=Qt.FastTransformation
                )
                QT_Scene_Add_Item_To_Index(
                    self.scene(),
                    image_focus,
                    index=1
                )
        return image_focus

    ### ### ### ### ### ###
    ### QTIMER - RENDER ###
    ### ### ### ### ### ###

    def init_Render_QTimer(self, connector_stream, delay = 1):
        # print("callable(connector_stream)", callable(connector_stream))
        # print("connector_stream()", connector_stream())
        if callable(connector_stream):
            self.qtimer_render = qtimer_Create_And_Run(
                self,
                self.__QTimer_Render_Function,
                delay
            )
            self.render_buffer_caller = connector_stream
            return 0
        else:
            return 1
    
    def stop_Render_QTimer(self):
        self.qtimer_render.stop() if self.qtimer_render is not None else None
        
    def __QTimer_Render_Function(self):
        stream_image = self.render_buffer_caller()
        self.set_Background_Image(stream_image)

    ### ### ### ### ### ###
    ### ### ### ### ### ###
    ### ### ### ### ### ###

class ROI_Rectangle(QGraphicsRectItem):

    def __init__(self, scene=None, x=0, y=0, w=10, h=10, init=True, keep_in_scene=True, create_holders=True, hold_point_offset=15, color=None, setAlphaF=0.25, setAlphaF_holders=0.25, pen_width=0):

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
            self.init(
                scene=scene, 
                create_holders=create_holders, 
                color=color,
                setAlphaF=setAlphaF, 
                setAlphaF_holders=setAlphaF_holders, 
                pen_width=pen_width
            )

    def init(self, scene=None, create_holders=True, color=None, setAlphaF=0.25, setAlphaF_holders=0.1, pen_width=0):
        if scene is not None:
            # scene.addItem(self)
            # QT_Scene_Add_Item_To_Index(scene, self, -1)
            QT_Scene_Add_Item_To_Foreground(scene, self)
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
        self.hold_up = ROI_Rectangle_Hold_Point()
        self.hold_down = ROI_Rectangle_Hold_Point()
        self.hold_left = ROI_Rectangle_Hold_Point()
        self.hold_right = ROI_Rectangle_Hold_Point()
        
        self.hold_up.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_down.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_left.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        self.hold_right.color(color=color, setAlphaF=setAlphaF, pen_width=0)
        
        self.hold_up.setParentItem(self)
        self.hold_down.setParentItem(self)
        self.hold_left.setParentItem(self)
        self.hold_right.setParentItem(self)
        """
        self.mid_roi = ROI_Rectangle_Hold_Point(scene=self.scene())
        # self.scene().addItem(self.mid_roi)
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

            if not (self.scenerect.contains(left_top_value) and self.scenerect.contains(right_bottom_value)):
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
        if self.scene is not None:
            self.scene().update() if self.scene() is not None else None
        # return QGraphicsItem.itemChange(self, change, value)
        return super().itemChange(change, value)  # Call super

    def info(self):
        start_pos = self.mapToScene(self.boundingRect().topLeft())
        start_x, start_y = start_pos.x(), start_pos.y()

        end_pos = self.mapToScene(self.boundingRect().bottomRight())
        end_x, end_y = end_pos.x(), end_pos.y()

        width, height = self.rect().width(), self.rect().height()

        return start_x, start_y, end_x, end_y, width, height

    def color(self, color=None, setAlphaF=1.0, pen_width=0):
        if type(color) is str:
            self.current_color = color
            self.current_setAlphaF = setAlphaF
            self.current_pen_width = pen_width

            qt_color = QColor(self.current_color)
            qt_color.setAlphaF(self.current_setAlphaF)

            self.setBrush(QBrush(qt_color))
            self.setPen(
                QPen(
                    QColor(
                        self.current_color
                    ),
                    pen_width,
                    Qt.SolidLine
                )
            )

            if self.scene is not None:
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
        self.mouseRelease_pos_scene = self.graphicsView.mapToScene(
            event.pos())

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

    ### #### ###
    ### TODO ###
    ### #### ###
    """
    def grabMouse(self):
        self.is_grabMouse = True
        #print("Item Grabbed.")
        self.grabMouse_graphics_pos_scene = self.mouseMove_graphics_pos_scene
        while self.is_grabMouse:
            self.qt_Priority()
            self.grabMouse_graphics_pos_scene = self.mouseMove_graphics_pos_scene

    def ungrabMouse(self):
        #print("Item UN-Grabbed.")
        self.is_grabMouse = False

    ### #### ###
    ### #### ###
    ### #### ###
    """

    def qt_Priority(self):
        # https://stackoverflow.com/questions/8766584/displayin-an-image-in-a-qgraphicsscene
        QCoreApplication.processEvents()

    def destroy(self):
        self.scene().removeItem(self) if self.scene() is not None else None


class ROI_Rectangle_Hold_Point(QGraphicsRectItem):
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

        self.init(scene=scene, color=color,
                  setAlphaF=setAlphaF, pen_width=pen_width)

    def init(self, scene=None, color=None, setAlphaF=0.25, pen_width=0):
        if scene is not None:
            # scene.addItem(self)
            QT_Scene_Add_Item_To_Foreground(scene, self)
            self.scenerect = self.scene().sceneRect() # if self.scene() is not None else None

        if self.scene is not None:
            self.scenerect = self.scene().sceneRect() if self.scene() is not None else None

        if color is not None:
            self.color(color=color, setAlphaF=setAlphaF, pen_width=pen_width)

    def info(self):
        start_pos = self.mapToScene(self.boundingRect().topLeft())
        start_x, start_y = start_pos.x(), start_pos.y()

        end_pos = self.mapToScene(self.boundingRect().bottomRight())
        end_x, end_y = end_pos.x(), end_pos.y()

        width, height = self.rect().width(), self.rect().height()

        return start_x, start_y, end_x, end_y, width, height

    def color(self, color=None, setAlphaF=1.0, pen_width=0):
        if color is not None:
            self.current_color = color
            self.current_setAlphaF = setAlphaF
            self.current_pen_width = pen_width

            qt_color = QColor(self.current_color)
            qt_color.setAlphaF(self.current_setAlphaF)

            self.setBrush(QBrush(qt_color))
            self.setPen(QPen(QColor(self.current_color),
                        pen_width, Qt.SolidLine))

            if self.scene is not None:
                self.scene().update() if self.scene() is not None else None
        else:
            return self.brush, self.pen

        
### ### ### ### ### ## ## ## ### ### ### ### ###
### ### ### UI WINDOW CONFIGURATIONS ### ### ###
### ### ### ### ### ## ## ## ### ### ### ### ###
class Ui_ImagePopup_Modified(QMainWindow):  # , Ui_ImagePopup):
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
class Structure_UI(QMainWindow):

    def __init__(self, Parent=None, UI_File_Path="", title="", logger_level=logging.NOTSET):
        super(Structure_UI, self).__init__()
        self.internet_Parser_Format = "{}"
        self.internet_Receiver_Dict = dict()
        self.internet_Sender_Dict = dict()

        self.is_quit_app = False
        self.Developer_Settings_PopUp = None

        # self.UI_File_Path = ""
        self.Garbage_Collection = list()
        
        self.QTimer_Dict = dict()
        self.Buffer_Dict = dict()
        
        self.mouse_Events = dict()
        
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.UI_File_Path = UI_File_Path
        self.Parent = Parent
        self.load_UI(self, self.UI_File_Path) if self.UI_File_Path is not None else self.setupUi(self)
        self.init_Buffers()
        self.init_Logging(logger_level)

    @staticmethod
    def load_UI(UI, UI_File_Path):
        # Load the .ui file
        uic.loadUi(UI_File_Path, UI) if UI_File_Path != "" else None

        # self.Parent.configure_Other_Settings()
        UI.init_QTimers()
        UI.configure_Button_Connections()

    @staticmethod
    def set_Style_Sheet(QObject, style_sheet_file_path):
        QObject.setStyleSheet("") if style_sheet_file_path == "" or style_sheet_file_path == "default.qss" else QObject.setStyleSheet(
            open(style_sheet_file_path, "r").read())

    @staticmethod
    def set_Style_Sheet_Globally(style_sheet_file_path):
        Structure_UI.set_Style_Sheet(
            QApplication.instance(),
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
        q_item_model = QStandardItemModel()
        listView.setModel(
            q_item_model
        )
        return q_item_model

    ### ### ### ## ### ###
    ### MDI SUB WINDOW ###
    ### ### ### ## ### ###
    
    @staticmethod
    def create_Sub_Window(parent, mdiArea, UI_Class, title="Sub Window", UI_File_Path=""):
        sub_window = UI_Class(parent, UI_File_Path)
        """
        app, sub_window = app, ui = init_UI(
            Class_UI=UI_Class, 
            UI_File_Path="camera_api_developer_UI.ui", 
            is_Maximized=False
        )
        sub_window.Parent = parent
        """
        sub_window.setWindowTitle(title)
        sub_window.mdiArea = mdiArea
        return mdiArea.addSubWindow(sub_window)
    
    @staticmethod
    def destroy_Sub_Window(mdiArea, sub_window):
        mdiArea.removeSubWindow(sub_window)
    
    @staticmethod
    def graphicsView_Renderer(graphicsView, numpy_image, index=-1):
        if numpy_image is None:
            pass
        else:
            # numpy_To_QT_Type_Converter(image, QType=QPixmap, keep_aspect_ratio=True, convert_bgr_to_rgb=True, width=None, height=None, cv2_resize_algorithm=None):
            qt_image = numpy_To_QT_Type_Converter(
                image=numpy_image,
                width=graphicsView.width() - 5,
                height=graphicsView.height() - 5
            )
    
            if index == -1:
                QT_Scene_Set_Item(
                    graphicsView.scene(),
                    qt_image
                )
                
            else:
                QT_Scene_Add_Item_To_Index(
                    graphicsView.scene(),
                    qt_image,
                    index=index
                )

            graphicsView.scene().setSceneRect(
                # QRect(10, 30, 161, 31)
                0, 0,
                qt_image.width(),
                qt_image.height()
            )
            graphicsView.scene().update()

    @staticmethod
    def init_qt_graphicsView(graphicsView, mouseMoveEvent=None, mouseReleaseEvent=None, mouseDoubleClickEvent=None, mousePressEvent=None):
        # graphicsView.mousePressEvent = self.mousePressEvent_qt_graphicsView
        # https://doc.qt.io/qtforpython-5/PySide2/QtGui/QMouseEvent.html#PySide2.QtGui.PySide2.QtGui.QMouseEvent.w
        graphicsView.mouseMoveEvent = mouseMoveEvent
        graphicsView.mouseReleaseEvent = mouseReleaseEvent
        graphicsView.mouseDoubleClickEvent = mouseDoubleClickEvent
        graphicsView.mousePressEvent = mousePressEvent
        
    @staticmethod
    def init_qt_graphicsView_Scene(graphicsView):
        scene = QGraphicsScene()
        graphicsView.setScene(scene)
        return scene
    
    def mouseReleaseEvent_graphics(self, event):
        self.mouse_Events["position"]
        self.mouseRelease_graphics_pos = event.pos()
        self.mouseRelease_graphics_pos_scene = self.graphicsView.mapToScene(
            event.pos()
        )

    def mouseDoubleClickEvent_graphics(self, event):
        self.is_doubleClick_graphics_control(True)
        self.doubleClick_graphics_pos = event.pos()
        self.doubleClick_graphics_pos_scene = self.graphicsView.mapToScene(
            event.pos())

        self.doubleClick_scene_current_item = self.scene().itemAt(
            self.doubleClick_graphics_pos_scene, QTransform())
        # print("doubleClick_scene_item", self.doubleClick_scene_current_item)

        """
        if self.scene().mouseGrabberItem() is None:
            if self.doubleClick_scene_current_item is not None:
                self.doubleClick_scene_current_item.grabMouse()
        else:
            self.scene().mouseGrabberItem().ungrabMouse()
        """

        """
        if self.scene().mouseGrabberItem() == self.doubleClick_scene_current_item:
            self.doubleClick_scene_current_item.ungrabMouse()
        else:
            self.doubleClick_scene_current_item.grabMouse()
        """

    def mouseClickEvent_graphics(self, event):
        self.is_click_graphics_control(True)
        self.click_graphics_pos = event.pos()
        self.click_graphics_pos_scene = self.graphicsView.mapToScene(
            event.pos())

        self.click_scene_current_item = self.scene().itemAt(
            self.click_graphics_pos_scene, QTransform())
    

    """
    ### ### ### ### ###
    ### BUFFER APIs ###
    ### ### ### ### ###

    @staticmethod
    def set_Buffer(buffer, data):
        buffer.append(data)

    @staticmethod
    def get_Buffer():
        return buffer.append(data)
    
    ### ### ### ### ###
    ### ### ### ### ###
    ### ### ### ### ###
    """
    
    ### ### ### ### ### ###
    ###  INTERNET APIs  ###
    ### ### ### ### ### ###

    ### ### ### ## ## ## ### ### ###
    ### INTERNET RECEIVER OBJECT ###
    ### ### ### ## ## ## ### ### ###
    def init_Internet_Receiver(
        self, 
        ip_receiver="127.0.0.1", 
        port_receiver=3333, 
        logger_level=logging.CRITICAL,
        delay=0.0000001,
        parsing_format=None,
        regex=None,
        special_data_parsing=None,
        disable_Logger=False
    ):
        global Internet_Receiver
        from structure_internet import Internet_Receiver
        
        _internet_Receiver = Internet_Receiver(
            host=ip_receiver,
            port=port_receiver,
            timeout=2,
            set_blocking=False,
            logger_level=logger_level,
            #parsing_format="\n\|--- *"
            #parsing_format="\[|]|\n",
            parsing_format=parsing_format,
            regex=regex,
            delay=delay,
            max_buffer_limit=30
        )
        _internet_Receiver.logger.disabled = disable_Logger

        if special_data_parsing is not None:
            _internet_Receiver.data_Parsing = special_data_parsing
        # _internet_Receiver.buffer_Overwrite()

        _internet_Receiver.start()
        self.internet_Receiver_Dict[_internet_Receiver.name] = _internet_Receiver
        return _internet_Receiver.name
        
    def Internet_Receiver_Buffer_Returner(self, name):
        return self.internet_Receiver_Dict[name].buffer_Get()
        
    ### ### ### ## ## ## ### ### ###
    ### ### ### ## ## ## ### ### ###
    ### ### ### ## ## ## ### ### ###
        
    ### ### ### ## ### ### ### ###
    ### INTERNET SENDER OBJECT ###
    ### ### ### ## ### ### ### ###
    def init_Internet_Sender(
        self, 
        ip_sender="127.0.0.1", 
        port_sender=3333, 
        logger_level=logging.CRITICAL,
        delay=0.0000001,
        parsing_format=None,
        regex=None,
        disable_Logger=False,
        max_buffer_limit=30
    ):
        global Internet_Sender
        from structure_internet import Internet_Sender
        
        _internet_Sender = Internet_Sender(
            host=ip_sender,
            port=port_sender,
            timeout=2,
            set_blocking=False,
            logger_level=logger_level,
            parsing_format=parsing_format,
            regex=regex,
            delay=delay,
            max_buffer_limit=max_buffer_limit
        )
        _internet_Sender.logger.disabled = disable_Logger
        # _internet_Sender.buffer_Overwrite()
        
        # _internet_Sender.buffer_Append(data, lock_until_done=True)
        _internet_Sender.start()
        self.internet_Sender_Dict[_internet_Sender.name] = _internet_Sender
        return _internet_Sender.name
        
    def Internet_Sender_Buffer_Append(self, name, data, lock_until_done=True):
        self.internet_Sender_Dict[name].buffer_Append(data, lock_until_done=lock_until_done)
        
    def set_Internet_Parser_Format(self, parser_format):
        self.internet_Parser_Format = parser_format
        
        
    ### ### ### ## ### ### ### ###
    ### ### ### ## ### ### ### ###
    ### ### ### ## ### ### ### ###
            
    
    ### ### ### ### ### ###
    ### ### ### ### ### ###
    ### ### ### ### ### ###

    ### ### ### ## ### ###
    ### ### ### ## ### ###
    ### ### ### ## ### ###

    def init_Logging(self, logger_level=logging.NOTSET, filename="", format='[%(asctime)s][%(levelname)s] %(name)s : %(message)s', datefmt="%Y-%m-%d %H:%M:%S"):
        self.Logger = logging.getLogger('[{}]'.format(self.windowTitle()))
        # self.Logger = logging.getLogger('Thread {} ({})'.format(self.thread_ID, self.name))

        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
        # self.Logger.setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            format,
            datefmt
        )
        handler.setFormatter(formatter)
        self.Logger.addHandler(handler)
        self.Logger.setLevel(logger_level)
        
        """
            is_logger_disabled: True/False
            logger_level
                |-logging.INFO
                |-logging.ERROR
                |-logging.CRITICAL
                |-logging.DEBUG
        """
        """
        logging.basicConfig(
            filename=filename,
            level=logger_level,
            format=format,
            datefmt=datefmt
        )
        """

    def init_QTimers(self):
        self.QTimer_Dict["garbage_Collector_Cleaner"] = qtimer_Create_And_Run(
            self, 
            self.garbage_Collector_Cleaner,
            60000
        )
        self.QTimer_Dict["qt_function"] = qtimer_Create_And_Run(
            self,
            connection=(
                lambda: self.qtimer_QTFunction_Caller_Event(
                    self.Buffer_Dict["qt_function"]
                )
            ),
            delay=5,
            is_needed_start=True,
            is_single_shot=False
        )
        self.QTimer_Dict["qt_messagebox"] = qtimer_Create_And_Run(
            parent=self,
            connection=(
                lambda: self.qtimer_QMessageBox_Event(
                    self.Buffer_Dict["qt_messagebox"]
                )
            ),
            delay=25,
            is_needed_start=True,
            is_single_shot=False
        )
        self.QTimer_Dict["qt_color_painter"] = qtimer_Create_And_Run(
            parent=self,
            connection=(
                lambda: self.qtimer_QColor_Painter_Event(
                    self.Buffer_Dict["qt_color_painter"]
                )
            ),
            delay=10,
            is_needed_start=True,
            is_single_shot=False
        )
        self.QTimer_Dict["qt_object_text"] = qtimer_Create_And_Run(
            parent=self,
            connection=(
                lambda: self.qtimer_QObject_Text_Event(
                    self.Buffer_Dict["qt_object_text"]
                )
            ),
            delay=10,
            is_needed_start=True,
            is_single_shot=False
        )
        self.QTimer_Dict["qt_file_dialog"] = qtimer_Create_And_Run(
            parent=self,
            connection=(
                lambda: self.qtimer_QFileDialog_Event(
                    self.Buffer_Dict["qt_file_dialog"]
                )
            ),
            delay=10,
            is_needed_start=True,
            is_single_shot=False
        )
        
    def init_Buffers(self):

        ###############################################################
        #################### QT UI UPDATER EVENTS #####################
        ###############################################################

        self.Buffer_Dict["qt_function"] = Structure_Buffer(name="{}:qt_function".format(self.windowTitle()), max_limit=500)

        self.Buffer_Dict["qt_messagebox"] = Structure_Buffer(name="{}:qt_messagebox".format(self.windowTitle()), max_limit=500)

        self.Buffer_Dict["qt_color_painter"] = Structure_Buffer(name="{}:qt_color_painter".format(self.windowTitle()), max_limit=500)

        self.Buffer_Dict["qt_object_text"] = Structure_Buffer(name="{}:qt_object_text".format(self.windowTitle()), max_limit=500)
        
        self.Buffer_Dict["qt_file_dialog"] = Structure_Buffer(name="{}:qt_file_dialog".format(self.windowTitle()), max_limit=1)
        
        self.Buffer_Dict["qt_file_dialog_return"] = Structure_Buffer(name="{}:qt_file_dialog_return".format(self.windowTitle()), max_limit=1)
        
        ###############################################################
        ###############################################################
        ###############################################################
        
    def garbage_Collector_Cleaner(self):
        if len(self.Garbage_Collection) > 10:
            self.Garbage_Collection.clear()

    def garbage_Collector_Add(self, garbage):
        self.Garbage_Collection.append(garbage)
        
    def garbage_Collector_Pop(self, garbage_index=0):
        return self.Garbage_Collection.pop(garbage_index)
        
    def configure_Button_Connections(self):
        pass
        #stdo("Overwrite configure_Button_Connections Function!")

    def configure_Other_Settings(self):
        print("Overwrite configure_Other_Settings Function!")

    def is_Quit_App(self):
        return self.is_quit_app

    def closeEvent(self, a0: QCloseEvent):
        self.is_quit_app = True
        self.Garbage_Collection.clear()
        qtimer_All_Stop(self.QTimer_Dict)
        # stdo(1, "Closing with Class Default closeEvent")
        return super().closeEvent(a0)
    
    ############################################################
    ######## QT MESSAGEBOX AND QT PAINTER STRUCTURE ############
    ############################################################

    def qtimer_QTFunction_Caller_Event(self, qt_function_buffer):
        if len(qt_function_buffer) > 0:
            buffer = qt_function_buffer.nts_Pop()
            self.QTFunction_Caller_Event(buffer[0], buffer[1])

    def QTFunction_Caller_Event(self, qt_function, qt_function_params):
        if qt_function_params is None or qt_function_params == []:
            qt_function()
        else:
            qt_function(*qt_function_params)

    def QTFunction_Caller_Event_Add(self, event):
        self.Buffer_Dict["qt_function"].nts_Append(event)

    def qtimer_QMessageBox_Event(self, qt_messagebox_buffer):
        if len(qt_messagebox_buffer) > 0:
            buffer = qt_messagebox_buffer.nts_Pop()
            self.QMessageBox_Event(buffer[0], buffer[1:])

    def QMessageBox_Event(self, qt_messagebox_flag, qt_messagebox_buffer):
        if qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.INFORMATION:
            QMessageBox.information(self, *qt_messagebox_buffer)

        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.WARNING:
            QMessageBox.warning(self, *qt_messagebox_buffer)

        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.WARNING_OK:
            QMessageBox.warning(self, *qt_messagebox_buffer, QMessageBox.Ok)
        
        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.QUESTION:
            ret = QMessageBox.question(self, *qt_messagebox_buffer, QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                return True
            else:
                return False

        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.CRITICAL:
            QMessageBox.critical(self, *qt_messagebox_buffer)

        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.ABOUT:
            QMessageBox.about(self, *qt_messagebox_buffer)

        elif qt_messagebox_flag == QT_MESSAGEBOX_FLAGS.ABOUT_QT:
            QMessageBox.aboutQt(self, *qt_messagebox_buffer)

    def QMessageBox_Event_Add(self, event):
        self.Buffer_Dict["qt_messagebox"].nts_Append(event)

    def qtimer_QColor_Painter_Event(self, qt_color_painter_buffer):
        if len(qt_color_painter_buffer) > 0:
            self.QColor_Painter_Event(*qt_color_painter_buffer.nts_Pop())

    def QColor_Painter_Event(self, qt_object, qt_color_painter_buffer):
        qt_object.setStyleSheet(qt_color_painter_buffer)

    def QColor_Painter_Event_Add(self, event):
        self.Buffer_Dict["qt_color_painter"].nts_Append(event)

    def qtimer_QObject_Text_Event(self, qt_object_text_buffer):
        if len(qt_object_text_buffer) > 0:
            self.QObject_Text_Event(*qt_object_text_buffer.nts_Pop())

    def QObject_Text_Event(self, qt_object, qt_object_text_buffer):
        qt_object.setText(str(qt_object_text_buffer))

    def QObject_Text_Event_Add(self, event):
        self.Buffer_Dict["qt_object_text"].nts_Append(event)
        
    ############################################################
    ############################################################
    ############################################################
    
    ############################################################
    ################ QT FILE DIALOG STRUCTURE ##################
    ############################################################
    
    def qtimer_QFileDialog_Event(self, qt_filedialog_buffer):
        if len(qt_filedialog_buffer) > 0:
            buffer = qt_filedialog_buffer.nts_Pop()
            self.Buffer_Dict["qt_file_dialog_return"].append(self.QFileDialog_Event(buffer))

    def QFileDialog_Event(self, qt_filedialog_function, qt_filedialog_buffer):
        if qt_filedialog_function == 'getOpenFileName':
            return QFileDialog.getOpenFileName(self, *qt_filedialog_buffer)
        elif qt_filedialog_function == 'getExistingDirectory':
            return QFileDialog.getExistingDirectory(self, *qt_filedialog_buffer)

    def QFileDialog_Event_Add(self, event):
        self.Buffer_Dict["qt_file_dialog"].nts_Append(event)
    
    ############################################################
    ############################################################
    ############################################################
    
    @staticmethod
    def disable_Elements(element_list):
        for element in element_list:
            element.setEnabled(False)
            
    @staticmethod
    def enable_Elements(element_list):
        for element in element_list:
            element.setEnabled(True)
    
    def qt_Priority(self):
        # https://stackoverflow.com/questions/8766584/displayin-an-image-in-a-qgraphicsscene
        QCoreApplication.processEvents()
        
        if self.is_Quit_App():
            # self.qtimer_stop_dict(self.dict_qtimer)
            self.closeEvent(None)

            #seppuku()
            #sys.exit()
        sleep(0.00001)
    

"""
class Structure_UI_MDI_SubWindow(Structure_UI, QMdiSubWindow):
    def __init__(self, *args, **kwargs):
        super(Structure_UI_MDI_SubWindow, self).__init__(*args, **kwargs)

    def closeEvent(self, *args, **kwargs):
        super(Structure_UI_MDI_SubWindow, self).closeEvent(*args, **kwargs)
        
        self.Parent.destroy_Sub_Window(None, self)
"""
        
### ### ### ### ## ## ## ### ### ### ###
### ### ### CUSTOM FUNCTIONS ### ### ###
### ### ### ### ## ## ## ### ### ### ###

def init_UI(Class_UI, UI_File_Path="test.ui"):
    global app

    if not "app" in globals():
        app = QApplication(sys.argv)
    else:
        stdo(2, "App already defined. No need action...")

    ui = Class_UI(UI_File_Path=UI_File_Path)

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


def init_and_run_UI(title, Class_UI, run=True, UI_File_Path="test.ui", show_UI=True, is_Maximized=False):
    app, ui = init_UI(
        Class_UI=Class_UI,
        UI_File_Path=UI_File_Path,
    )
    if run:
        run_UI(
            app=app,
            UI=ui,
            title=title,
            show_UI=show_UI,
            is_Maximized=is_Maximized
        )
    return app, ui


if __name__ == "__main__":

    app, ui = init_and_run_UI(
        title="Test",
        Class_UI=Structure_UI,
        UI_File_Path=sys.argv[1] if len(sys.argv) == 2 else "test.ui"
    )
