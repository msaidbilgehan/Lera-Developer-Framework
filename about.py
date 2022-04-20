# References;
#   - Variable Existence Control / "Is variable created?" Control
#       - https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists
#   - MDI SubWindow Examples:
#       - https://stackoverflow.com/questions/62291103/how-do-i-dock-a-subwindow-in-an-mdi-area-in-PyQt5-using-qtdesigner
#       - https://codeloop.org/PyQt5-make-multi-document-interface-mdi-application/
#   - Mouse Events
#       - https://doc.qt.io/qtforpython-5/PySide2/QtGui/QMouseEvent.html#PySide2.QtGui.PySide2.QtGui.QMouseEvent.w

import sys

### ### ### ### ## ### ###
### EXTERNAL LIBRARIES ###
### ### ### ### ## ### ###
import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication

### ### ### ### ## ### ###
### ### ### ### ## ### ###
### ### ### ### ## ### ###

### ### ### ### ### # ### ### ### ### ###
### ### ### UI CONFIGURATIONS ### ### ###
### ### ### ### ### # ### ### ### ### ###
class About_UI(QMainWindow):
    def __init__(self, Parent=None):
        super(About_UI, self).__init__()
        self.Parent = Parent
        uic.loadUi("about.ui", self)
        
        self.image_msaidbilgehan = cv2.imread('about/msaidbilgehan.jpeg')
        self.graphicsView_msaidbilgehan.set_Background_Image(
            self.image_msaidbilgehan
        )
        # self.graphicsView_msaidbilgehan.init_Render_QTimer(
        #     lambda: self.image_msaidbilgehan,
        #     delay = 1
        # )
        
        self.image_msyasak = cv2.imread('about/msyasak.jpg')
        self.graphicsView_msyasak.set_Background_Image(
            self.image_msyasak
        )
        # self.graphicsView_msyasak.init_Render_QTimer(
        #     lambda: self.image_msyasak,
        #     delay = 1
        # )

    def resizeEvent(self, event):
        self.update_GraphicsView()
        return super(About_UI, self).resizeEvent(event)

    def update_GraphicsView(self):
        self.graphicsView_msaidbilgehan.update_Scene_Size()
        self.graphicsView_msyasak.update_Scene_Size()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = About_UI()
    ui.setWindowTitle("About")
    ui.show()  # Show at default window size#
    ui.update_GraphicsView()

    sys.exit(app.exec())
