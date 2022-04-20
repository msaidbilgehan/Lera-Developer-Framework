# https://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt

import sys

#For PyQt5 :
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget

import libraries.libs
from libraries.structure_ui import Structure_UI, init_and_run_UI

class UIWindow_0(QWidget):
    def __init__(self, parent=None):
        super(UIWindow_0, self).__init__(parent)
        # mainwindow.setWindowIcon(QtGui.QIcon('PhotoIcon.png'))
        self.change_page = QPushButton('(UIWindow_0) Change Page', self)
        self.change_page.move(50, 350)


class UIWindow_1(QWidget):
    def __init__(self, parent=None):
        super(UIWindow_1, self).__init__(parent)
        self.change_page = QPushButton("(UIWindow_1) Change Page", self)
        self.change_page.move(100, 350)


class UIWindow_2(QWidget):
    def __init__(self, parent=None):
        super(UIWindow_2, self).__init__(parent)
        self.change_page = QPushButton("(UIWindow_2) Change Page", self)
        self.change_page.move(100, 350)


class MainWindow(Structure_UI):
    Windows = None

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.setGeometry(50, 50, 400, 450)
        # self.setFixedSize(400, 450)
        
        self.Windows_Object = {
            "UIWindow_2": UIWindow_2,
            "UIWindow_1": UIWindow_1,
            "UIWindow_0": UIWindow_0
        }
        
        self.switch_Page("UIWindow_0", "UIWindow_1", "UIWindow_2")

    def switch_Page(self, page_name, connected_page, connected_page_connected_page):
        self.setWindowTitle(page_name)

        self.Windows = self.Windows_Object[page_name](self)
        self.Windows.change_page.clicked.connect(
            lambda: self.switch_Page(
                connected_page, 
                connected_page_connected_page, 
                page_name
            )
        )
        self.setCentralWidget(self.Windows)
        self.show()


if __name__ == '__main__':
    """
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec())
    """
    app, ui = init_and_run_UI(
        "Page Switch Tester",
        MainWindow,
        UI_File_Path="test.ui"
    )
    sys.exit(app.exec())
