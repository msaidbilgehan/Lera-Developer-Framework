# https://stackoverflow.com/questions/62291103/how-do-i-dock-a-subwindow-in-an-mdi-area-in-pyqt5-using-qtdesigner
# https://codeloop.org/pyqt5-make-multi-document-interface-mdi-application/

import os
import sys
from PyQt6 import QtCore, QtWidgets, uic

uiFile = 'mdi_control_UI.ui'

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.initUI()
    def initUI(self):
        #load the UI page
        self.ui_main = uic.loadUi(uiFile, self)
        self.actionCreateWindow.triggered.connect(
            lambda: self.fileBarTrig('test')
        )

    def fileBarTrig(self, p):
        sw1 = self.mdiArea.addSubWindow(self.subwindow)
        sw1.show()
        sw2 = self.mdiArea.addSubWindow(self.subwindow_2)
        sw2.show()
        self.mdiArea.tileSubWindows()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()