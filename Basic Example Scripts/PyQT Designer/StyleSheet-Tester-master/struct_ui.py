import sys

from PyQt5 import uic, QtWidgets  # , QtCore, QtGui
# , QIcon, QPixmap, QPainter, QPen, QBrush, QImage, QTransform
from PyQt5.QtGui import QPalette, QColor


### ### ### ### ### # ### ### ### ### ###
### ### ### UI CONFIGURATIONS ### ### ###
### ### ### ### ### # ### ### ### ### ###
class Structure_UI(QtWidgets.QMainWindow):
    
    Main_Window = None
    Developer_Settings_PopUp = None
    
    UI_File_Path = ""

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
        if style_sheet_file_path != "":
            QObject.setStyleSheet(open(style_sheet_file_path, "r").read())
            
    @staticmethod
    def set_Style_Sheet_Globally(style_sheet_file_path):
        Structure_UI.set_Style_Sheet(
            QtWidgets.QApplication.instance(), 
            style_sheet_file_path
        )

    """
    @staticmethod
    def load_themes_to_combobox(comboBox, themes_path, clear_before_update=True):
        themes_subtree = list_files(
            themes_path,
            extensions=[".qss"],
            recursive=True
        )
        themes_dict = dict()
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
    """

    def configure_Button_Connections(self):
        print("Overwrite configure_Button_Connections Function!")

    def configure_Other_Settings(self):
        print("Overwrite configure_Other_Settings Function!")



### ### ### ### ## ## ## ### ### ### ###
### ### ### CUSTOM FUNCTIONS ### ### ###
### ### ### ### ## ## ## ### ### ### ###


def init_and_run_UI(title, Class_UI, run=True, UI_File_Path= "test.ui", show_UI = True, is_Maximized=False):
    app = QtWidgets.QApplication(sys.argv)

    if len(sys.argv) == 2:
        UI_File_Path = sys.argv[1]

    ui = Class_UI(UI_File_Path = UI_File_Path)
    ui.setWindowTitle(title)
        
    if show_UI:
        if is_Maximized:
            ui.showMaximized()  # Show in fully fitted window size#
        else:
            ui.show() #Show at default window size#

    if run is True:
        sys.exit(app.exec_())
    else:
        # Need to be returned to run namespace, else it won't show window render and stuck
        return app, ui


if __name__ == "__main__":
    app, ui = init_and_run_UI("Test", Structure_UI)
