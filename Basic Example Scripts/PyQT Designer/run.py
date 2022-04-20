import libs
import structure_ui
from tools import list_files


### ### ### ### ### # ### ### ### ### ###
### ### ### UI CONFIGURATIONS ### ### ###
### ### ### ### ### # ### ### ### ### ###
class Ui_Modified(structure_ui.Structure_UI):
    
    Main_Window = None
    Developer_Settings_PopUp = None
    
    UI_File_Path = ""
    themes_list = {
        "default": "default.qss"
    }

    def __init__(self, *args, obj=None, **kwargs):
        super(Ui_Modified, self).__init__(*args, **kwargs)
        """
        self.UI_File_Path = UI_File_Path
        self.load_UI(self, self.UI_File_Path)
        # self.configure_Button_Connections()
        """

        self.init()

    def init(self):
        self.themes_list = self.load_themes_to_combobox(
            self.comboBox_theme_chooser, 
            "themes"
        )
        
    def configure_Button_Connections(self):
        self.comboBox_theme_chooser.currentIndexChanged.connect(
            self.comboBox_theme_chooser_currentIndexChanged
        )
        self.pushButton_theme_chooser.clicked.connect(self.pushButton_theme_chooser_Clicked)
        
    def comboBox_theme_chooser_currentIndexChanged(self):
        self.set_Style_Sheet_Globally(
            self.themes_list[self.comboBox_theme_chooser.currentText()] if self.comboBox_theme_chooser.currentText() in self.themes_list else ""
        )
    
    def pushButton_theme_chooser_Clicked(self):
        self.themes_list = self.load_themes_to_combobox(
            self.comboBox_theme_chooser,
            "themes",
            True
        )


if __name__ == "__main__":
    # import sys
    
    # title, Class_UI, run=True, UI_File_Path= "test.ui", qss_File_Path = ""
    app, ui = structure_ui.init_and_run_UI(
        "Theme Viewer",
        Ui_Modified,
        #UI_File_Path="test.ui" if len(sys.argv) == 1 else sys.argv[1]
        UI_File_Path="test.ui"
    )
