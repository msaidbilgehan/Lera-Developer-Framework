import libs
from structure_ui import init_and_run_UI
from constructor_ui import Ui_Deep_Face
# from structure_system import System_Object

if __name__ == "__main__":
    """
    system_info = System_Object()
    system_info.thread_print_info()
    """
    app, ui = init_and_run_UI(
        "Deep Face UI",
        Ui_Deep_Face,
        UI_File_Path="deep_face_UI.ui"
    )
