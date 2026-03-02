import libs
# import logging
from structure_ui import init_UI, run_UI  # , init_and_run_UI
from constructor_ui import Ui_Image_Difference

# logging.basicConfig(
#     filename="details_harvester.log",
#     level=logging.WARNING,
#     # format=format,
#     # datefmt=datefmt
# )

# #### #### #### #### #
# Program Information #
# #### #### #### #### #
BRAND_NAME = "Harvester"
APP_NAME = "Kamera Kontrollü PCB Lehim Cihazı"
VERSION = "1.0.0"
VERSION_NAME = "BETA"
TITLE = BRAND_NAME + " | " + APP_NAME + " v" + VERSION + " " + VERSION_NAME

LOGGER_LEVEL = 0  # 0, 1, 2, 3, 4


if __name__ == "__main__":
    #system_info = System_Object()
    #system_info.thread_print_info()

    # app, ui = init_and_run_UI(
    #     "Process UI",
    #     Ui_Harvester,
    #     UI_File_Path="harvester_UI.ui"
    # )
    title = "Process UI"
    Class_UI = Ui_Image_Difference
    UI_File_Path = "image_difference.ui"
    show_UI = True
    is_Maximized = False

    app, ui = init_UI(
        Class_UI,
        UI_File_Path=UI_File_Path,
    )

    """
    ui.init_Internet_Objects(

        ip_sender=IP_SENDER,
        port_sender=PORT_SENDER,
        regex_sender=None,
        parsing_format_sender=None,

        ip_receiver=IP_RECEIVER,
        port_receiver=PORT_RECEIVER,
        regex_receiver=None,
        parsing_format_receiver=None,

        internet_Parser_Format=INTERNET_DATA_FORMAT,

        delay=0.0000001
    )
    """

    run_UI(
        app=app,
        UI=ui,
        title=title,
        show_UI=show_UI,
        is_Maximized=is_Maximized
    )
