#cython: language_level=3, boundscheck=False
import time
import logging

"""
Threading with Logging;
    - https://stackoverflow.com/questions/16929639/python-logging-from-multiple-threads
"""


def init_logging(
    level=logging.INFO,
    filename="app.log",
    autoformat=True,
    format="{} | {} - {} >> {}",
    datefmt="%H:%M:%S",
    processing=False,
    filemode="w",
    # create_logger=False,
    logger_name="Logger",
):
    # https://realpython.com/python-logging
    # Create handlers
    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Create handlers
    user_handler = logging.StreamHandler(filename)

    # Create formatters and add it to handlers
    user_handler.setLevel(level)

    if autoformat:
        if processing:
            format.format("%(asctime)s", "%(process)d",
                          "%(levelname)s", "%(message)s")
        else:
            format.format("%(asctime)s", "", "%(levelname)s", "%(message)s")

    user_handler.setFormatter(format)
    
    # TODO: Need to fix below config. 
    # Exception: AttributeError: 'StreamHandler' object has no attribute 'basicConfig'
    """
    user_handler.basicConfig(
        filename=filename,
        format=format,
        level=level,
        datefmt=datefmt,
        filemode=filemode,
    )
    """
    return logger


def logging_load():
    with open("config.yaml", "r") as f:
        import yaml

        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


"""
def logging_save():
    import yaml
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
"""


####################################################################


def stdo(flag = 4, string="No Info Found", frame_info=None):

    # Proper Time Parsing
    currentTime = ""
    for part in get_time().split("_"):
        if len(part) != 2:
            currentTime = currentTime + "0" + part[0] + ":"
        else:
            currentTime = currentTime + part + ":"
    currentTime = currentTime[:-1]

    if string == "No Info Found":
        print(
            """
[{0}] - ERR: Invalid arguments!
                       |- FLAG: {1}
                       |- FRAMEINFO: {2}
                       '- STRING: {3}\n
""".format(
                currentTime, flag, frame_info, string
            )
        )

    elif flag == 1:
        print("[{0}] - INF: {1}".format(currentTime, string))

    elif flag == 2:
        print("[{0}] - WRN: {1}".format(currentTime, string))

    elif flag == 3:
        if frame_info is not None:
            print(
                """
[{0}] - ERR:   Invalid arguments!
                    |- PATH: {1} ({2})
                    '- MESSAGE: {3}
    """.format(
                    currentTime,
                    str(frame_info.filename),
                    str(frame_info.lineno),
                    string,
                )
            )
        else:
            print("[{0}] - ERR:    {1}".format(currentTime, string))
    elif flag == 4:
        print("[{0}] - DBG:    {1}".format(currentTime, string))
    
    return 0


def get_time(level=1):

    if level == 0:  # To get raw time (For seeding random library)
        return time.time()
    if level == 1 or level == 2:
        string_current_time = str()
        current_time = None

        if level == 1:  # To get only clock
            current_time = time.localtime(time.time())[3:6]

        elif level == 2:  # To get only date
            current_time = time.localtime(time.time())[:3]

        for part in current_time:
            string_current_time += str(part) + "_"
        return string_current_time[:-1]

    elif level == 3:  # To get date-clock (For file names)
        return "{0}-{1}".format(get_time(level=2), get_time(level=1))

    elif level == 4:  # To get date | clock (For output logs)
        return "{0} | {1}".format(get_time(level=2), get_time(level=1))

    else:
        return ""
