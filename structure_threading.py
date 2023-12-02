# References;
#   __new__ vs __init__:
#       - https://stackoverflow.com/questions/674304/why-is-init-always-called-after-new

# BUILT-IN LIBRARIES
#from enum import Enum
from time import sleep
import threading
import logging
from typing import Iterable
from structure_data import Structure_Buffer

# EXTERNAL LIBRARIES
#import cv2
#from ordered_enum import OrderedEnum

# CUSTOM LIBRARIES
from stdo import stdo
import secrets

#from structure_data import structure_buffer

# TEST LIBRARIES
# import random
# from time import time
"""
def connector(self, connection, hook, delay=0.001, quit_trigger=None):
    while True:
        connection(hook)
        sleep(delay)
        if self.statement_quit:
            break
        if quit_trigger is not None:
            if quit_trigger():
                break
"""
"""
class __Thread_Object():
    statement_quit = False
    
    def connector(self, connection, hook, delay=0.001, quit_trigger=None):
        while True:
            connection(hook)
            sleep(delay)
            if self.statement_quit:
                break
            if quit_trigger is not None:
                if quit_trigger():
                    break
"""


class Thread_Object(threading.Thread):
    """
        Thread Object is a custom threading structure to use with;
         - Connector&Hook; A function is a connection which takes another function with variable or function parameters or only variable parameters or no parameter as a hook which will be called on every given time
         or
         - Task; A custom function itself can be Task

        Usage;
            Create a object with Thread_Object
            If it is a connector;
                Connector thread will be activated after connection_init function call
            If it is a Task;
                Developer should overwrite the task function
            After these step above, created object will be run with start function call

            IMPORTANT: Threads can only be started once. So run once at a time with specified delay and/or quit statement and/or manuel quit 

        Example;
            connector_thread_1 = Thread_Object("The 1", delay=0.1, logger_level=4, set_Deamon=False, run_number=1)

            # Run stdo with [1, "Message: 1"] parameter(s)
            connector_thread_1.init(stdo, [1, "Message: 1"])
            connector_thread_1.start()
            sleep(1)

            # Run custom_connection with (stdo) parameter(s)
            connector_thread_1.init(custom_connection, (stdo))
            connector_thread_1.start()
            sleep(1)

            # Run custom_connection with no parameter(s)
            connector_thread_1.init(custom_connection)
            connector_thread_1.start()
            sleep(1)

            # Run default task with no parameter(s)
            connector_thread_1.start()
            sleep(1)

            # Run task overwrite by custom_connection with no parameter(s)
            connector_thread_1.task = custom_connection
            connector_thread_1.start()
            sleep(1)

            connector_thread_2 = Thread_Object("The 2", delay=0.01, logger_level=4)
            connector_thread_2.init(stdo, ("Message: 2"))
            connector_thread_2.init(custom_connection, custom_hook("Something Else"))
            connector_thread_2.start()

            connector_thread_3 = Thread_Object("The 3", delay=0.01, logger_level=4)
            connector_thread_3.init(custom_connection, ("Message: 2"))
            connector_thread_3.init(custom_connection) # , ("Message: 2 Overwrite"))
            connector_thread_3.start()

        Variables;
            statement_quit is a manuel quit statement which will force thread to quit. No dirty quit here.
            quit_trigger is a quit trigger that calls the trigger at every loop to get quit statement which will force thread to quit. No dirty quit here.
            run_number is a number of times to run the given task or connection and after the given number of run is done, force thread to quit. No dirty quit here.
            delay is a delay time to give permission for other thread to work synchronously

            thread_ID is a unique ID of thread
            name is given name of thread by developer at initialize of object
    """

    __thread_counter = 0

    """
    # https://stackoverflow.com/questions/674304/why-is-init-always-called-after-new
    def __new__(cls):
        # Node ID
        Thread_Object.__thread_counter += 1
        return super(Thread_Object, cls).__new__(cls)
    """

    def __init__(self, name="Thread", delay=0.001, set_Deamon=True, run_number=None, quit_trigger=None, logger_level=logging.INFO, max_limit=10):
        threading.Thread.__init__(self)
        Thread_Object.__thread_counter += 1
        self.setDaemon(set_Deamon)  # don't hang on exit

        self.statement_quit = False

        #######################################
        # For Connector Thread Configurations #
        #######################################
        self.connection = None
        self.params = None
        self.is_connector = False

        self.__thread_Buffer = Structure_Buffer(max_limit=max_limit)
        self.thread_ID = Thread_Object.__thread_counter
        self.name = name

        self.delay = delay
        self.quit_trigger = quit_trigger
        self.run_number = run_number

        self.logger = logging.getLogger('[{}]'.format(self.name))
        # self.logger = logging.getLogger('Thread {} ({})'.format(self.thread_ID, self.name))
        
        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] [%(name)s] : %(message)s', 
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)

    @classmethod
    def __len__(cls):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return cls.__thread_counter

    def run(self):
        self.logger.debug('Running...')

        while True:
            # ### #### #### ### #
            # Quit Case Control #
            # ### #### #### ### #
            if self.run_number is not None:
                if self.run_number <= 0:
                    self.logger.debug('Runned task with given number of times.')
                    break
                self.run_number -= 1

            if self.statement_quit:
                self.logger.debug('Manuel quit statement is triggered at run.')
                break
            if self.quit_trigger is not None:
                if self.quit_trigger():
                    self.logger.debug('Quit trigger is triggered.')
                    break
            # ### #### #### ### #
            
            # ### #### #### ### #
            # ### TASK RUN  ### #
            # ### #### #### ### #
            if self.connection:
                self.append_Result(
                    self.connector()
                )
            else:
                if isinstance(self.params, Iterable):
                    self.append_Result(
                        self.task(
                            *self.params 
                        )
                    )
                else:
                    self.append_Result(
                        self.task(
                            self.params 
                        )
                    )
                #self.logger.debug('Task is complated.')
            # ### #### #### ### #
            sleep(self.delay)

        self.quit()
        self.logger.debug('Run is complated.')

    def init(self, params=None, connection=None, task=None):
        self.params = params
        if callable(connection):
            self.connection = connection
        if callable(task):
            self.task = task
        
    def connector(self):
        self.logger.debug('Connector is started, ID: {}'.format(self.thread_ID))
        if self.connection is not None:
            if self.params is None:
                self.connection()
            # https://www.pythontutorial.net/python-basics/python-unpack-list/
            elif type(self.params) == list or type(self.params) == tuple:
                self.connection(*self.params)
            else:
                self.connection(self.params)
        else:
            self.logger.debug('Connector connection is None, ID: {}'.format(self.thread_ID))
        self.logger.debug('Connector task is complated, ID: {}'.format(self.thread_ID))

    def task(self, params=None):
        from time import time
        secrets.SystemRandom().seed(time())
        stdo(1, "Thread Normal Task to generate Random Data: {} | ID: {}".format(secrets.SystemRandom().random(), self.thread_ID))

        """
        while True:
            stdo(1, "Thread Normal Task")
            sleep(self.delay)
            if self.statement_quit:
                self.logger.debug('Manuel quit statement is triggered at task.')
                break
            if self.quit_trigger is not None:
                if self.quit_trigger():
                    self.logger.debug('Quit trigger is triggered.')
                    break
        """

    @staticmethod
    def task_informative(format=["{}"], connections=[], connection_params=[], connections_set_buffer=[]):
        if format is None:
            connection_returns = dict()
            for index, connection in enumerate(connections):
                connection_returns[ str(connection.__qualname__) ] = connection( *connection_params[index] )

            for connection_set_buffer in connections_set_buffer:
                connection_set_buffer(connection_returns)
        else:
            formatted_information = ""
            for index, connection in enumerate(connections):
                #stdo(4, "index: {} | connection: {}".format(index, connection))
                formatted_information += format[index].format( str( connection( *connection_params[index] ) ) )
            for connection_set_buffer in connections_set_buffer:
                connection_set_buffer(formatted_information)

    def get_Result(self):
        return self.__thread_Buffer.get_API(-1)
 
    def append_Result(self, data):
        #print("self.__thread_Buffer", len(self.__thread_Buffer))
        return self.__thread_Buffer.append(data)
    
    def clear_Buffer(self):
        return self.__thread_Buffer.clear()
    
    def quit(self):
        """
        # No need to clear it since we can't run the thread again after it dies.
        # But you can overwrite run functions (like task or connector) at runtime
        if self.connection:
            self.connection = None
            self.params = None
        """
        # self.clear_Buffer()
        self.logger.debug('Quiting...')


def custom_connection(param="Nothing"):
    if callable(param):
        param()
    else:
        print("Custom Connection ", param)

def custom_hook(param="Nothing"):
    if param == "Nothing":
        print("Custom Hook ", param)
    else:
        return "OwO"

def main():
    # IMPORTANT: Threads can only be started once. So run once at a time
    connector_thread_1 = Thread_Object(
        "The 1", 
        delay=0.1, 
        logger_level=4, 
        set_Deamon=False, 
        run_number=1
    )

    """
    # Run stdo with [1, "Message: 1"] parameter(s)
    connector_thread_1.init([1, "Message: 1"], stdo)
    connector_thread_1.start()
    sleep(1)
    """

    """
    # Run custom_connection with (stdo) parameter(s)
    connector_thread_1.init((stdo), custom_connection)
    connector_thread_1.start()
    sleep(1)
    """

    """
    # Run custom_connection with no parameter(s)
    connector_thread_1.init(connection = custom_connection)
    # OR # 
    # connector_thread_1.init(None, custom_connection)
    connector_thread_1.start()
    sleep(1)
    """

    """
    # Run default task with no parameter(s)
    connector_thread_1.start()
    sleep(1)
    """

    # Run task overwrite by custom_connection with no parameter(s)
    
    connector_thread_1.task = custom_connection
    # OR 
    connector_thread_1.init(task = custom_connection)
    
    # connector_thread_1.init(("Message: 2")) # For parameter
    connector_thread_1.start()
    sleep(1)

    """
    connector_thread_2 = Thread_Object("The 2", delay=0.01, logger_level=4)
    connector_thread_2.init(("Message: 2"), stdo)
    connector_thread_2.init(custom_hook("Something Else"), custom_connection)
    connector_thread_2.start()

    connector_thread_3 = Thread_Object("The 3", delay=0.01, logger_level=4)
    connector_thread_3.init(("Message: 2"), custom_connection)
    connector_thread_3.init(connection = custom_connection) # , ("Message: 2 Overwrite"))
    connector_thread_3.start()
    """

    print("Number of threads in global len(connector_thread_1):", len(connector_thread_1))
    """
    print("Number of threads in global len(connector_thread_1):", len(connector_thread_1))
    
    # Will not work because __thread_counter is in class private scope
    connector_thread_1.__thread_counter = 0 
    Thread_Object.__thread_counter = 0

    print("Number of threads in global len(connector_thread_1):", len(connector_thread_1))
    """

    connector_thread_1.statement_quit = True
    sleep(0.1)
    # connector_thread_2.statement_quit = True
    # connector_thread_3.statement_quit = True
    

# Program Body Trigger
if __name__ == "__main__":
    main()
