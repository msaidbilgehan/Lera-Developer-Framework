# https://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
# https://www.geeksforgeeks.org/how-to-find-size-of-an-object-in-python/
# https://stackoverflow.com/questions/938733/total-memory-used-by-python-process

# BUILT-IN LIBRARIES
import logging
from os import getpid
from sys import platform
from psutil import Process, cpu_percent


from structure_threading import Thread_Object
from security import safe_command

class System_Object():

    def __init__(self, logger_level=logging.INFO):
        self.logger = logging.getLogger("System_Object")

        # https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(name)s : %(message)s',
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)
        
        self.pid = getpid()
        self.platform = platform
        self.process = Process(self.pid)

        self.__thread_Dict = dict()
        

    def __del__(self):
        self.logger.error("Destroying System Object")
        for thread_id, thread in self.__thread_Dict.items():
            thread.statement_quit = True
            self.logger.info(
                "Thread {}:{} stopped.".format(
                    thread_id, thread.name
                )
            )
        self.__thread_Dict.clear()
        return 0

    def thread_print_info(self, trigger_quit=None):
        self.__thread_Dict["main_Thread"] = Thread_Object(
            name="System_Object.thread_print_info",
            delay=0.0001,
            logger_level=self.logger.getEffectiveLevel(),
            set_Deamon=True,
            run_number=None,
            quit_trigger=trigger_quit
        )
        self.__thread_Dict["main_Thread"].init(
            params=['\r'],
            task=self.print_info
        )
        self.__thread_Dict["main_Thread"].start()

    def print_info(self, end=''):
        print(f"Used CPU: {self.cpu_percent_Psutil()} | Used Memory: {self.memory_Usage_Psutil()}", end=end)
        #print("Used Memory Size:", self.memory_usage_ps(), end=end)

    def memory_Usage_Psutil(self):
        # return the memory usage in MB
        if self.platform == 'darwin':
            global getrusage, RUSAGE_SELF
            from resource import getrusage, RUSAGE_SELF
            rusage_denom = 1024.
            # ... it seems that in OSX the output is different units ...
            rusage_denom = rusage_denom * rusage_denom
            memory_usage = getrusage(RUSAGE_SELF).ru_maxrss / rusage_denom
        else:
            memory_usage = self.process.memory_info().rss / float(1024 ** 2)
        return memory_usage

    def memory_Usage_Subprocess(self):
        global Popen, PIPE
        from subprocess import Popen, PIPE
        out = safe_command.run(Popen, ['ps', 'v', '-p', str(self.pid)],
            stdout=PIPE
        ).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        return float(out[1].split()[vsz_index]) / 1024.

    def cpu_percent_Psutil(self, time_interval = 1):
        # Calling psutil.cpu_precent() for 1 second(s)
        return cpu_percent(time_interval)
