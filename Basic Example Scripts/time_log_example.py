import libs
from time import sleep
from tools import time_log, time_list, TIME_FLAGS
from stdo import stdo

delay_time = 1

stdo(1, "Created.")

time_log()
stdo(1, "Time Logging Started.")

"""
for i in range(1000):
    pass
"""
sleep(delay_time)

time_log()
stdo(1, "Time Logging Ended.")

stdo(
    1, "\n\t\t{} - {} = \n\t\t{} Time Passed. \n\t\t{} Unwanted Delay.".format(
        time_list["id"][TIME_FLAGS.START], 
        time_list["id"][TIME_FLAGS.END], 
        time_list["id"][TIME_FLAGS.PASSED],
        time_list["id"][TIME_FLAGS.PASSED]-delay_time*1000
    )
)


