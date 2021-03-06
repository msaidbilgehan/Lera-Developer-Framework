# BUILT-IN Threading Libraries
from threading import Lock
from enum import Enum
import logging

# Custom Libraries
from stdo import stdo  
from tools import load_from_json, save_to_json, path_control
#from structure_threading import Thread_Object


""" # https://realpython.com/intro-to-python-threading/#producer-consumer-using-queue
import queue
class Structure_Buffer(queue.Queue):
    def __init__(self, maxsize = 10):
        super().__init__(maxsize = maxsize)

    def get_message(self, name):
        value = self.get()
        return value

    def set_message(self, value, name):
        self.put(value)
"""


class STORE_WAY_FLAG(Enum):
    LAST_IN_LAST_OUT = 0
    LAST_IN_FIRST_OUT = 1


class Structure_Buffer():
    __number_of_instance = 0
    ### ### ### ### ### ###
    ### ## BUILT-IN ### ###
    ### ### ### ### ### ###

    '''
    This code is a buffer that stores the data in a list.
    The code has two modes of operation, which are defined by STORE_WAY_FLAG.LAST_IN_LAST_OUT and STORE_WAY_FLAG.FIRST IN FIRST OUT (FIFO).
    When storing new data, it will check if the number of stored data exceeds max limit or not. If yes, then it will remove some old data to make space for new one.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def __init__(self, max_limit=25, store_way=STORE_WAY_FLAG.LAST_IN_LAST_OUT, logger_level=logging.INFO, name=""):
        #self.number_of_data = 0
        Structure_Buffer.__number_of_instance += 1
        self.instance_id = Structure_Buffer.__number_of_instance
        
        self.__buffer = list()
        self.__lock_of_buffer = Lock()
        self.max_limit = max_limit
        self.store_way = store_way

        self.name = "Structure_Buffer-{}{}".format(
            self.instance_id, "-" + name if name != "" else name
        )
        
        self.logger = logging.getLogger(self.name)
        # DEBUG
        #self.buffer_debug()
        # ___DEBUG___
        
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

    def buffer_debug(self):
        import threading
        self.__thread_info = threading.Thread(
            target=self.info,
            daemon=True
        )
        self.__thread_info.start()

    def info(self):
        from time import sleep
        # from gc import collect
        while True:
            # collect()
            #print("{}:len - {}".format(self.name, len(self)))
            print("{}: {}".format(self.instance_id, len(self)))
            sleep(1)

    '''
    This code is returning the number of data in the list.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def __len__(self):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return len(self.__buffer)
        # return len(self)

    def __enter__(self):
        self.buffer_Lock(blocking=True, timeout=-1)

    '''
    This code is creating a new logger.
    Then, it is setting the level of the logger to DEBUG. Then, it creates a 
    new buffer and sets its size to 10MB. Finally, it locks the buffer so that 
    it can be written to by other threads while this thread is writing to it.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def __exit__(self, type, value, traceback):
        if self.logger is None:
            stdo(
                1, 
                "type:{} | value:{} | traceback:{}".format(
                    type, 
                    value, 
                    traceback
                )
            )
        else:
            self.logger.info(
                "type:{} | value:{} | traceback:{}".format(
                    type,
                    value,
                    traceback
                )
            )

        self.buffer_Unlock()

    ### ### ### ### ### ###
    ### ### ### ### ### ###
    ### ### ### ### ### ###

    '''
    This code is inserting data into the buffer.
    The code block is used to lock the buffer so that no other thread can access it at the same time.
    After adding data, we increment number_of_data by 1 and unlock the buffer.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def insert(self, index, data):
        with self.__lock_of_buffer:
            self.__buffer.insert(index, data)
            #self.number_of_data += 1

    '''
    This code is inserting the data into the buffer
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def nts_Insert(self, index, data):
        self.__buffer.insert(index, data)
        #self.number_of_data += 1
        #print("{}:len - {}".format(self.name, len(self)))

    '''
    This code is a buffer.
    It has two parts, the first part is for storing data and the second part is for retrieving data from it.
    The first part of this code stores data in the buffer while the second part retrieves them from it.
    If you want to know more about how these two parts work, please refer to [this article]() or [this video]().
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def append(self, data):
        with self.__lock_of_buffer:
            if self.store_way is None or self.max_limit == 0 or len(self) < self.max_limit:
                self.__buffer.append(data)
                #self.number_of_data += 1
            elif self.is_Buffer_Full():
                if self.store_way is STORE_WAY_FLAG.LAST_IN_LAST_OUT:
                    self.__buffer.pop(0)
                    self.__buffer.append(data)
                elif self.store_way is STORE_WAY_FLAG.LAST_IN_FIRST_OUT:
                    self.__buffer.pop(-1)
                    self.__buffer.append(data)
                else:
                    stdo(3, "Undefined Store Way Flag ::: {}".format(self.store_way))
            else:
                stdo(3, "Undefined Append Situation ::: store_way:{} | max_limit:{} | number_of_data:{}".format(
                    self.store_way, self.max_limit, len(self)))
        #print("{}:len - {}".format(self.name, len(self)))

    '''
    This code is doing the following things.
        1. Check if the store_way is None or max_limit == 0 or number_of_data < max_limit, then append data to buffer and increase number of data by one.
        2. If not, check if number of data >= max limit, then pop out first element in buffer and append new element to it (last in last out).
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def nts_Append(self, data):
        #print("nts_Append", len(self), " | ", len(self.__buffer))
        if self.store_way is None or self.max_limit == 0 or len(self) < self.max_limit:
            self.__buffer.append(data)
            #self.number_of_data += 1
        elif self.is_Buffer_Full():
            if self.store_way is STORE_WAY_FLAG.LAST_IN_LAST_OUT:
                self.__buffer.pop(0)
                self.__buffer.append(data)
            elif self.store_way is STORE_WAY_FLAG.LAST_IN_FIRST_OUT:
                self.__buffer.pop(-1)
                self.__buffer.append(data)
            else:
                stdo(3, "Undefined Store Way Flag ::: {}".format(self.store_way))
        else:
            stdo(3, "Undefined NTS Append Situation ::: store_way:{} | max_limit:{} | number_of_data:{}".format(
                self.store_way, self.max_limit, len(self)))

    def get(self, index):
        return self.__buffer[index]

    '''
    This code is getting the API from the buffer. If there are no APIs in the buffer, it will return None.
    If there are APIs in the buffer and index is less than number of data, then it returns that API at that index.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def get_API(self, index):
        return self.__buffer[index] if 0 < len(self.__buffer) and index < len(self) else None

    def get_Last(self):
        return self.get_API(-1)

    def get_First(self):
        return self.get_API(0)

    def get_Bulk(self, start=0, end=0):
        return_responce = None
        if 0 < len(self.__buffer) and start < len(self.__buffer) and start > -1 and end > -1 and end < len(self.__buffer):
            if start == 0 and end == 0:
                return_responce = self.__buffer[:] 
            else:
                return_responce = self.__buffer[start:end]
        # print(f"bulk_buffer ({len(self.__buffer)}={start, end}): {return_responce}")
        return return_responce

    '''
    This code is removing the first element of the list and returning it.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def pop(self, index = 0):
        if len(self) > 0:
            return self.remove_From_Buffer(index)[0][0]

    '''
    This code is removing the first element of the list and returning it
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def nts_Pop(self, index = 0):
        if len(self) > 0:
            return self.nts_Remove_From_Buffer(index)[0][0]

    '''
    This code is clearing the buffer.
    If you want to clear the buffer fast, then set is_fast=True and it will return a list of removed data.
    Otherwise, if you want to clear all data in the buffer, then set is_fast=False and it will return a list of removed data.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def clear(self, is_fast=False):
        if is_fast:
            self.__buffer.clear()
            #self.number_of_data = 0
            return list()
        else:
            removed = list()
            for _ in range(len(self.__buffer)):
                removed.append(self.remove_From_Buffer(0)[0])
            #self.number_of_data = 0
            return removed

    '''
    This code is trying to acquire the lock of buffer.
    If it can\'t, then it will return False and exit the function.
    Otherwise, it will set the lock to locked and return True.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def buffer_Lock(self, blocking=True, timeout=15):
        """
            When invoked with the blocking argument set to True (the default), 
            block until the lock is unlocked, then set it to locked and return True.
            
            When invoked with the blocking argument set to False, do not block. 
            If a call with blocking set to True would block, 
            return False immediately; otherwise, set the lock to locked and return True.

            When invoked with the floating-point timeout argument set to a positive value, 
            block for at most the number of seconds specified by timeout and as long as 
            the lock cannot be acquired. A timeout argument of -1 specifies an unbounded wait. 
            It is forbidden to specify a timeout when blocking is false.

            The return value is True if the lock is acquired successfully, 
            False if not (for example if the timeout expired).

            IMPORTANT: While manually using lock, timeout need to be set to a time. 
            If timeout is infinite (-1), and you forget to unlock at block statement, 
            buffer will be unreachable.
        """
        self.__lock_of_buffer.acquire(blocking=blocking, timeout=timeout)

    '''
    This code is trying to acquire a lock on the buffer. If it can\'t, then it will wait until the lock becomes available.
    Once the lock is acquired, we are going to check if there\'s any data in the buffer and if so, we\'re going to print out that data.
    After printing out that data, we\'re releasing our hold on this particular thread by calling release() method of Lock object which was created when 
    the Buffer class was instantiated.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def buffer_Unlock(self):
        """
            Release a lock. 
            This can be called from any thread, not only the thread which has acquired the lock.

            When the lock is locked, reset it to unlocked, and return. If any other threads are 
            blocked waiting for the lock to become unlocked, allow exactly one of them to proceed.

            When invoked on an unlocked lock, a RuntimeError is raised.

            There is no return value.
        """
        self.__lock_of_buffer.release()

    '''
    This code is removing data from the buffer.
    If index is not None, it will remove the element at that index.
    If data_to_search is not None, it will search for all elements with that value and remove them if found. If only first one is wanted to be removed, set `is_only_first=True`.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def remove_From_Buffer(self, index=None, data_to_search=None, is_only_first=True):
        with self.__lock_of_buffer:
            removed_data = list()
            removed_index = list()
            if len(self) > 0:

                if index is not None:
                    if type(index) is int:
                        removed_data.append(self.__buffer.pop(index))
                        removed_index.append(index)
                    elif type(index) is list:
                        for i in range(len(index)):
                            removed_data.append(self.__buffer.pop(i))
                            removed_index.append(index)

                elif data_to_search is not None:

                    if not is_only_first:
                        index_list = self.search(data_to_search, is_only_first=False)
                        if len(index_list) > 0:
                            removed_data.append(self.__buffer.pop(index_list[0]))
                            removed_index.append(index)
                        
                        # TODO: WTF? Add search algorithm and connect to remove_From_Buffer!



                    for i in range(len(self.__buffer)):
                        if data_to_search == self.__buffer[i]:
                            removed_data.append(self.__buffer.pop(i))
                            if not is_only_first:
                                self.remove_From_Buffer(
                                    index=None,
                                    data_to_search=data_to_search,
                                    is_only_first=True
                                )
                                break
                #self.number_of_data -= len(removed_data)
            else:
                removed_data.append(-1)
                removed_index.append(-1)

            return removed_data, removed_index

    '''
    This code is removing the data from the buffer.
    If index is given, it will remove that specific element in the buffer.
    If data_to_search is given, it will search for that specific element and remove all of them if found.
    The removed elements are returned as a list with their indexes (if any).
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def nts_Remove_From_Buffer(self, index=None, data_to_search=None, is_only_first=True):
        removed_data = list()
        removed_index = list()
        if len(self) > 0:

            if index is not None:
                if type(index) is int:
                    removed_data.append(self.__buffer.pop(index))
                    removed_index.append(index)
                elif type(index) is list:
                    for i in range(len(index)):
                        removed_data.append(self.__buffer.pop(i))
                        removed_index.append(index)

            elif data_to_search is not None:

                if not is_only_first:
                    index_list = self.search(data_to_search, is_only_first=False)
                    if len(index_list) > 0:
                        removed_data.append(self.__buffer.pop(index_list[0]))
                        removed_index.append(index)
                    
                    # TODO: WTF? Add search algorithm and connect to remove_From_Buffer!



                for i in range(len(self.__buffer)):
                    if data_to_search == self.__buffer[i]:
                        removed_data.append(self.__buffer.pop(i))
                        if not is_only_first:
                            self.nts_Remove_From_Buffer(
                                index=None,
                                data_to_search=data_to_search,
                                is_only_first=True
                            )
                            break
            #self.number_of_data -= len(removed_data)
        else:
            removed_data.append(-1)
            removed_index.append(-1)

        return removed_data, removed_index
    
    '''
    This code is searching for the data_to_search in the buffer. If it finds it, then it will add that index to a list and return that list.
    If you want to search for multiple instances of data_to_search, set is_only_first=False.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def search(self, data_to_search, is_only_first=True):
        # TODO: Search and return index and data
        index_list = list()
        
        # for i in range(len(self.__buffer)):
        for i, data_current in enumerate(self.__buffer):
            if data_current == data_to_search:
                index_list.append(i)
                if is_only_first:
                    break

        return index_list

    '''
    This code is loading the buffer from a file.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def buffer_From_File(self, path):
        is_file, _ = path_control(path, is_file=True, is_directory=True)

        if not is_file:
            self.logger.error("Path is not a file path!")
            return -1
        
        self.logger.debug("Buffer Loaded from path: {}".format(path))
        json_file = load_from_json(path)
        self.__buffer = json_file if json_file is not None else self.__buffer
        
    '''
    This code is saving the buffer to a file.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def buffer_To_File(self, path):
        is_file, _ = path_control(path, is_file=True, is_directory=True)

        if not is_file:
            self.logger.error("Path is not a file path!")
            return -1

        self.logger.debug("Buffer Saved to path: {}".format(path))
        return save_to_json(path, self.__buffer)

    '''
    This code is setting the max_limit to a new value. It\'s also checking if the length of the buffer is greater than
    the max limit, and then removing items from the buffer until it\'s back at its maximum size.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def update_Buffer_Size(self, max_limit):
        self.max_limit = max_limit
        #while len(self) > self.max_limit:
        while self.is_Buffer_Full():
            self.pop()

    '''
    This code is checking if the number of data in buffer is equal to max limit.
    If it\'s true, then return True else False.
    - generated by stenography autopilot [ 🚗👩‍✈️ ]
    '''
    def is_Buffer_Full(self):
        return True if len(self) >= self.max_limit \
            else False