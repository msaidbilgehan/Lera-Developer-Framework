# REF:
#  - https://docs.python.org/3/library/socketserver.html
#  - https://pymotw.com/2/socketserver/

# ##### #### #### #### ##### #
# BUILT-IN GENERAL Libraries #
# ##### #### #### #### ##### #
import socket
import logging
# import socketserver
from time import sleep

# ##### #### #### #### ##### #
#     EXTERNAL LIBRARIES     #
# ##### #### #### #### ##### #
import netifaces


# ##### #### ##### #
# Custom Libraries #
# ##### #### ##### #
from tools import get_OS
from structure_data import Structure_Buffer
from structure_threading import Thread_Object

# #### #### #### #### #
#       GLOBALS       #
# #### #### #### #### #
SERVER_PORT = 9966


### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ###


class Internet_Receiver():
    __Internet_Receiver_Object_Counter = 0
    
    def __init__(self, host="127.0.0.1", port=9956, timeout=2, set_blocking=True, parsing_format=None, regex=None, logger_level=logging.INFO, delay=0.0001, error_counter_max = 3, max_buffer_limit=10):
        self.instance_Exit_Statement = False

        self.__is_Connection_Ok = False
        self.address_Last = None
        self.data_Received = None
        self.data_Last_Received = None
        self.action_After_Receive = None
        self.action_Before_Receive = None
        self.__thread_Dict = dict()
        
        self.host = host
        self.port = port
        self.set_Blocking = set_blocking
        self.timeout = timeout
        self.parsing_format = parsing_format
        self.regex = regex
        self.__error_Counter_Max = error_counter_max
        self.delay = delay
        
        self.name = "Internet_Receiver:{}:{}".format(self.host, self.port)
        self.logger = logging.getLogger(self.name)
        
        # https://stackoverflow.com/questions/3220284/how-to-customize-from-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET) the sleep
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] [%(name)s] : %(message)s', 
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)
        
        self.__buffer_Stream = Structure_Buffer(max_limit=max_buffer_limit)
        
        self.is_Object_Initialized = True
        Internet_Receiver.__Internet_Receiver_Object_Counter += 1

    @classmethod
    def __len__(cls):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return cls.__Internet_Receiver_Object_Counter

    def __del__(self):
        self.is_Object_Initialized = False

    @staticmethod
    def data_Parsing(data, split="=", regex=None, is_decode_needed=True, decode="utf-8"):
        if is_decode_needed:
            data = data.decode(decode)

        if type(data) is str:
            if regex is not None:
                global re
                from re import split
                # https://docs.python.org/3/library/re.html
                
                # data = 'alfa=10&salsa=213'
                # regex = "[=,&]"
                # re.split("[=,&]", data)
                data = split(regex, data)
            elif split is not None:
                data = data.split(split)

        return data

    def receive_Data(self):
        address = ""
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver:
            receiver.settimeout(self.timeout)

            try:
                receiver.bind(("", self.port))
                receiver.listen(1)
                self.logger.debug("BIND is done and Listening...")

            except Exception as error:
                self.connection_Error_Handler(True)
                self.logger.error(
                    "Bind Port {} Error: {}".format(self.port, error)
                )
                receiver.close()
                sleep(self.delay)
                return address, None

            try:
                conn, address = receiver.accept()
                self.logger.info("Address is {}".format(address))

                while conn:
                    self.data_Received = conn.recv(1024)

                    # FOR TEST
                    # self.data_Received = b"is_snapshot:1"

                    if len(self.data_Received) != 0:
                        self.data_Last_Received = self.data_Received
                        self.logger.info("Received Data is: {}".format(
                                str(self.data_Last_Received)
                            )
                        )

                        self.data_Last_Received = self.data_Parsing(
                            data=self.data_Last_Received, 
                            split=self.parsing_format,
                            is_decode_needed=True, 
                            decode="utf-8"
                        )
                        self.buffer_Append(
                            data=self.data_Last_Received,
                            lock_until_done=False
                        )
                        self.connection_Error_Handler(False)

                    else:
                        self.logger.info("Data is waiting...")
                        break
                    sleep(self.delay)

            except socket.timeout as to:
                self.logger.debug("Timeout Error: {}".format(to))
                receiver.close()
                sleep(self.delay)
                return address, None

            except Exception as error:
                self.connection_Error_Handler(True)
                self.logger.error("Error: {}".format(error))
                receiver.close()
                sleep(self.delay)
                return address, None
            sleep(self.delay)
            
            return address, self.data_Last_Received

    def connection_Error_Handler(self, error):
        if error:
            self.__error_Counter_Max += 1 
            if self.__error_Counter_Max >= self.__error_Counter_Max:
                self.__is_Connection_Ok = False
        else:
            self.__error_Counter_Max = 0
            self.__is_Connection_Ok = True

    ### ### ### ### ### ### ###
    ### ### BUFFER APIs ### ###
    ### ### ### ### ### ### ###

    def buffer_Pop(self, index=0):
        return self.__buffer_Stream.pop(index)
    
    def buffer_Append(self, data, lock_until_done=False, delay=0.00001):
        while lock_until_done:
            if self.__buffer_Stream.is_Buffer_Full():
                sleep(delay)
            else:
                self.__buffer_Stream.append(data)
                break
        else:
            self.__buffer_Stream.append(data)

    def buffer_Clear(self):
        self.logger.debug("buffer_Clear Buffer is cleaning...")
        return self.__buffer_Stream.clear()

    def buffer_Get(self, index=0):
        return self.__buffer_Stream.get_API(index)
    
    def buffer_Get_Len(self):
        return len(self.__buffer_Stream)

    def buffer_Get_Bulk(self, start=0, end=-1):
        return self.__buffer_Stream.get_Bulk(start, end)

    def buffer_Overwrite(self, buffer):
        self.__buffer_Stream.clear()
        self.__buffer_Stream = buffer
        return

    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###

    ### ### ### ### ### ### ###
    ### ### THREAD APIs ### ###
    ### ### ### ### ### ### ###
    
    def start(self, trigger_pause=None, trigger_quit=None, number_of_receive=-1, delay=0.001, lock_until_done=False):
        if self.get_Is_Object_Initialized():
            self.__thread_Dict[self.name] = Thread_Object(
                name=self.name + ".start",
                delay=0.0001,
                logger_level=self.logger.getEffectiveLevel(),
                set_Deamon=True,
                run_number=None,
                quit_trigger=trigger_quit
            )
            self.__thread_Dict[self.name].init(
                params=[
                    trigger_pause,
                    trigger_quit,
                    number_of_receive,
                    delay,
                    lock_until_done
                ],
                task=self.start_Thread
            )
            self.__thread_Dict[self.name].start()

            return self.__thread_Dict[self.name]
        else:
            return None

    def start_Thread(self, trigger_pause=None, trigger_quit=None, number_of_receive=-1, delay=0.001, lock_until_done=False):
        if trigger_pause is None or not callable(trigger_pause):
            trigger_pause = self.trigger_pause

        pause_Delay = 1
        data = None
        address = ""
        while number_of_receive != 0:
            if trigger_pause():
                self.logger.debug(
                    "start_Thread get trigger_pause. Waiting {} sec.".format(
                        pause_Delay
                    )
                )
                sleep(pause_Delay)
            else:
                self.action_Before_Receive(address, data) if callable(self.action_Before_Receive) \
                    else None
                address, data = self.receive_Data()
                self.action_After_Receive(address, data) if callable(self.action_After_Receive) \
                    else None
                    
                if data is None:
                    self.logger.debug(
                        "UNSuccessfull receive_Data action!"
                    )
                else:
                    self.buffer_Append(
                        data, 
                        lock_until_done=lock_until_done
                    )

                    if number_of_receive > 0:
                        number_of_receive -= 1
                """
                # Buffer is clear, just pass
                else:
                    pass
                """

            if trigger_quit is not None:
                if trigger_quit():
                    self.logger.debug(
                        "start_Thread quit trigger activated! Quiting..."
                    )
                    self.quit()
                    return 0  # break # Lets make it clear at quit statement
            elif self.instance_Exit_Statement:
                self.logger.debug(
                    "{} Internet Sender Instance start_Thread Exit Statement activated! Quiting...".format(
                        self.name
                    )
                )
                return 0  # break # Lets make it clear at quit statement
            sleep(delay)
        return 0

    def trigger_pause(self):
        return False

    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###

    def quit(self):
        self.logger.info(
            "Exiting from {} ...".format(
                self.name
            )
        )

        self.instance_Exit_Statement = True
        self.logger.info(
            "Buffer cleaned. {} number of buffer element removed.".format(
                len(self.buffer_Clear())
            )
        )

        for thread_id, thread in self.__thread_Dict.items():
            thread.statement_quit = True
            self.logger.info("Thread {}:{} stopped.".format(
                thread_id,
                thread.name
            )
            )
        self.__thread_Dict.clear()

        self.__del__()

    def run(self):
        while True:
            self.logger.info("Network Interface is not active. Please activate to communication.")
            if self.is_Interface_Active():
                self.receive_Data()
            else:
                self.logger.info("Network Interface is not active. Please activate to communication.")
            sleep(0.1)

    def is_Interface_Active(self):
        platform_name = get_OS().lower()
        try:
            if platform_name == "windows":
                # TODO: Add platform specific control
                return True
            elif platform_name == "linux":
                if len(netifaces.ifaddresses('eth0')) > 1:
                    return True
                else:
                    return True
            else:
                return True

        except Exception as error:
            self.logger.error("Network Interface Control Error: {}".format(error))

        return False
    
    def get_Information(self):
        dict_information = dict()
        dict_information["host"] = self.host
        dict_information["port"] = self.port
        dict_information["is_connection_ok"] = int(self.__is_Connection_Ok)
        dict_information["set_blocking"] = self.set_Blocking
        dict_information["timeout"] = self.timeout
        dict_information["address_Last"] = self.address_Last
        dict_information["data_Received"] = self.data_Received
        dict_information["data_Last_Received"] = self.data_Last_Received
        dict_information["buffer_Length"] = self.buffer_Get_Len()
        dict_information["buffer_Max_Length"] = self.__buffer_Stream.max_limit
        return dict_information

    def get_Is_Object_Initialized(self):
        return self.is_Object_Initialized


class Internet_Sender():
    __Internet_Sender_Object_Counter = 0
    def __init__(self, host, port=9954, timeout=2, set_blocking=True, parsing_format=None, regex=None, logger_level=logging.INFO, delay=0.0001, error_counter_max = 3, max_buffer_limit=10):
        
        self.instance_Exit_Statement = False
        
        self.__is_Connection_Ok = False
        self.address_Last = None
        self.data_Last_Sended = None
        self.action_After_Send = None
        self.action_Before_Send = None
        self.__thread_Dict = dict()

        self.host = host
        self.port = port
        self.set_Blocking = set_blocking
        self.timeout = timeout
        self.delay = delay
        self.__error_Counter_Max = error_counter_max

        self.parsing_format = parsing_format
        self.regex = regex
        
        self.name = "Internet_Sender:{}:{}".format(self.host, self.port)
        self.logger = logging.getLogger(self.name)

        self.__buffer_Stream = Structure_Buffer(max_limit=max_buffer_limit)
        #self.__buffer_Stream.append("Buffered Data Initialize")

        # https://stackoverflow.com/questions/3220284/how-to-customize-from-time-format-for-python-logging
        # self.logger.setLevel(logging.NOTSET) the sleep
        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] [%(name)s] : %(message)s', 
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logger_level)
        
        self.is_Object_Initialized = True
        Internet_Sender.__Internet_Sender_Object_Counter += 1

    @classmethod
    def __len__(cls):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return cls.__Internet_Sender_Object_Counter

    def __del__(self):
        self.is_Object_Initialized = False

    @staticmethod
    def data_Parsing(data, split="=", regex=None, is_decode_needed=True, decode="utf-8"):
        if is_decode_needed:
            data = data.decode(decode)

        if type(data) is str:
            if regex is not None:
                global re
                from re import split
                # https://docs.python.org/3/library/re.html
                
                # data = 'alfa=10&salsa=213'
                # regex = "[=,&]"
                # re.split("[=,&]", data)
                data = split(regex, data)
            elif split is not None:
                data = data.split(split)

        return data

    @staticmethod
    def data_Packaging(data_pack, index_pack, split="=", ends_with="\n"): #, is_decode_needed=False, decode="utf-8"):
        # TODO: package data as split model
        package = ""
        for current_index in range(max([len(data_pack), len(index_pack)], key=len)):
            index = index_pack[current_index] if len(index_pack) > current_index else "None" # "_Index"
            data = data_pack[current_index] if len(data_pack) > current_index else "None" # "_Data"

            package += index + split + data + ends_with

            """
            if type(data) is str:
                if regex is not None:
                    global re
                    from re import split
                    # https://docs.python.org/3/library/re.html

                    # data = 'alfa=10&salsa=213'
                    # regex = "[=,&]"
                    # re.split("[=,&]", data)
                    data = split(regex, data)
                elif split is not None:
                    data = data.split(split)
            """

        return package

    def start(self, trigger_pause=None, trigger_quit=None, number_of_send=-1, delay=0.001):
        if self.get_Is_Object_Initialized():
            self.__thread_Dict[self.name] = Thread_Object(
                name=self.name + ".start",
                delay=0.0001,
                logger_level=self.logger.getEffectiveLevel(),
                set_Deamon=True,
                run_number=None,
                quit_trigger=trigger_quit
            )
            self.__thread_Dict[self.name].init(
                params=[
                    trigger_pause,
                    trigger_quit,
                    number_of_send,
                    delay
                ],
                task=self.start_Thread
            )
            self.__thread_Dict[self.name].start()

            return self.__thread_Dict[self.name]
        else:
            return None

    def start_Thread(self, trigger_pause=None, trigger_quit=None, number_of_send=-1, delay=0.001):

        if trigger_pause is None or not callable(trigger_pause):
            trigger_pause = self.trigger_pause

        pause_Delay = 1
        while number_of_send != 0:
            if trigger_pause():
                self.logger.debug(
                    "start_Thread get trigger_pause. Waiting {} sec.".format(pause_Delay)
                )
                sleep(pause_Delay)
            else:
                if len(self.__buffer_Stream) > 0:
                    # data = self.buffer_Pop()
                    data = self.buffer_Get(index=0)

                    is_Successfull = self.send_Data(
                        data
                    )

                    if is_Successfull:
                        self.buffer_Pop()
                        if number_of_send > 0:
                            number_of_send -= 1
                    else:
                        self.logger.debug(
                            "UNSuccessfull send_Data action!"
                        )
                        # self.__buffer_Stream.insert(0, data)
                """
                # Buffer is clear, just pass
                else:
                    pass
                """

            if trigger_quit is not None:
                if trigger_quit():
                    self.logger.debug(
                        "start_Thread quit trigger activated! Quiting..."
                    )
                    self.quit()
                    return 0  # break # Lets make it clear at quit statement
            elif self.instance_Exit_Statement:
                self.logger.debug(
                    "{} Internet Sender Instance start_Thread Exit Statement activated! Quiting...".format(
                        self.name
                    )
                )
                return 0  # break # Lets make it clear at quit statement
            sleep(delay)
        return 0

    def quit(self):
        self.logger.info(
            "Exiting from {} ...".format(
                self.name
            )
        )

        self.instance_Exit_Statement = True
        self.logger.info(
            "Buffer cleaned. {} number of buffer element removed.".format(
                len(self.buffer_Clear())
            )
        )

        for thread_id, thread in self.__thread_Dict.items():
            thread.statement_quit = True
            self.logger.info("Thread {}:{} stopped.".format(
                    thread_id, 
                    thread.name
                )
            )
        self.__thread_Dict.clear()

        self.__del__()

    def send_Data(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sender:
            try:
                sender.settimeout(self.timeout)
                sender.connect((self.host, self.port))
                
                if type(data) is str:
                    data = bytes(data, "utf-8")
                else:
                    self.logger.error("Unknown data type for '{}' is {}. Trying to convert to String -> Byte".format(data, type(data)))
                    data = bytes(str(data), "utf-8")

                self.action_Before_Send(data) if callable(self.action_Before_Send) \
                    else None
                
                sender.sendall(data)
                
                self.action_After_Send(data) if callable(self.action_After_Send) \
                    else None
                    
                sender.close()
                
                self.data_Last_Sended = data
                self.connection_Error_Handler(False)
                
                sleep(self.delay)
                    
            except socket.timeout as to:
                self.logger.debug("Timeout Error: {}".format(to))
                sender.close()
                sleep(self.delay)
                return False

            except Exception as error:
                self.logger.error("Error: {}".format(error))
                self.connection_Error_Handler(True)
                sender.close()
                sleep(self.delay)
                return False
        return True

    def send_Snapshot_Data(self, data="TEST"):
        # Ping to CONNECTION_IP
        # https://www.kite.com/python/answers/how-to-ping-an-ip-address-in-python
        if self.is_Interface_Active():
            if callable(data):
                data = data()

            self.send_Data(data)
        else:
            self.logger.warning(
                "Network Interface is not active. Please activate to communication."
            )
            
    def connection_Error_Handler(self, error):
        if error:
            self.__error_Counter_Max += 1 
            if self.__error_Counter_Max >= self.__error_Counter_Max:
                self.__is_Connection_Ok = False
        else:
            self.__error_Counter_Max = 0
            self.__is_Connection_Ok = True

    ### ### ### ### ### ### ###
    ### ### BUFFER APIs ### ###
    ### ### ### ### ### ### ###

    def buffer_Pop(self, index=0):
        return self.__buffer_Stream.pop(index)

    def buffer_Append(self, data, lock_until_done=False, delay=0.00001):
        while lock_until_done: 
            # if len(self.__buffer_Stream) == self.__buffer_Stream.max_limit:
            if self.__buffer_Stream.is_Buffer_Full():
                sleep(delay)
            else:
                self.__buffer_Stream.append(data)
                break
        else:
            self.__buffer_Stream.append(data)

    def buffer_Clear(self):
        self.logger.debug("buffer_Clear Buffer is cleaning...")
        return self.__buffer_Stream.clear()

    def buffer_Get(self, index=0):
        return self.__buffer_Stream.get_API(index) 
    
    def buffer_Get_Bulk(self, start=0, end=-1):
        return self.__buffer_Stream.get_Bulk(start, end)

    def buffer_Get_Len(self):
        return len(self.__buffer_Stream)

    def buffer_Overwrite(self, buffer):
        self.__buffer_Stream.clear()
        self.__buffer_Stream = buffer
        return

    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###
    ### ### ### ### ### ### ###

    def is_Interface_Active(self):
        platform_name = get_OS().lower()
        try:
            if platform_name == "windows":
                # TODO: Add platform specific control
                return True
            elif platform_name == "linux":
                if len(netifaces.ifaddresses('eth0')) > 1:
                    return True
                else:
                    return True
            else:
                return True

        except Exception as error:
            self.logger.error("Network Interface Control Error: {}".format(error))

        return False

    def get_Information(self):
        dict_information = dict()
        dict_information["host"] = self.host
        dict_information["port"] = self.port
        dict_information["is_connection_ok"] = int(self.__is_Connection_Ok)
        dict_information["set_blocking"] = self.set_Blocking
        dict_information["timeout"] = self.timeout
        dict_information["data_Last_Sended"] = self.data_Last_Sended
        dict_information["buffer_Length"] = self.buffer_Get_Len()
        dict_information["buffer_Max_Length"] = self.__buffer_Stream.max_limit
        return dict_information

    def get_Is_Object_Initialized(self):
        return self.is_Object_Initialized
    
    def trigger_pause(self):
        return False



if __name__ == '__main__':
    from stdo import stdo
    
    stdo(1, "================")
    stdo(1, "Starting Main...")
    stdo(1, "================")
    
    max_try_time = 2
    delay = 0.0000001
    
    ip_receiver = "127.0.0.1"
    port_receiver = 3344
    
    # ip_sender = "127.0.0.1"
    # port_sender = 3344
    ip_sender = "192.168.22.22"
    port_sender = 8888
    
    # parsing_format = "\n\|--- *"
    parsing_format = "\[|]|\n"
    # parsing_format = None
    
    timeout = 2
    max_buffer_limit = 10
    error_counter_max = 3
    
    set_blocking = False
    logger_level = logging.INFO # DEBUG
    disable_Logger = True #False
    
    stdo(1, "Initialized Variables.")

    def data_Parsing_Custom(data):
        return data
    
    def get_Random_Data(number_from=30, number_to=0, number_step=-1):
        return list(
            (
                range(
                    number_from, 
                    number_to, 
                    number_step
                )
            )
        )
    
    blind_spot_all = dict()
    
    for try_time in range(1, max_try_time + 1):
        stdo(1, "Try Time: {}".format(try_time))

        ### ### ### ### ### ### ###
        ### Receiver Initialize ###
        ### ### ### ### ### ### ###

        Internet_Receiver_1 = Internet_Receiver(
            host=ip_receiver,
            port=port_receiver,
            timeout=timeout,
            set_blocking=set_blocking,
            logger_level=logger_level,
            parsing_format=parsing_format,
            delay=delay,
            error_counter_max=error_counter_max,
            max_buffer_limit=max_buffer_limit
        )
        Internet_Receiver_1.logger.disabled = disable_Logger

        # Internet_Receiver_1.data_Parsing = data_Parsing_Custom
        # Internet_Receiver_1.buffer_Overwrite()

        ### ### ### ### ### ### ###
        ### ### ### ### ### ### ###
        ### ### ### ### ### ### ###

        ### ### ### ### ### ### ###
        ###  Sender Initialize  ###
        ### ### ### ### ### ### ###

        Internet_Sender_1 = Internet_Sender(
            host=ip_sender,
            port=port_sender,
            timeout=timeout,
            set_blocking=set_blocking,
            logger_level=logger_level,
            delay=delay,
            error_counter_max=error_counter_max,
            max_buffer_limit=max_buffer_limit
        )
        Internet_Sender_1.logger.disabled = disable_Logger
        # Internet_Sender_1.buffer_Overwrite()
        
        # Internet_Sender_1.buffer_Append(data, lock_until_done=True)

        ### ### ### ### ### ### ###
        ### ### ### ### ### ### ###
        ### ### ### ### ### ### ###

        stdo(1, "Initialized Objects.")

        # Fill the Buffer of Sender
        random_data = get_Random_Data(
            number_from=max_buffer_limit, 
            number_to=0, 
            number_step=-1
        )
        stdo(1, "Initialized Random Data: {}".format(random_data))

        for data in random_data:
            Internet_Sender_1.buffer_Append(data, lock_until_done=True)
            
        stdo(
            1, 
            "Random Data Filled to Sender Buffer: {}".format(
                Internet_Sender_1.buffer_Get_Len()
            )
        )
        
        # Start Internet Object
        Internet_Receiver_1.start()
        Internet_Sender_1.start()
        stdo(1, "Internet Objects are started")
        
        # Start Receiving
        """
        while True:
            received_data = Internet_Receiver_1.buffer_Pop() # buffer_Get()

            if received_data is None:
                continue

            stdo(
                1, 
                "received_data: {}".format(
                    received_data
                )
            )
        """
        # if Internet_Sender_1.buffer_Get_Len() == 0:
        
        # Wait Sender Finish Its Buffer
        stdo(1, "Waiting to finish Internet Sender Object to It's Buffer")
        while Internet_Sender_1.buffer_Get_Len() != 0:
            pass
        else:
            stdo(
                1, 
                "Sender Buffer is Cleared: {}".format(
                    Internet_Sender_1.buffer_Get_Len()
                )
            )
        
        stdo(1, "Waiting to finish Internet Receiver Object to It's Buffer")
        received_all = list()
        while Internet_Receiver_1.buffer_Get_Len() != 0:
            received_all.append(
                int(
                    Internet_Receiver_1.buffer_Pop()
                )
            )
        Internet_Receiver_1.buffer_Clear()
        Internet_Sender_1.buffer_Clear()
        
        Internet_Receiver_1.quit()
        Internet_Sender_1.quit()
        
        stdo(1, "\t\t Sended Data:{}".format(random_data))
        stdo(1, "\t\t Received Data:{}".format(received_all))
        stdo(
            1, 
            "Blind Spot [{}]:".format(
                str(len(set(random_data)-set(received_all)))
            )
        )
        stdo(1, "\t\t |- {}".format(set(random_data)-set(received_all)))
        
        # while True: pass

        blind_spot_all[
            "{}. Try: Blind Spot is ({})".format(
                try_time, 
                len(set(random_data)-set(received_all))
            )
        ] = set(random_data)-set(received_all)
            

    stdo(1, "===============================")
    stdo(1, "============RESULTS============")
    stdo(1, "===============================")

    for key, value in blind_spot_all.items():
        stdo(1, "{} -> {}".format(key, value))

    stdo(1, "===============================")
    stdo(1, "===============================")
    stdo(1, "===============================")
