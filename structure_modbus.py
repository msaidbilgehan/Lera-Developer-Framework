

# import serial
from serial.tools.list_ports import comports
import minimalmodbus as mmRtu

from stdo import stdo
from structure_threading import Thread_Object
from structure_data import Structure_Buffer

# from tools import time_log, TIME_FLAGS, time_list, load_from_json, save_to_json, path_control
# from preprocess_image_processing import preprocess_of_Fiducial_Detection

class OBJECT_MODBUS():
    __Object_Counter = 0

    def __init__(self, name, timeout=2, delay=0.0001, max_buffer_limit=25):

        self.name = "OBJECT_MODBUS:{}".format(name)
        self.connected_com = ""
        self.logs = Structure_Buffer(max_limit=max_buffer_limit)

        self.instance_Exit_Statement = False

        self.__is_Connection_Ok = False
        self.__thread_Dict = dict()

        self.timeout = timeout
        self.delay = delay
        # self.__error_Counter_Max = error_counter_max
        
        self.__buffer_write = Structure_Buffer(max_limit=max_buffer_limit)
        self.__buffer_read = Structure_Buffer(max_limit=max_buffer_limit)
        self.readed_Addresses = dict()

        self.is_Object_Initialized = True
        OBJECT_MODBUS.__Object_Counter += 1

    @classmethod
    def __len__(cls):
        # https://www.programiz.com/python-programming/methods/built-in/len
        # https://stackoverflow.com/questions/13012159/how-create-a-len-function-on-init
        # return self.__len__()
        return cls.__Object_Counter

    def __del__(self):
        self.is_Object_Initialized = False
        
    def connect_ModBus_PLC(self, connection_com="COM0", boudrate = 9600, timeout = 1, test_address=0, expected_return=""):
        # self.connected_com = connection_com
        try:
            """The serial port object as defined by the pySerial module. Created by the constructor.

            Attributes that could be changed after initialization:

                - port (str):      Serial port name.
                    - Most often set by the constructor (see the class documentation).
                - baudrate (int):  Baudrate in Baud.
                    - Defaults to 19200.
                - parity (use PARITY_xxx constants): Parity. See the pySerial module for documentation.
                    - Defaults to :const:`serial.PARITY_NONE`.
                - bytesize (int):  Bytesize in bits.
                    - Defaults to 8.
                - stopbits (use STOPBITS_xxx constants):  The number of stopbits. See pySerial docs.
                    - Defaults to :const:`serial.STOPBITS_ONE`.
                - timeout (float): Read timeout value in seconds.
                    - Defaults to 0.05 s.
                - write_timeout (float): Write timeout value in seconds.
                    - Defaults to 2.0 s.
            """
            self.object_node = mmRtu.Instrument(connection_com, 1)
            if self.object_node.serial is not None:
                self.object_node.serial.baudrate = boudrate
                self.object_node.serial.timeout = timeout
                
                ret_value = self.read_Register(
                    register_address=test_address
                )

                if ret_value == expected_return:
                    self.__is_Connection_Ok = True
                    self.connected_com = connection_com
                    self.logs.append(
                        connection_com + ":OK"
                    )
                    stdo(
                        1,
                        f"{ret_value}, {connection_com}, ModBus Connection Successful!"
                    )
                else:
                    self.__is_Connection_Ok = False
                    self.connected_com = ""
                    stdo(
                        2, 
                        f"{ret_value}, {connection_com}, ModBus Connection Can NOT Established!"
                    )
                    self.logs.append(
                        connection_com + ":ERR:Unexpected Return"
                    )
            else:
                self.__is_Connection_Ok = False
                self.connected_com = ""
                stdo(
                    2, 
                    f"{connection_com}, ModBus Objection not Initialized!"
                )
                self.logs.append(
                    connection_com + ":ERR:ModBus Objection not Initialized"
                )
            
        except Exception as e:
            self.__is_Connection_Ok = False
            self.logs.append(
                connection_com + ":ERR:Connection Exception:{e}"
            )
            self.connected_com = ""
            stdo(
                2, 
                f"Error Occurred while Connecting to PLC over '{connection_com}' COM: {e}"
            )
            stdo(1, f"No ModBus Connection for '{connection_com}'")

            if self.object_node is not None:
                if self.object_node.serial is not None:
                    self.object_node.serial.close()

    def read_Register(self, register_address=0):
        # if self.object_node is not None and register_address > 40001:
        if self.object_node is not None:
            response = self.object_node.read_register(register_address-40001, 0)
            self.logs.append(
                self.connected_com + f":ERR:Readed Value:{response}"
            )
            return response
            #register_value = object_node.read_bits(2090, 1)
            #register_value = object_node.read_bit(4, 2)

        self.logs.append(
            self.connected_com + ":ERR:Read Failed"
        )

    def write_Register(self, register_address=0, register_value=0):
        if self.__is_Connection_Ok:
            self.logs.append(
                self.connected_com +
                f":INF:Writing '{register_value}' value to '{register_address}' Register"
            )
            try:
                self.object_node.write_register(register_address, register_value)
            except:
                self.logs.append(
                    self.connected_com +
                    f":ERR:write_Register Error, Writing Not Finished, '{register_value}' value, '{register_address}' Register"
                )
        else:
            self.logs.append(
                self.connected_com + 
                f":ERR:write_Register, Connection Not Established, '{register_value}' value, '{register_address}' Register"
            )

    def write_Bit(self, register_address=0, register_value=0):
        if self.__is_Connection_Ok:
            self.logs.append(
                self.connected_com +
                f":INF:Writing Bit '{register_value}' value to '{register_address}' Register"
            )
            try:
                self.object_node.write_bit(register_address, register_value)
            except:
                self.logs.append(
                    self.connected_com +
                    f":ERR:write_bit Error, Writing Not Finished, '{register_value}' value, '{register_address}' Register"
                )
        else:
            self.logs.append(
                self.connected_com +
                f":ERR:write_bit, Connection Not Established, '{register_value}' value, '{register_address}' Register"
            )
    
    def read_Available_COMs(self):
        iterator = sorted(comports())
        list_COMs = list()
        for n, (port, desc, hw_id) in enumerate(iterator, 1):
            list_COMs.append(port)
        return list_COMs

    def read_Thread_Start(self, trigger_quit=None, trigger_pause=None):
        self.__thread_Dict["read_Thread_Start"] = Thread_Object(
            name=self.name + ".read_Thread_Start",
            delay=0.1,
            # logger_level=None,
            set_Daemon=True,
            run_number=None,
            quit_trigger=trigger_quit
        )
        self.__thread_Dict["read_Thread_Start"].init(
            params=[trigger_pause],
            task=self.read_Thread_Job
        )
        self.__thread_Dict["read_Thread_Start"].start()

    def read_Thread_Job(self, trigger_pause):
        if self.__is_Connection_Ok and trigger_pause():
            target_address = self.__buffer_read.pop()
            if type(target_address) is int:
                self.readed_Addresses[target_address] = self.read_Register(
                    register_address=target_address
                )
            else:
                self.logs.append(
                    self.connected_com +
                    f":ERR:read_Thread_Job, type(target_address) is '{type(target_address)}', expected int."
                )

    def write_Thread_Start(self, trigger_quit=None, trigger_pause=None):
        self.__thread_Dict["write_Thread_Start"] = Thread_Object(
            name=self.name + ".write_Thread_Start",
            delay=0.1,
            # logger_level=None,
            set_Daemon=True,
            run_number=None,
            quit_trigger=trigger_quit
        )
        self.__thread_Dict["write_Thread_Start"].init(
            params=[trigger_pause],
            task=self.write_Thread_Job
        )
        self.__thread_Dict["write_Thread_Start"].start()

    def write_Thread_Job(self, trigger_pause):
        if self.__is_Connection_Ok and trigger_pause():
            buffer_pack = self.__buffer_write.pop()
            if type(buffer_pack) is list and len(buffer_pack) == 2:
                target_address, value = buffer_pack
                if type(target_address) is int and type(value) is int:
                    self.write_Register(
                        register_address=target_address,
                        register_value=value
                    )
                else:
                    self.logs.append(
                        self.connected_com +
                        f":ERR:write_Thread_Job, target_address or value is '{type(target_address), type(value)}', both expected int."
                    )
            else:
                self.logs.append(
                    self.connected_com +
                    f":ERR:write_Thread_Job, target_address type is '{type(buffer_pack)}', expected list with 2 elements."
                )
