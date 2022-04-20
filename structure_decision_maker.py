# BUILT-IN LIBRARIES
#from enum import Enum
import logging

# EXTERNAL LIBRARIES
# None

# CUSTOM LIBRARIES
# from stdo import stdo

# TEST LIBRARIES
# import random
# from time import time


class Decision_Maker():

    name = None
    logger = None
    formatted_information = None
    list_decision_buffer = None
    
    decision_history = list()
    dict_last_decision = dict()
    decision_result_data = dict()
    

    def __init__(self, name="Decision Maker", logger_level=logging.INFO):

        self.name = name

        self.logger = logging.getLogger('[{}]'.format(self.name))
        # self.logger = logging.getLogger('Thread {} ({})'.format(self.thread_ID, self.name))

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


    def set_decision_buffer(self, data):
        self.list_decision_buffer = data

    # TODO: Remove
    @staticmethod
    def send_all_connections(decision_data, pass_action_data, connection_decision_result_action, is_format=True, format="{}: {}"):
        # OR #
        # connection_decision_result_action(dict_action)
        
        if decision_data != dict():
            for result_action in connection_decision_result_action:
                if is_format:
                    will_be_sended = ""
                else:
                    will_be_sended = dict()
                if callable(result_action):
                    for key, value in pass_action_data.items():
                        if is_format:
                            will_be_sended += format.format(key, value) + "\n"
                        else:
                            will_be_sended[key] = value

                    result_action(will_be_sended)
                else:
                    result_action = will_be_sended
                
            """
            if decision_data != dict():
                for result_action in connection_decision_result_action:
                    will_be_sended = None
                    if callable(result_action):
                        for key, value in pass_action_data.items():
                            if is_format:
                                will_be_sended = format.format(key, value)
                            else:
                                will_be_sended = {key: value}
                        
                        result_action(will_be_sended)
                    else:
                        result_action = will_be_sended
            """


    def make_decision(
            self, 
            connection_decision, 
            connection_decision_params, 
            connection_decision_result_action, 
            is_format=True, 
            format="{}: {}", 
            key_actions=["snapShot"], 
            keys_pass=["panoId", "ethernetState", "emergencyStop"],
            default_start_data=None
        ):
        decision_data = connection_decision[0](
            *connection_decision_params[0]
        )["decision_data"]
        decision_result_data = connection_decision[1](
            *connection_decision_params[1]
        )

        dict_action = dict()
        dict_keys_passed = dict()
        is_everything_zero = True

        if decision_data is not None and type(decision_data) is dict:
            # key_contains = "snapShot"
            # ethernet_state = decision_data["ethernetState"]
            # pano_ID = decision_data["panoId"]
            #print("decision_data:", decision_data)
            
            for key, value in decision_data.items():
                
                if key in keys_pass:        
                    dict_keys_passed[key] = value
                    continue
                
                if int(value) != 0:
                    is_everything_zero = False

                # print("key:", key, "| value:", value)

                #if key_contains not in key:
                #    break
                
                if key_actions is not None:
                    """
                    will_break = False
                    for key_action in key_actions:
                        if key_action not in key:
                            will_break = True
                            break
                    if will_break:
                        break
                    """
                
                    # Wrong type of action will break the loop
                    if len([key_action for key_action in key_actions if key_action not in key]) > 0:
                        break

                if int(value) != 0:
                    dict_action[key] = value
                


            if dict_action != dict():
                # OR #
                # connection_decision_result_action(dict_action)
                
                for result_action in connection_decision_result_action:
                    will_be_sended = None
                    if callable(result_action):
                        for key, value in dict_action.items():
                            
                            if is_format:
                                will_be_sended = format.format(key, value)
                                
                                for sub_key, sub_value in dict_keys_passed.items():
                                    will_be_sended += "\n" + format.format(sub_key, sub_value)
                                # result_action(format.format(key, value))
                            else:
                                will_be_sended = {key: value}
                                
                                for sub_key, sub_value in dict_keys_passed.items():
                                    will_be_sended[sub_key] = sub_value
                                # result_action({key: value})
                        
                        result_action(will_be_sended)
                    else:
                        result_action = will_be_sended
                
                self.dict_last_decision = dict_action
                
                for d in (dict_action, dict_keys_passed): self.dict_last_decision.update(d)
                #self.logger.debug("Last Actions: {}".format(self.dict_last_decision))
        
        
            temp_decision_history = dict()
            for d in (dict_action, dict_keys_passed): temp_decision_history.update(d)
            self.decision_history.append(temp_decision_history)
            #self.logger.debug("self.decision_history: {}".format(self.decision_history))
            
            """
            if True:
                import random, time
                for key, value in decision_result_data.items():
                    random.seed(time.time())
                    decision_result_data[key] = random.randrange(2)
            """
            # import pdb; pdb.set_trace()
            
            if is_everything_zero:
                
                # print("ZERO") #, decision_data)
                for key, value in decision_result_data.items():
                    decision_result_data[key] = 0

                pass_action_data = dict()
                for d in (dict_keys_passed, {"pass": 1}): pass_action_data.update(d)
                
                # decision_data, pass_action_data, connection_decision_result_action, is_format=True, format="{}: {}"
                self.send_all_connections(decision_data, pass_action_data, connection_decision_result_action, is_format=True, format="{}: {}")
                """
                if decision_data != dict():
                    for result_action in connection_decision_result_action:
                        will_be_sended = None
                        if callable(result_action):
                            for key, value in pass_action_data.items():
                                if is_format:
                                    will_be_sended = format.format(key, value)
                                else:
                                    will_be_sended = {key: value}
                            
                            result_action(will_be_sended)
                        else:
                            result_action = will_be_sended
                """
            #else:
                #print("NON-ZERO")  # , decision_data)

            self.decision_result_data = decision_result_data
            
            return dict_action, dict_keys_passed, decision_data, decision_result_data
        # print("Decision Maker: WHAT")
        # print("dict_action: {}\n dict_keys_passed: {}\n connection_decision_result_action: {}".format(
        #     dict_action, dict_keys_passed, connection_decision_result_action))
        # import pdb; pdb.set_trace()
        
        if default_start_data is not None:
            pass_action_data = default_start_data
            self.send_all_connections(decision_data, pass_action_data, connection_decision_result_action, is_format=True, format="{}: {}")
        return -1, -1, -1, -1

       
    def get_information(self, filter_keys=None, rename=None, return_only_text=True, return_only_dict=True, text_format="{}:{}", ends="\n"):
        """ 
        dict_information = dict()
        dict_information["host"] = self.host
        dict_information["port"] = self.port
        dict_information["is_connection_ok"] = self.is_connection_ok
        dict_information["set_blocking"] = self.set_blocking
        dict_information["timeout"] = self.timeout
        dict_information["data_last_sended"] = self.data_last_sended
        """
        # import pdb; pdb.set_trace()
        if self.decision_result_data != dict():
            get_dict_information = self.decision_result_data
            if filter_keys is not None:
                swap_dict_information = dict()

                if rename is not None:
                    for index, filter_key in enumerate(filter_keys):
                        swap_dict_information[ rename[index] ] = get_dict_information[filter_key]

                else:
                    for filter_key in filter_keys:
                        swap_dict_information[filter_key] = get_dict_information[filter_key]

                get_dict_information = swap_dict_information

            temp_text = ""
            if return_only_dict is False and return_only_text is True:
                temp_text = ""
                for key, value in get_dict_information.items():
                    # print("key:", key, "value:", value)
                    temp_text += text_format.format(str(key), str(value)) + ends
        else:
            get_dict_information = self.decision_result_data
            temp_text = ""

        if return_only_text:
            return temp_text
        else:
            if return_only_dict:
                return get_dict_information
            else:
                return temp_text, get_dict_information

