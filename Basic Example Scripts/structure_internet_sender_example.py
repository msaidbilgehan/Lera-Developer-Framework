from structure_internet import Internet_Sender

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
    # logger_level = logging.INFO  # DEBUG
    # disable_Logger = True  # False

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

        # Internet_Receiver_1 = Internet_Receiver(
        #     host=ip_receiver,
        #     port=port_receiver,
        #     timeout=timeout,
        #     set_blocking=set_blocking,
        #     logger_level=logger_level,
        #     parsing_format=parsing_format,
        #     delay=delay,
        #     error_counter_max=error_counter_max,
        #     max_buffer_limit=max_buffer_limit
        # )
        # Internet_Receiver_1.logger.disabled = disable_Logger

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
            # logger_level=logger_level,
            delay=delay,
            error_counter_max=error_counter_max,
            max_buffer_limit=max_buffer_limit
        )
        # Internet_Sender_1.logger.disabled = disable_Logger
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
            Internet_Sender_1.buffer_Append(str(data), lock_until_done=True)

        stdo(
            1,
            "Random Data Filled to Sender Buffer: {}".format(
                Internet_Sender_1.buffer_Get_Len()
            )
        )

        # Start Internet Object
        # Internet_Receiver_1.start()
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
            print(
                f"Buffer Element Number:{Internet_Sender_1.buffer_Get_Len()}  ", 
                end='\r'
            )
            pass
        else:
            stdo(
                1,
                "\nSender Buffer is Cleared: {}".format(
                    Internet_Sender_1.buffer_Get_Len()
                )
            )

        stdo(1, "Waiting to finish Internet Receiver Object to It's Buffer")
        # received_all = list()
        # while Internet_Receiver_1.buffer_Get_Len() != 0:
        #     # print("Sender Len", Internet_Receiver_1.buffer_Get_Len())
        #     received_all.append(
        #         int(
        #             Internet_Receiver_1.buffer_Pop()
        #         )
        #     )
        # Internet_Receiver_1.buffer_Clear()
        Internet_Sender_1.buffer_Clear()

        # Internet_Receiver_1.quit()
        Internet_Sender_1.quit()

        stdo(1, "\t\t Sended Data:{}".format(random_data))
        # stdo(1, "\t\t Received Data:{}".format(received_all))
        # stdo(
        #     1,
        #     "Blind Spot [{}]:".format(
        #         str(len(set(random_data)-set(received_all)))
        #     )
        # )
        # stdo(1, "\t\t |- {}".format(set(random_data)-set(received_all)))

        # while True: pass

        # blind_spot_all[
        #     "{}. Try: Blind Spot is ({})".format(
        #         try_time,
        #         len(set(random_data)-set(received_all))
        #     )
        # ] = set(random_data)-set(received_all)

    stdo(1, "===============================")
    stdo(1, "============RESULTS============")
    stdo(1, "===============================")

    for key, value in blind_spot_all.items():
        stdo(1, "{} -> {}".format(key, value))

    stdo(1, "===============================")
    stdo(1, "===============================")
    stdo(1, "===============================")
