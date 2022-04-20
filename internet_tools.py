def request_Time_From_NTP(addr='0.tr.pool.ntp.org'):
    global socket, struct, time
    import socket
    import struct
    import time

    REF_TIME_1970 = 2208988800  # Reference time
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = b'\x1b' + 47 * b'\0'
    client.sendto(data, (addr, 123))
    data, address = client.recvfrom(1024)
    if data:
        ntp_time = struct.unpack('!12I', data)[10]
        ntp_time -= REF_TIME_1970
    else:
        ntp_time = None
    return time.ctime(ntp_time), ntp_time


def request_Time_Over_MySQL_Database(host, user, password, auth_plugin, daytime_format="%Y-%m-%d_%H-%M-%S"):
    import mysql.connector

    # CONNECT TO DATABASE
    mysql_db_connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        auth_plugin=auth_plugin
    )
    mysql_db_cursor = mysql_db_connection.cursor()

    # EXECUTE AND FETCH RESULT
    mysql_db_cursor.execute("SELECT NOW()")
    db_time = mysql_db_cursor.fetchone()[0]

    # CLOSE CURSOR AND CONNECTION
    mysql_db_cursor.close()
    mysql_db_connection.close()

    return db_time.strftime(daytime_format)

if __name__ == '__main__':
    ntp_server_address = '0.tr.pool.ntp.org'
    print("Time from {0} NTP Server is {1}".format(ntp_server_address, request_Time_From_NTP(ntp_server_address)))