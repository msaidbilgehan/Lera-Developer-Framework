

def request_Time_From_NTP(addr='0.tr.pool.ntp.org'):
    global socket, struct, time
    import socket, struct, time
    
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

if __name__ == "__main__":
    print(request_Time_From_NTP())