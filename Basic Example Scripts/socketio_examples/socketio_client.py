import socketio

IP = "localhost"
# IP = "192.168.22.22"
# PORT = "8888"
PORT = "8080"

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')


@sio.event
def my_message(data):
    print('message received with ', data)
    sio.emit('my response', {'response': 'my response'})


@sio.event
def disconnect():
    print('disconnected from server')


sio.connect(f'http://{IP}:{PORT}')
sio.wait()
