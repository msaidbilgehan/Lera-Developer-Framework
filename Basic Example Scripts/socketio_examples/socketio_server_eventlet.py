import eventlet
import socketio


IP = "localhost"
# IP = "192.168.22.22"
PORT = 8888

sio_Server = socketio.Server()
app = socketio.WSGIApp(
    sio_Server,
    static_files={
        '/': {'content_type': 'text/html', 'filename': 'loading.html'}
    }
)

@sio_Server.event
def connect(sid, environ):
    print('connect ', sid)

@sio_Server.event
def my_message(sid, data):
    print('message ', data)

@sio_Server.event
def disconnect(sid):
    print('disconnect ', sid)


eventlet.wsgi.server(
    eventlet.listen((IP, PORT)),
    app
)
