import asyncio
import socketio

IP = "localhost"
# IP = "192.168.22.22"
# PORT = "8888"
PORT = "8080"

sio = socketio.AsyncClient()


@sio.event
async def connect():
    print('connection established')


@sio.event
async def my_message(data):
    print('message received with ', data)
    await sio.emit('my response', {'response': 'my response'})


@sio.event
async def disconnect():
    print('disconnected from server')


async def main():
    await sio.connect(f'http://{IP}:{PORT}')
    await sio.wait()

if __name__ == '__main__':
    asyncio.run(main())
