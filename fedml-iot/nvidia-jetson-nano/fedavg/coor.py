import socket
import os
import argparse
import threading
import time

HOST = '127.0.0.1'

def add_args(parser):
    parser.add_argument('--port', type=int, default="8000", help='port for socket programming')
    args = parser.parse_args()
    return args

def multi_connect(port):
	PORT = port
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((HOST, PORT))
	s.listen()

	while True:
		print('= = = client starts connecting on port %d = = =' % PORT)
		conn, addr = s.accept()
		print('Client Connected by', addr)
	
		data = conn.recv(1024)
		if data != None:
			time.sleep(5)
			os.system(data.decode('utf-8'))
		print('while true')


parser = argparse.ArgumentParser()
args = add_args(parser)

multi_connect(args.port)
  # threads.append(threading.Thread(target = multi_connect, args = (8000 + i,)))
  # threads[i].start()

# for i in range(2):
  # threads[i].join()
	