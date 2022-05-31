import socket
import time

dataset_index = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]

HOST = ['192.168.50.101', '192.168.50.102', '192.168.50.103', '192.168.50.104', '192.168.50.105',
			 '192.168.50.201', '192.168.50.202', '192.168.50.203', '192.168.50.204', '192.168.50.205']

for i in range(8000, 8010):
	# print(i)
	PORT = i
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((HOST[i % 10], PORT))

	s.send(('python3 fedavg_jetson_nano_client.py --server_ip http://192.168.50.106:5000 --client_uuid %d --dataset_index %d' % (i % 10, dataset_index[i % 10])).encode())
	print('Client: %d| datast_index %d send out request!' % (i % 10, dataset_index[i % 10]))
