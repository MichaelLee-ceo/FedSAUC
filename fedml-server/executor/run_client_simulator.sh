cd ../client_simulator
nohup python3 ../client_simulator/mobile_client_simulator.py --client_uuid '0' > ./mobile_client_log_0.txt 2>&1 &
nohup python3 ../client_simulator/mobile_client_simulator.py --client_uuid '1' > ./mobile_client_log_1.txt 2>&1 &
nohup python3 ../client_simulator/mobile_client_simulator.py --client_uuid '2' > ./mobile_client_log_2.txt 2>&1 &
nohup python3 ../client_simulator/mobile_client_simulator.py --client_uuid '3' > ./mobile_client_log_3.txt 2>&1 &