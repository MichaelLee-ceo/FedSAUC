# run this script before starting app.py with desired device number and communication rounds
cd preprocessed_dataset
python ../../FedML/fedml_api/data_preprocessing/MNIST/mnist_mobile_preprocessor.py --client_num_per_round 4 --comm_round 200