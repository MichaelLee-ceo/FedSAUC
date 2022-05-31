import logging
import os
import sys
import time

import argparse
import numpy as np
import torch
import wandb

#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from FedML.fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer

from FedML.fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from FedML.fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from FedML.fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from FedML.fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from FedML.fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from FedML.fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist

from FedML.fedml_api.model.cv.cnn import CNN_DropOut
from FedML.fedml_api.model.cv.resnet_gn import resnet18
from FedML.fedml_api.model.linear.lr import LogisticRegression
from FedML.fedml_api.model.nlp.rnn import RNN_OriginalFedAvg
from FedML.fedml_api.model.mlp import MLP

from FedML.fedml_core.distributed.communication.observer import Observer

from flask import Flask, request, jsonify, send_from_directory, abort


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../data/MNIST',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=4,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=1,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='continuous integration')

    parser.add_argument('--is_preprocessed', type=bool, default=False, help='True if data has been preprocessed')

    args = parser.parse_args()
    return args


# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)

device_id_to_client_id_dict = dict()
dataset_indexes = []


@app.route('/', methods=['GET'])
def index():
    return 'backend service for Fed_mobile'


@app.route('/get-preprocessed-data/<dataset_name>', methods = ['GET'])
def get_preprocessed_data(dataset_name):
    directory = app.config['MOBILE_PREPROCESSED_DATASETS'] + args.dataset.upper() + '_mobile_zip/'
    try:
        return send_from_directory(
            directory,
            filename=dataset_name + '.zip',
            as_attachment=True)

    except FileNotFoundError:
        abort(404)


@app.route('/api/register', methods=['POST'])
def register_device():
    global device_id_to_client_id_dict
    # __log.info("register_device()")
    device_id = int(request.args['device_id'])
    dataset_indexes.append(int(request.args['dataset_indexes']) % 10)
    # registered_client_num = len(device_id_to_client_id_dict)
    # if device_id in device_id_to_client_id_dict:
    #     client_id = device_id_to_client_id_dict[device_id]
    # else:
    #     client_id = registered_client_num + 1
    client_id = device_id + 1
    device_id_to_client_id_dict[device_id] = False                  # used in FedAVGAggregator for check_whether_all_received
    print('Registered devices:', device_id_to_client_id_dict)

    training_task_args = {"dataset": args.dataset,
                          "data_dir": args.data_dir,
                          "partition_method": args.partition_method,
                          "partition_alpha": args.partition_alpha,
                          "model": args.model,  
                          "client_num_per_round": args.client_num_per_round,
                          "client_num_in_total": args.client_num_in_total,
                          "comm_round": args.comm_round,
                          "epochs": args.epochs,
                          "client_optimizer" : args.client_optimizer,
                          "lr": args.lr,
                          "wd": args.wd,
                          "batch_size": args.batch_size,
                          "frequency_of_the_test": args.frequency_of_the_test,
                          "is_mobile": args.is_mobile,
                          'dataset_url': '{}/get-preprocessed-data/{}'.format(
                              request.url_root,
                              client_id
                          ),
                          'is_preprocessed': args.is_preprocessed}

    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    print('in shutdown')
    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return 'shutting down'


def load_data(args, dataset_name):
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size, 
											  train_path="../../data/MNIST/train",
                                              test_path="../../data/MNIST/test")
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        # default client_num_in_total = 1000
        # args.client_num_in_total = client_num
    elif dataset_name == "femnist":
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir, args.batch_size)
        print('Client Num:', client_num)
    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        elif dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        model = LogisticRegression(28 * 28, output_dim)
        # args.client_optimizer = "sgd"
    elif model_name == "rnn" and args.dataset == "shakespeare":
        model = RNN_OriginalFedAvg(28 * 28, output_dim)
        args.client_optimizer = "sgd"
    elif model_name == "resnet18":
        model = resnet18()
        args.client_optimizer = "adam"
    elif model_name == "cnn":
        model = CNN_DropOut(True)
        # args.client_optimizer = "adam"
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mlp":
        model = MLP(28 * 28, output_dim)
        args.client_optimizer = "adam"

    return model


if __name__ == '__main__':
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.info(args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Server using device:', device)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    print('Train data num:', len(train_data_local_dict))
    print('Test data num:', len(test_data_local_dict))
    
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    cluster_algorithm = ['KMeans', 'AgglomerativeClustering', 'SpectralClustering']
    #cluster_algorithm = ['SpectralClustering']
    #cluster_material = ['W-Weights']
    cluster_material = ['W-Weights', 'W&B-Weights', 'W-Gradients', 'W&B-Gradients']

    for material in cluster_material:
        for method in cluster_algorithm:
            for c in range(500, 510, 10):
                for t in range(1,6):
                    run = wandb.init(
                    # project="federated_nas",
                    project="fedml",
                    name="LR(" + str(c) + ")- equal-distributed(" + material + ",0/2/4/6/8)_" + method + "_" + str(t),
                    config=args,
                    reinit=True
                    )

                    model = create_model(args, model_name=args.model, output_dim=dataset[7])
                    model_trainer = MyModelTrainer(model)

                    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                              train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                              args.client_num_per_round, device, args, model_trainer, device_id_to_client_id_dict, dataset_indexes, c, method, material)
                    size = args.client_num_per_round + 1
                    server_manager = FedAVGServerManager(args,
                                                 aggregator,
                                                 rank=27,
                                                 size=size,
                                                 backend="MQTT",
                                                 is_preprocessed=args.is_preprocessed,
                                                 registered_device=device_id_to_client_id_dict,
                                                 cluster_round=c)
                    server_manager.run()

                    os.system('python3 comm.py')

                    # if run in debug mode, process will be single threaded by default
                    app.run(host='192.168.50.106', port=5000)

                    run.finish()
