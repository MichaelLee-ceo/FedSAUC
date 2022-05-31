import json
import os

import numpy as np
import torch
import torchvision

import colorama
colorama.init()


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data

def load_mnist():
    np.random.seed(0)
    torch.manual_seed(10)
    train_data = torchvision.datasets.MNIST(root='../../data', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='../../data', train=False)

    train_data_local_dict = [dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]})]
    test_data_local_dict = [dict({'x':[], 'y': []}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]}), dict({'x':[], 'y':[]})]
    # test_data_global_dict = [dict({'x':[], 'y':[]})]
    for i in range(60000):
        if train_data[i][1] == 0 or train_data[i][1] == 1:
            train_data_local_dict[0]['x'].append(np.array(train_data[i][0]))
            train_data_local_dict[0]['y'].append(train_data[i][1])
        elif train_data[i][1] == 2 or train_data[i][1] == 3:
            train_data_local_dict[1]['x'].append(np.array(train_data[i][0]))
            train_data_local_dict[1]['y'].append(train_data[i][1])
        elif train_data[i][1] == 4 or train_data[i][1] == 5:
            train_data_local_dict[2]['x'].append(np.array(train_data[i][0]))
            train_data_local_dict[2]['y'].append(train_data[i][1])
        elif train_data[i][1] == 6 or train_data[i][1] == 7:
            train_data_local_dict[3]['x'].append(np.array(train_data[i][0]))
            train_data_local_dict[3]['y'].append(train_data[i][1])
        elif train_data[i][1] == 8 or train_data[i][1] == 9:
            train_data_local_dict[4]['x'].append(np.array(train_data[i][0]))
            train_data_local_dict[4]['y'].append(train_data[i][1])
    for i in range(10000):
        # test_data_global_dict[0]['x'].append(np.array(test_data[i][0]))
        # test_data_global_dict[0]['y'].append(test_data[i][1])

        if test_data[i][1] == 0 or test_data[i][1] == 1:
            test_data_local_dict[0]['x'].append(np.array(test_data[i][0]))
            test_data_local_dict[0]['y'].append(test_data[i][1])
        elif test_data[i][1] == 2 or test_data[i][1] == 3:
            test_data_local_dict[1]['x'].append(np.array(test_data[i][0]))
            test_data_local_dict[1]['y'].append(test_data[i][1])
        elif test_data[i][1] == 4 or test_data[i][1] == 5:
            test_data_local_dict[2]['x'].append(np.array(test_data[i][0]))
            test_data_local_dict[2]['y'].append(test_data[i][1])
        elif test_data[i][1] == 6 or test_data[i][1] == 7:
            test_data_local_dict[3]['x'].append(np.array(test_data[i][0]))
            test_data_local_dict[3]['y'].append(test_data[i][1])
        elif test_data[i][1] == 8 or test_data[i][1] == 9:
            test_data_local_dict[4]['x'].append(np.array(test_data[i][0]))
            test_data_local_dict[4]['y'].append(test_data[i][1])


    return train_data_local_dict, test_data_local_dict


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size,
                                           device_id,
                                           train_path="MNIST_mobile",
                                           test_path="MNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_mnist(batch_size)


def load_partition_data_mnist(batch_size):
    print(colorama.Fore.LIGHTGREEN_EX + "start load_partition_data_mnist in data_loader.py" + colorama.Style.RESET_ALL)

    # check file directory
    # print('current directory:', os.getcwd())    
    # users, groups, train_data, test_data = read_data(train_path, test_path)
    #
    # if len(groups) == 0:
    #     groups = [None for _ in users]

    train_data, test_data = load_mnist()

    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for i in range(5):
        user_train_data_num = len(train_data[i]['x'])
        user_test_data_num = len(test_data[i]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[i], batch_size)
        test_batch = batch_data(test_data[i], batch_size)

        # each client takes data from train_data_local_dict[distribution]
        # distribution = {'0': (0, 1)} / {'1': (2, 3)} / {'2': (4, 5)}
        #                                {'3': (6, 7)} / {'4': (8, 9)}
        # batch_data from ==>
        # for (x, label) in enumerate(train_data_local_dict[distribution][batch]):

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    client_num = client_idx
    class_num = 10

    print(colorama.Fore.LIGHTGREEN_EX + "finish load_partition_data_mnist in data_loader.py" + colorama.Style.RESET_ALL)
	
    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num