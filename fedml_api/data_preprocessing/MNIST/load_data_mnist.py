import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import json
import numpy as np
import os
# from noniid_partition import non_iid_partition_with_dirichlet_distribution

np.random.seed(0)
torch.manual_seed(10)

train_data = torchvision.datasets.MNIST(root='../../data', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='../../data', train=False)

'''
print(type(train_data_local_dict[0][0][0]), train_data_local_dict[0][0][0].shape, type(train_data_local_dict[0][0][1]), train_data_local_dict[0][0][1])
print(train_data_local_num_dict)
print('Start stacking label_list')
label_list_to_tensor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10):
    label_list[i] = torch.Tensor(label_list[i])
    print(len(label_list[i]))
print(label_list[1][:10])

==========================================================================================
data partition for non-iid distribution

buffer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print('Start non_iid partitioning for label:')
indexes = non_iid_partition_with_dirichlet_distribution(torch.Tensor(ll), 50, 10, 0.5)

for k in range(0, 2):
    for i in range(len(indexes[k])):
        buffer[train_data[indexes[k][i]][1]] += 1
    print(buffer)

for i in range(10):
    print('Start non_iid partitioning for label:', i)
    print(torch.Tensor(label_list[i]).shape)
    indexes = non_iid_partition_with_dirichlet_distribution(torch.Tensor(label_list[i]), 50, 1, 0.5)
    print('Total length:', len(label_list[i]))
    path = "./label_" + str(i) + '/'
    os.mkdir(path)
    for k in range(len(indexes)):
        print('In partition:', k)
        buffer = []
        for t in range(len(indexes[k])):
            buffer.append((train_data_local_dict[i][indexes[k][t]][0].tolist(), train_data_local_dict[i][indexes[k][t]][1]))
        buffer = json.dumps(buffer)
        with open(path + '/tensors' + '_' + str(i) + '_' + str(k) + '.json', 'w') as outfile:
            json.dump(buffer, outfile)
        print('Saving:', './tensors' + '_' + str(i) + '_' + str(k) + '.json')
'''

def load_mnist():
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