import logging
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_images_labels_prediction(images, labels, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx].reshape(28, 28), cmap = 'binary')

        title = str(i) + '.' + str(labels[i])

        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
from fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.linear.lr import LogisticRegression

# data = torch.load('6.pt')
data = torch.load('10.pt')

for i in range(len(data)):
    plot_images_labels_prediction(data[i][0], data[i][1], 0, len(data[i][1]))

# print(data1[0][0][0])