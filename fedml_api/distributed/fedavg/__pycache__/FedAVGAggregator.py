import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor
import colorama

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime

colorama.init()

class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer, registered_devices, dataset_indexes):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.model_gradient = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = registered_devices
        self.dataset_indexes = dataset_indexes
        self.clusterCorrect = 0
        self.clusterIncorrect = 0
        # self.result_file = open('cluster_result.txt', 'w')
        # for idx in range(self.worker_num):
        #     self.flag_client_model_uploaded_dict[idx] = False
        self.now = datetime.now() # current date and time
        self.date_time = self.now.strftime("%Y_%m_%d_%H_%M_%S")


    # def __del__(self):
        # self.result_file.close()
        # print('File closed()')

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, model_gradients, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        # weight = torch.Tensor(model_params['linear.weight']).reshape(-1,)
        # bias = torch.Tensor(model_params['linear.bias']).reshape(-1,)
        # self.model_gradient[index] = torch.cat((weight, bias), 0)
        self.model_gradient[index] = torch.Tensor(model_gradients)    
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        if len(self.flag_client_model_uploaded_dict) != self.worker_num:
            return False

        for idx in self.flag_client_model_uploaded_dict:
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in self.flag_client_model_uploaded_dict:
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self, round_idx, total_round):
        start_time = time.time()
        model_list = []
        gradient_list = []
        training_num = 0

        # print('start collecting model and gradients')
        for idx in self.model_dict:
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

            gradient_list.append(self.model_gradient[idx])
            print('Appending model_gradient:', idx, self.model_gradient[idx].shape)

        print('Start clustering, gradient list length:', len(gradient_list))

        result_file = open('result_' + self.date_time + '.txt', 'a')

        # k-means for gradient
        cluster = AgglomerativeClustering(n_clusters=5)
        cluster.fit(torch.stack(gradient_list))

        classes = [set(),set(),set(),set(),set()]
        for i ,idx in enumerate(self.model_dict):
            classes[cluster.labels_[i]].add(idx)

        Error= [0,0,0,0,0]
        countt = 0
        for clas in classes:
            currentClassError=0
            inCurrentGroup=0
            for id in clas:
                if id %2 ==0:
                    if(id+1) not in clas :
                        currentClassError+=1
                    else:
                        inCurrentGroup+=1
            Error[countt]= currentClassError + max((inCurrentGroup-1)*2,0)
            countt += 1

        tempError = sum(Error)
        round_error = tempError
        round_correct = 10 - round_error

        self.clusterIncorrect += round_error
        self.clusterCorrect += round_correct

        print(Error)
        print(classes)

        '''
        for i, idx in enumerate(self.model_dict):                               # type(cluster.labels) -> np.ndarray
            print('Round: ' + str(round_idx) + '| ' + 'client: ' + str(idx) +  ', class: ' + str(cluster.labels_[i]))
            result_file.write('Round: ' + str(round_idx) + '| ' + 'client: ' + str(idx) +  ', class: ' + str(cluster.labels_[i]) + '\n')

            # partition devices by cluster result
            if cluster.labels_[i] == 0:
                class_0.add(idx)
            elif cluster.labels_[i] == 1:
                class_1.add(idx)
            elif cluster.labels_[i] == 2:
                class_2.add(idx)
            elif cluster.labels_[i] == 3:
                class_3.add(idx)
            elif cluster.labels_[i] == 4:
                class_4.add(idx)
        print(cluster.labels_)

        partition1_label = -1
        partition2_label = -1
        partition3_label = -1
        partition4_label = -1
        partition5_label = -1
        if len(partition_1.intersection(class_0)) > len(partition_2.intersection(class_0)):
            partition1_label = 0
            partition2_label = 1
        else:
            partition1_label = 1
            partition2_label = 0

        round_acc1 = 0
        round_acc2 = 0
        tempIncorrect = 0

        for i, idx in enumerate(self.model_dict):
            if idx in partition_1 and cluster.labels_[i] == partition1_label:
                round_acc1 += 1
                self.clusterCorrect += 1
            elif idx in partition_2 and cluster.labels_[i] == partition2_label:
                round_acc2 += 1
                self.clusterCorrect += 1
            else:
                self.clusterIncorrect += 1
                tempIncorrect += 1

        round_correct = round_acc1 + round_acc2
        '''
        '''
        keys = list((self.model_dict).keys())
        tempIncorrect = 0
        for i in range(0, 10, 5):
            index1 = keys.index(i)
            index2 = keys.index(i + 1)
            index3 = keys.index(i + 2)
            index4 = keys.index(i + 3)
            index5 = keys.index(i + 4)
            if cluster.labels_[index1] == cluster.labels_[index2] == cluster.labels_[index3] == cluster.labels_[index4] == cluster.labels_[index5]:
                self.clusterCorrect += 1
            else:
                self.clusterIncorrect += 1
                tempIncorrect += 1
        '''

        wandb.log({"Round Correct": round_correct / 10, "round": round_idx})
        #wandb.log({"Round partition_1 Correct": round_acc1 / 5, "round": round_idx})
        #wandb.log({"Round partition_2 Correct": round_acc2 / 5, "round": round_idx})
        #wandb.log({"Round partition_1 Error": (5 - round_acc1) / 5, "round": round_idx})
        #wandb.log({"Round partition_2 Error": (5 - round_acc2) / 5, "round": round_idx})
        wandb.log({"Round Error": round_error / 10, "round": round_idx})
        wandb.log({"Total Cluster Accuracy": self.clusterCorrect / ((round_idx+1) * 10), "round": round_idx})

        print('Total cluster accuracy: ' + str(self.clusterCorrect / ((round_idx+1) * 10)) + ', Error rate: ' + str(self.clusterIncorrect / ((round_idx+1) * 10)))
        result_file.write('Round Correct: ' + str(round_correct / 10) + ', Round Incorrect: ' + str(round_error / 10) + '\n')
        result_file.write('Total cluster accuracy: ' + str(self.clusterCorrect / ((round_idx+1) * 10)) + ', Error rate: ' + str(self.clusterIncorrect / ((round_idx+1) * 10)) + '\n')
        result_file.write('Error: ' + str(Error))
        result_file.write('Classes: ' + str(classes))
        result_file.write('= = = = = = = = = = = = = = = = = = = = = = = = \n')
        result_file.close()

        # logging.info("len of self.model_dict= " + str(len(self.model_dict)))
        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index + round_idx * 10 for client_index in range(client_num_in_total)]         # 10 clients per cycle with similar labels
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))

            # train data
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            # record testing result for all classes -- 10 classes, from training data 
            accuracies = np.zeros(10)
            losses = np.zeros(10)

            ds = []
            for client_idx in range(1000): # self.args.client_num_in_total
                # train data
                if (client_idx % 10) in self.dataset_indexes:
                    ds.append(client_idx)
                    metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                    metrics_2 = self.trainer.test(self.test_data_local_dict[client_idx], self.device, self.args)
                    train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                    test_tot_correct, test_num_sample, test_loss = metrics_2['test_correct'], metrics_2['test_total'], metrics_2['test_loss']

                    # all classes testing result from training data
                    accuracies[(client_idx) % 10] += train_tot_correct / train_num_sample
                    losses[(client_idx) % 10] += train_loss / train_num_sample
                    print('correct:', train_tot_correct, 'loss:', train_loss, 'sample:', train_num_sample)

                    '''
                    # print each client's training result
                    if int(round_idx/10)*10 <= client_idx and client_idx < (int(round_idx/10)+1)*10:
                        print('client' + colorama.Fore.LIGHTCYAN_EX + '[' + str(client_idx) + ']' + colorama.Style.RESET_ALL + ' training data:')
                        for i in range(len(self.train_data_local_dict[client_idx])):
                            print(self.train_data_local_dict[client_idx][i][1])
                        print('client[' + str(client_idx) + '] training_acc:' + colorama.Fore.LIGHTRED_EX, train_tot_correct / train_num_sample,  colorama.Style.RESET_ALL + 'training_loss:' + colorama.Fore.LIGHTRED_EX, train_loss / train_num_sample, colorama.Style.RESET_ALL, '\n')
                    '''

                    train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                    train_num_samples.append(copy.deepcopy(train_num_sample))
                    train_losses.append(copy.deepcopy(train_loss))

                    test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                    test_num_samples.append(copy.deepcopy(test_num_sample))
                    test_losses.append(copy.deepcopy(test_loss))

                    """
                    Note: CI environment is CPU-based computing. 
                    The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                    """
                    if self.args.ci == 1:
                        break

            print('dataset_indexes:', ds)
            # average the result
            accuracies /= 100
            losses /= 100
            for i in range(10):
                # temp_idx = (idx + 10) % 10
                # print('client' + colorama.Fore.LIGHTCYAN_EX + '[' + str(idx) + ']' + colorama.Style.RESET_ALL + ' training data:')
                # for i in range(len(self.train_data_local_dict[idx])):
                #     print(self.train_data_local_dict[idx][i][1])
                print('Label' + colorama.Fore.LIGHTCYAN_EX + '[' + str(i) + ']' + colorama.Style.RESET_ALL + ' training_acc:' + colorama.Fore.LIGHTRED_EX, accuracies[i], colorama.Style.RESET_ALL + 'training_loss:' + colorama.Fore.LIGHTRED_EX, losses[i], colorama.Style.RESET_ALL)

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test on testing dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

            '''
            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
            '''
