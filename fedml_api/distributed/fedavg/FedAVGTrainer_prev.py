from .utils import transform_tensor_to_list
import colorama

colorama.init()

class FedAVGTrainer(object):

    def __init__(self, client_index, dataset_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.dataset_index = dataset_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[dataset_index]
        self.local_sample_number = self.train_data_local_num_dict[dataset_index]
        self.test_local = self.test_data_local_dict[dataset_index]

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, dataset_index):
        self.dataset_index = dataset_index
        self.train_local = self.train_data_local_dict[dataset_index]
        self.local_sample_number = self.train_data_local_num_dict[dataset_index]
        self.test_local = self.test_data_local_dict[dataset_index]

    def train(self, round_idx = None):
        self.args.round_idx = round_idx
        print(colorama.Fore.LIGHTGREEN_EX + 'train() in FedAVGTrainer, dataset_index:', str(self.dataset_index), ', round:' + str(round_idx) + colorama.Style.RESET_ALL)
        
        # print training data based on client's index
        # for i in range(len(self.train_local)):
        #     print(self.train_local[i][1])
        
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        gradients = self.trainer.get_model_gradients().tolist()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number, gradients

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample