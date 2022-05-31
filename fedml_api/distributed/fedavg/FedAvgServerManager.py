import logging
import os
import sys
import requests
import random
import time

from .message_define import MyMessage
from .utils import transform_tensor_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager


class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, registered_device=None, cluster_round=50):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.registered_device = registered_device
        self.cluster_round = cluster_round

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], False)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        # print('handle message from client')
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        gradients = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_WGRADIENTS)
        #b_model_gradients = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_BGRADIENTS)
        #model_gradients = 0

        # print('after message decoding')
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, gradients, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params, cluster_result = self.aggregator.aggregate(self.round_idx, self.round_num)
            global_model_params = transform_tensor_to_list(global_model_params)

            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                for receiver in self.registered_device:
                    self.send_message_sync_model_to_client(receiver + 1, global_model_params, receiver, True)
                    print('Terminating device:', receiver)
                self.finish()

                print('sending shutdown to app.route')
                URL = 'http://192.168.50.106:5000' + "/shutdown"
                r = requests.post(url=URL)

                print('sent shutdown to app.route')
                return

            # if self.is_preprocessed:
            #     # sampling has already been done in data preprocessor
            #     client_indexes = [self.round_idx] * self.args.client_num_per_round
            # else:
            #     # # sampling clients
            #     client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
            #                                                      self.args.client_num_per_round)

            # if self.args.is_mobile == 1:
            # print("transform_tensor_to_list")

             # implement device selection based on the cluster_result return from aggregate() after predefined round
            if self.round_idx < self.cluster_round:                                                                                      # train model using all devices
                print('training with all devices')
                for receiver in self.registered_device:
                    self.send_message_sync_model_to_client(receiver + 1, global_model_params, receiver, False)
            else:
                print('training with selected devices')
                next_round_devices = []
                # for c in cluster_result:
                #     candidate = list(c)[random.randint(0, len(c) - 1)]
                #     next_round_devices.append(candidate)
                devices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                next_round_devices = random.sample(devices, 5)
                logging.info('Selected %d devices for next round: %s' % (len(next_round_devices), str(next_round_devices)))
                # reset previous devices records stored in aggregator
                self.aggregator.reset_round_training_devices(next_round_devices, len(next_round_devices))
                # send synchronous model to selected devices for next round
                for receiver in next_round_devices:
                    self.send_message_sync_model_to_client(receiver + 1, global_model_params, receiver, False)

            # logging.info('indexes of clients: %s' % str(self.registered_device))
            # logging.info("size = %d" % self.size)


    def send_message_init_config(self, receive_id, global_model_params, client_index, isfinished):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_FINISH, isfinished)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, isfinished):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_FINISH, isfinished)
        self.send_message(message)
