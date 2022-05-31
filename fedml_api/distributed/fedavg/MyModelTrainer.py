import logging
import colorama

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

colorama.init()

class MyModelTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict(keep_vars=True)

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_gradients(self):
        grads = []
        for param in self.model.parameters():
            g = param.grad.view(-1).tolist()
            grads.append(g)
            print("Getting model's gradient:", torch.Tensor(g).shape)

        return grads

    def train(self, train_data, device, args, train_label):
        model = self.model

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd, amsgrad=True)
            optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
            print('using Adam without learning rate decay')

        # print("= = = = = = = = = = training data = = = = = = = = = =")
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # xx, llabels = x[0].view(1, -1), labels[0].view(-1,)
                # for i in range(1, len(labels)):
                #     if labels[i] == train_label:
                #         xx = torch.cat((xx, x[i].view(1, -1)), 0)
                #         # print('before cat:', llabels, 'label:', labels[i].view(-1,))
                #         llabels = torch.cat((llabels, labels[i].view(-1,)), 0)


                # if labels[0] != train_label:
                #     xx = xx[1:]
                #     llabels = llabels[1:]

                if epoch == 0:
                    print(labels, labels.shape)

                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,
                                                                                              epoch,
                                                                                              sum(epoch_loss) / len(
                                                                                                  epoch_loss)))
        #print('Gradient shape(weight):', self.get_model_w_gradients().shape)
        #print('Gradient shape(bias):', self.get_model_b_gradients().shape)

    def test(self, test_data, device, args, test_label):
        model = self.model

        model.eval()
        model.to(device)

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                # xx, ttarget = x[0].view(1, -1), target[0].view(-1,)
                # for i in range(1, len(target)):
                #     if target[i] == test_label:
                #         xx = torch.cat((xx, x[i].view(1, -1)), 0)
                #         ttarget = torch.cat((ttarget, target[i].view(-1,)), 0)

                # if target[0] != test_label:
                #     xx = xx[1:]
                #     ttarget = ttarget[1:]

                # if len(ttarget) == 0:
                #     continue

                x, target = x.to(device), target.to(device)
                #print(ttarget, target, ttarget.shape, target.shape)


                pred = model(x)
                loss = criterion(pred, target)
                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    # print('pred:', pred)
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()
                    # if batch_idx <= 10:
                        # print('Predicted:', predicted, 'Target:', target)

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
