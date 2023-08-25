import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad


class Server(BaseFedarated):
    def __init__(self, hyper_params, Model, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(hyper_params['learning_rate'], hyper_params['mu'])
        super(Server, self).__init__(hyper_params, Model, dataset)

    def train(self):
        """Train using Federated Proximal"""
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.evaluate('test') # have set the latest model for all clients
                stats_train = self.evaluate('train')

                tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} acc. variance: {}'.format(i, np.var([i/j for i,j in zip(stats[3], stats[2])])))  # testing accuracy variance
                tqdm.write('At round {} acc. 10th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))  # testing accuracy variance
                tqdm.write('At round {} acc. 20th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))  # testing accuracy variance
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

            model_len = process_grad(self.latest_model_params).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                client_grad = c.get_grads(model_len)  # get client_grad and operate on it
                #local_grads.append(client_grad)
                #num_samples.append(num)
                #global_grads = np.add(global_grads, client_grad * num)
            #global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            #difference = 0
            #for idx in range(len(self.clients)):
            #    difference += np.sum(np.square(global_grads - local_grads[idx]))
            #difference = difference * 1.0 / len(self.clients)
            #tqdm.write('gradient difference: {}'.format(difference))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model_params, self.client_model)

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model_params)

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

                # solve minimization locally
                if c in active_clients:
                    soln, stats, grads = c.train_for_epochs(num_epochs=self.num_epochs, batch_size=self.batch_size)
                else:
                    #soln, stats = c.train_for_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size)
                    soln, stats, grads = c.train_for_epochs(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model_params = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model_params)

        # final test model
        stats = self.evaluate('test')
        stats_train = self.evaluate('train')
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} acc. variance: {}'.format(self.num_rounds, np.var([i/j for i,j in zip(stats[3], stats[2])])))
        tqdm.write('At round {} acc. 10th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))
        tqdm.write('At round {} acc. 20th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
