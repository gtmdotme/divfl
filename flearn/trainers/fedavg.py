import os
from tqdm import trange, tqdm
from loguru import logger

import numpy as np
from sklearn.metrics import pairwise_distances
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, hyper_params, Model, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.compat.v1.train.GradientDescentOptimizer(hyper_params['learning_rate'])
        super(Server, self).__init__(hyper_params, Model, dataset)
        self.rng = np.random.default_rng()

    def train(self):
        """Train using Federated Proximal"""
        print('Training with {} workers ---'.format(self.clients_per_round))

        # metrics
        test_accuracies = []
        test_losses = []
        test_per_client_accuracies = []
        test_acc_10quant = []
        test_acc_20quant = []
        test_acc_mean = []
        test_acc_var = []
        train_accuracies = []
        train_losses = []
        train_per_client_accuracies = []
        num_sampled = []
        client_sets_all = np.zeros([self.num_rounds, self.clients_per_round], dtype=int)
        diff_grad = np.zeros([self.num_rounds, len(self.clients)])
        
        # training loop at server
        for i in range(self.num_rounds):            
            # compute metrics at start of each round
            if i % self.eval_every == 0:
                stats = self.evaluate('test')
                stats_train = self.evaluate('train')
                
                ## test metrics: `stats`
                # accuracy := total correct over all clients / total samples over all clients
                test_accuracies.append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                # loss := loss of all clients weighted by their # of samples
                test_losses.append(np.dot(stats[4], stats[2]) * 1.0 / np.sum(stats[2]))
                # accuracy per client := list of accuracies of each client
                test_per_client_accuracies.append([i/j for i,j in zip(stats[3], stats[2])])
                # accuracy 10th quantile := 10th quantile of accuracy of all clients
                test_acc_10quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1))
                # accuracy 20th quantile := 20th quantile of accuracy of all clients
                test_acc_20quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2))
                # accuracy mean := mean of accuracy of all clients
                test_acc_mean.append(np.mean([i/j for i,j in zip(stats[3], stats[2])]))
                # accuracy variance := variance of accuracy of all clients
                test_acc_var.append(np.var([i/j for i,j in zip(stats[3], stats[2])]))

                ## train metrics: `stats_train`
                train_accuracies.append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                train_losses.append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))
                train_per_client_accuracies.append([i/j for i,j in zip(stats_train[3], stats_train[2])])

                # print metrics
                print('At round {} per-client-accuracy: {}'.format(i, test_per_client_accuracies[-1]))
                print('At round {} accuracy: {}'.format(i, test_accuracies[-1]))
                print('At round {} loss: {}'.format(i, test_losses[-1]))
                print('At round {} acc. 10th quantile: {}'.format(i, test_acc_10quant[-1]))
                print('At round {} acc. 20th quantile: {}'.format(i, test_acc_20quant[-1]))
                print('At round {} acc. mean: {}'.format(i, test_acc_mean[-1]))
                print('At round {} acc. variance: {}'.format(i, test_acc_var[-1]))
                print('At round {} training accuracy: {}'.format(i, train_accuracies[-1]))
                print('At round {} training loss: {}'.format(i, train_losses[-1]))

            # client selection algo
            if self.clientsel_algo == 'submodular':
                #if i % self.m_interval == 0: # Moved the condition inside the function
                if i == 0 or self.clients_per_round == 1:  # at the first iteration or when m=1, collect gradients from all clients
                    self.all_grads = np.asarray(self.show_grads()[:-1])  # get all gradients from clients (last one contains weightage avg)
                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean") # compute pairwise distance between gradients
                    np.fill_diagonal(self.norm_diff, 0)
                indices, selected_clients, all_grad = self.select_cl_submod(i, num_clients=self.clients_per_round, stochastic_greedy = False)
                active_clients = selected_clients # Dropping clients don't apply in this case
                if i == 0:
                    diff_grad[i] = np.zeros(len(all_grad))
                else:
                    diff_grad[i] = np.linalg.norm(all_grad - old_grad, axis=1)
                old_grad = all_grad.copy()
            elif self.clientsel_algo == 'lossbased':
                print('Power of choice')
                if i % self.m_interval == 0:
                    lprob = stats_train[2]/np.sum(stats_train[2], axis=0)
                    #d=100
                    # subsample = 0.1
                    #d = max(self.clients_per_round, int(subsample * len(self.clients)))
                    d = len(self.clients)
                    lvals = self.rng.choice(stats_train[4], size=d, replace = False, p=lprob)
                    Mlist = [np.where(stats_train[4] == i)[0][0] for i in lvals]
                    lossvals = np.asarray(stats_train[4]) #All loss values
                    sel_losses = lossvals[Mlist]
                    idx_losses = np.argpartition(sel_losses, self.clients_per_round)
                    values = sel_losses[idx_losses[-self.clients_per_round:]]
                    
                    listvalues = values.tolist()
                    listlossvals = lossvals.tolist()
                    indices = [listlossvals.index(i) for i in listvalues] 
                
                #indices = np.argsort(stats_train[4], axis=0)[::-1][:self.clients_per_round]
                selected_clients = np.asarray(self.clients)[indices]
                np.random.seed(i)
                active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)
            else:
                indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
                np.random.seed(i)
                active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)
                
            print('Client set is ', indices)
            client_sets_all[i] = indices
            print('At round {} num. clients sampled: {}'.format(i, len(indices)))
            num_sampled.append(len(indices))
            csolns = []  # buffer for receiving client solutions
            
            # glob_copy = np.append(self.latest_model_params[0].flatten(), self.latest_model_params[1])

            # iterate over all selected clients
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # step 1: send the latest model
                c.set_params(self.latest_model_params)

                # step 2: local training on client
                soln, stats, grads = c.train_for_epochs(num_epochs=self.num_epochs, batch_size=self.batch_size)
                #print("Shape of grads", np.shape(grads))
                
                # step 3: receive updated model from client
                csolns.append(soln)

                if self.clientsel_algo == 'submodular':
                    self.all_grads[indices[idx]] = grads
                
                # Update server's view of clients' models (only for the selected clients)
                #c.updatevec = (glob_copy - np.append(c.get_params()[0].flatten(), c.get_params()[1]))*0.01
                # c.updatevec = np.append(c.get_params()[0].flatten(), c.get_params()[1])

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # step 4: aggregate updated models
            if self.clientsel_algo == 'submodular':
                self.norm_diff[indices] = pairwise_distances(self.all_grads[indices], self.all_grads, metric="euclidean")
                self.norm_diff[:, indices] = self.norm_diff[indices].T
                self.latest_model_params = self.aggregate(csolns)
                #self.latest_model_params = self.aggregate_submod(csolns, gammas)
            elif self.clientsel_algo == 'lossbased':
                self.latest_model_params = self.aggregate_simple(csolns)
            else:
                self.latest_model_params = self.aggregate(csolns)
                

        # compute final metrics after last round
        stats = self.evaluate('test')
        stats_train = self.evaluate('train')
        
        ###### adapted from above ######
        ## test metrics: `stats`
        # accuracy := total correct over all clients / total samples over all clients
        test_accuracies.append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
        # loss := loss of all clients weighted by their # of samples
        test_losses.append(np.dot(stats[4], stats[2]) * 1.0 / np.sum(stats[2]))
        # accuracy per client := list of accuracies of each client
        test_per_client_accuracies.append([i/j for i,j in zip(stats[3], stats[2])])
        # accuracy 10th quantile := 10th quantile of accuracy of all clients
        test_acc_10quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1))
        # accuracy 20th quantile := 20th quantile of accuracy of all clients
        test_acc_20quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2))
        # accuracy mean := mean of accuracy of all clients
        test_acc_mean.append(np.mean([i/j for i,j in zip(stats[3], stats[2])]))
        # accuracy variance := variance of accuracy of all clients
        test_acc_var.append(np.var([i/j for i,j in zip(stats[3], stats[2])]))

        ## train metrics: `stats_train`
        train_accuracies.append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
        train_losses.append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))
        train_per_client_accuracies.append([i/j for i,j in zip(stats_train[3], stats_train[2])])

        # print metrics
        print('At round {} per-client-accuracy: {}'.format(i, test_per_client_accuracies[-1]))
        print('At round {} accuracy: {}'.format(i, test_accuracies[-1]))
        print('At round {} loss: {}'.format(i, test_losses[-1]))
        print('At round {} acc. 10th quantile: {}'.format(i, test_acc_10quant[-1]))
        print('At round {} acc. 20th quantile: {}'.format(i, test_acc_20quant[-1]))
        print('At round {} acc. mean: {}'.format(i, test_acc_mean[-1]))
        print('At round {} acc. variance: {}'.format(i, test_acc_var[-1]))
        print('At round {} training accuracy: {}'.format(i, train_accuracies[-1]))
        print('At round {} training loss: {}'.format(i, train_losses[-1]))

        #  save metrics
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        self.metrics.debug = {
            'test_accuracies': test_accuracies,
            'test_losses': test_losses,
            'test_per_client_accuracies': test_per_client_accuracies,
            'test_acc_10quant': test_acc_10quant,
            'test_acc_20quant': test_acc_20quant,
            'test_acc_mean': test_acc_mean,
            'test_acc_var': test_acc_var,
            'train_accuracies': train_accuracies,
            'train_losses': train_losses,
            'train_per_client_accuracies': train_per_client_accuracies,
        }
        self.metrics.write()

        if self.clientsel_algo == 'submodular':
            np.save(f'./results/{self.dataset}/psubmod_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)
            np.save(f'./results/{self.dataset}/psubmod_client_diff_grad_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), diff_grad)
        elif self.clientsel_algo == 'lossbased':
            np.save(f'./results/{self.dataset}/powerofchoice_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)
