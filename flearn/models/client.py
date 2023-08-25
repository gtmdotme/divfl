import numpy as np
from loguru import logger

class Client(object):
    """
    Abstraction of client device that holds their personal data
    - has its own local data
    - has its own local copy of model
    - mostly interfaces with model class for:
        - getting/setting model parameters
        - getting gradients
        - doing local training/testing
    """
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        # params
        self.id = id # integer
        self.group = group

        # data
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.test_data = {k: np.array(v) for k, v in test_data.items()}
        self.num_train_samples = len(self.train_data['y'])
        self.num_test_samples = len(self.test_data['y'])

        # model
        self.model = model
        # self.updatevec = np.append(model.get_params()[0].flatten(), model.get_params()[1])

    def set_params(self, model_params):
        """ set model parameters """
        self.model.set_params(model_params)

    def get_params(self):
        """ get model parameters """
        return self.model.get_params()

    def get_grads(self, model_len):
        """ get model gradient """
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        """ get model gradient with cost """
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_train_samples
        bytes_r = self.model.size
        return ((self.num_train_samples, grads), (bytes_w, comp, bytes_r))

    def train_for_epochs(self, num_epochs=1, batch_size=10):
        """ local training based on epochs
        
        Return:
            1: num_train_samples: number of samples used in training
            1: model_params: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        """

        bytes_w = self.model.size
        model_params, comp, grads = self.model.train_for_epochs(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        return (self.num_train_samples, model_params), (bytes_w, comp, bytes_r), grads

    def train_for_iters(self, num_iters=1, batch_size=10):
        """ local training based on iterations

        Return:
            1: num_train_samples: number of samples used in training
            1: model_params: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        """

        bytes_w = self.model.size
        model_params, comp = self.model.train_for_iters(self.train_data, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_train_samples, model_params), (bytes_w, comp, bytes_r)

    def evaluate(self, mode):
        """ evaluates current model on local data based on mode
        if mode='train', train_data is used else test_data
        
        Return:
            tot_correct: int: total #correct predictions
            loss: int: loss function value
            num_train_samples: int: number of training samples
        """
        data = None
        num_samples = None
        if mode == 'train':
            data = self.train_data
            num_samples = self.num_train_samples
        elif mode == 'test':
            data = self.test_data
            num_samples = self.num_test_samples
        else:
            logger.error(f"incorrect mode: {mode}")
        
        tot_correct, loss = self.model.evaluate(data)
        return tot_correct, loss, num_samples
