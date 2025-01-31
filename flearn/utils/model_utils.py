import os
import json
import numpy as np
# import Image
from PIL import Image


def load_image(img_name):
    IMAGE_SIZE = 84

    IMAGES_DIR = os.path.join('data', 'celeba', 'data', 'raw', 'img_align_celeba')
    img = Image.open(os.path.join(IMAGES_DIR, img_name))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    return np.array(img)


def process_x(raw_x_batch):
    x_batch = [load_image(i) for i in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    return raw_y_batch


def batch_data(data, batch_size):
    """
    data: dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    # TODO: discard last batch if its size < batch_size
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def batch_data_celeba(data, batch_size):

    data_x_name = data['x']
    data_y_name = data['y']

    raw_x = np.asarray(process_x(data_x_name))
    raw_y = np.asarray(process_y(data_y_name))

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(raw_x)
    np.random.set_state(rng_state)
    np.random.shuffle(raw_y)

    # loop through mini-batches
    for i in range(0, len(raw_y), batch_size):
        batched_x = raw_x[i:i + batch_size]
        batched_y = raw_y[i:i + batch_size]
        yield (batched_x, batched_y)


def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)


def read_data(train_data_dir, test_data_dir):
    """ 
    parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class Metrics(object):
    def __init__(self, clients, hyper_params):
        self.hyper_params = hyper_params
        num_rounds = hyper_params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []
        self.train_accuracies = []
        self.debug = {}

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        for key, val in self.hyper_params.items(): metrics[key] = val;
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics['debug'] = self.debug
        metrics_filename = os.path.join('results', self.hyper_params['dataset'], 'metrics_{}_{}_{}_{}_{}_{}.json'.format(self.hyper_params['clientsel_algo'], 
                                                                                                       self.hyper_params['seed'], 
                                                                                                       self.hyper_params['trainer'], 
                                                                                                       self.hyper_params['learning_rate'], 
                                                                                                       self.hyper_params['num_epochs'], 
                                                                                                       self.hyper_params['mu']))

        print(f'Writing metrics to file: {metrics_filename}')
        os.makedirs(os.path.join('results', self.hyper_params['dataset']), exist_ok=True)
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f)
