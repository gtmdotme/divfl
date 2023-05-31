import os
import numpy as np
import tensorflow as tf
from tqdm import trange


from flearn.utils.tf_utils import graph_size, process_grad
from flearn.utils.model_utils import batch_data, batch_data_celeba, process_x, process_y

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')


class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, \
                self.loss = self.create_model(optimizer)
            self.saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            metadata = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.compat.v1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        out = input_ph
        for _ in range(4):
            out = tf.compat.v1.layers.conv2d(out, 32, 3, padding='same')
            out = tf.compat.v1.layers.batch_normalization(out, training=True)
            out = tf.compat.v1.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.compat.v1.layers.dense(out, self.num_classes)
        label_ph = tf.compat.v1.placeholder(tf.int64, shape=(None,))
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=label_ph, logits=logits)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = tf.math.count_nonzero(tf.equal(label_ph, tf.argmax(input=logits, axis=1)))

        return input_ph, label_ph, train_op, grads, eval_metric_ops, loss



    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        num_samples = len(data['y']) # Need model len
        with self.graph.as_default():
            grads = self.sess.run(self.grads,
                                        feed_dict={self.features: process_x(data['x']), self.labels: process_y(data['y'])})
            grads = process_grad(grads)

        return num_samples, grads


    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''

        with self.graph.as_default():
            _, grads = self.get_gradients(data, 610) # Ignore the hardcoding, it's not used anywhere

        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data_celeba(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp, grads

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})
        weights = self.get_params()
        return grads, loss, weights

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: process_x(data['x']),
                                                         self.labels: process_y(data['y'])})
        return tot_correct, loss


    def close(self):
        self.sess.close()