import numpy as np
from tqdm import trange
from loguru import logger
import tensorflow as tf

from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss, self.predictions = self.create_model(optimizer)
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
        """ CNN Model """
        features = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='labels')
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
        conv1 = tf.compat.v1.layers.conv2d(
          inputs=input_layer,
          filters=16,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.compat.v1.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
        dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

        logits = tf.compat.v1.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = tf.math.count_nonzero(tf.equal(labels, predictions["classes"]))

        return features, labels, train_op, grads, eval_metric_ops, loss, predictions['classes']


    def set_params(self, model_params=None):
        """ set model parameters """
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)
        logger.debug('set params in model')

    def get_params(self):
        """ get model parameters """
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        """ get model gradient """
        #grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], 
                                                   self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads


    def train_for_epochs(self, data, num_epochs=1, batch_size=32):
        """ trains model on given data based on num_epochs """
        with self.graph.as_default():
            _, grads = self.get_gradients(data, 610) # Ignore the hardcoding, it's not used anywhere

        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        model_params = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return model_params, comp, grads

    # TODO: remove this function, it's not used anywhere
    # def solve_sgd(self, mini_batch_data):
    #     with self.graph.as_default():
    #         grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
    #                                        feed_dict={self.features: mini_batch_data[0],
    #                                                   self.labels: mini_batch_data[1]})
    #     weights = self.get_params()
    #     return grads, loss, weights

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss


    def evaluate(self, data):
        """
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        """
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
