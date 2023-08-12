import numpy as np
from loguru import logger
import tensorflow as tf

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    """
    Assumes that images are 28px by 28px
    """
    def __init__(self, num_classes, optimizer, seed=1):
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss, self.pred = self.create_model(optimizer)
            self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            metadata = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.compat.v1.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        """ Logistic Regression Model """
        features = tf.compat.v1.placeholder(tf.float32, shape=[None, 60], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None,], name='labels')
        logits = tf.compat.v1.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.5 * (0.001)))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = tf.math.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss, predictions["classes"]

    def set_params(self, model_params=None):
        """ set model parameters """
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.compat.v1.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)
        logger.debug(f'set params in model: {self}')

    def get_params(self):
        """ get model parameters """
        with self.graph.as_default():
            model_params = self.sess.run(tf.compat.v1.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        """ get model gradient """
        grads = np.zeros(model_len)
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

    def train_for_iters(self, data, num_iters=1, batch_size=32):
        """ trains model on given data based on num_iters """

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        model_params = self.get_params()
        comp = 0
        return model_params, comp

    def evaluate(self, data):
        """
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        """
        with self.graph.as_default():
            tot_correct, loss, pred = self.sess.run([self.eval_metric_ops, self.loss, self.pred], 
                                                    feed_dict={self.features: data['x'], 
                                                               self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
