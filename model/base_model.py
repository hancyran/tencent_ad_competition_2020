"""define base class model"""
import abc

import tensorflow as tf
import os
import utils.misc_utils as utils
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import time
from tqdm import tqdm

__all__ = ["BaseModel"]


class BaseModel(object):
    def __init__(self, hparams, scope=None):
        tf.set_random_seed(hparams.seed)
        self.hparams = hparams

    @abc.abstractmethod
    def _build_graph(self, hparams):
        """Subclass must implement this."""
        pass

    def _get_initializer(self, hparams):
        """
        Variable Initializer

        """
        if hparams.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'uniform':
            return tf.random_uniform_initializer(-hparams.init_value, hparams.init_value)
        elif hparams.init_method == 'normal':
            return tf.random_normal_initializer(stddev=hparams.init_value)
        elif hparams.init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif hparams.init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif hparams.init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=False)
        elif hparams.init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer( \
                factor=2.0, mode='FAN_AVG', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=hparams.init_value)

    def _build_train_opt(self, hparams):
        """
        Optimizer

        """

        def train_opt(hparams):
            if hparams.optimizer == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adagrad':
                train_step = tf.train.AdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'sgd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'adam':
                train_step = tf.train.AdamOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'ftrl':
                train_step = tf.train.FtrlOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'gd':
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'padagrad':
                train_step = tf.train.ProximalAdagradOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'pgd':
                train_step = tf.train.ProximalGradientDescentOptimizer( \
                    hparams.learning_rate)
            elif hparams.optimizer == 'rmsprop':
                train_step = tf.train.RMSPropOptimizer( \
                    hparams.learning_rate)
            else:
                train_step = tf.train.GradientDescentOptimizer( \
                    hparams.learning_rate)
            return train_step

        train_step = train_opt(hparams)
        return train_step

    def _active_layer(self, logit, scope, activation, layer_idx):
        logit = self._activate(logit, activation)
        return logit

    def _activate(self, logit, activation):
        """
        Activation Layer

        """
        if activation == 'sigmoid':
            return tf.nn.sigmoid(logit)
        elif activation == 'softmax':
            return tf.nn.softmax(logit)
        elif activation == 'relu':
            return tf.nn.relu(logit)
        elif activation == 'tanh':
            return tf.nn.tanh(logit)
        elif activation == 'elu':
            return tf.nn.elu(logit)
        elif activation == 'identity':
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, layer_idx):
        """
        Dropout Layer

        """
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        return logit

    def batch_norm_layer(self, x, train_phase, scope_bn):
        """
        Norm Layer

        """
        z = tf.cond(train_phase, lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True,
                                                    updates_collections=None, is_training=True, reuse=None,
                                                    trainable=True, scope=scope_bn),
                    lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True,
                                       updates_collections=None, is_training=False, reuse=True, trainable=False,
                                       scope=scope_bn))
        return z

    def optimizer(self, hparams):
        """
        Build Optimizer

        """
        opt = self._build_train_opt(hparams)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        self.grad_norm = gradient_norm
        self.update = opt.apply_gradients(zip(clipped_grads, params))

    # =======================================================================================================================
    # =======================================================================================================================

    @property
    def model_dir(self):
        return "{}".format(self.hparams.model_name)

    def save(self, checkpoint_dir, step):
        model_name = "RAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # =======================================================================================================================
    # =======================================================================================================================

    def train(self, train, dev, is_val=False):
        """
        Train Model

        """
        # init
        hparams = self.hparams
        sess = self.sess

        # fetch data
        if hparams.single_features is not None:
            train_single_features = train[hparams.single_features].values
        if hparams.label is not None:
            train_label = train[hparams.label].values
        if hparams.multi_features is not None:
            train_multi_features = train[hparams.multi_features].values
        if hparams.dense_features is not None:
            train_dense_features = train[hparams.dense_features].values
        if hparams.kv_features is not None:
            train_kv_features = train[hparams.kv_features].values
        if hparams.cross_features is not None:
            train_cross_features = train[hparams.cross_features].values

        # start training
        print('Start Train')
        for epoch in range(hparams.epoch):
            info = {}
            info['loss'] = []
            info['norm'] = []
            start_time = time.time()

            # shuffle
            train.sample(frac=1)
            # each batch
            for idx in range(len(train) // hparams.batch_size + 3):
                if idx * hparams.batch_size >= len(train):
                    T = (time.time() - start_time)
                    break
                feed_dic = {}

                # single_features
                if hparams.single_features is not None:
                    single_batch = \
                        train_single_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(train))]
                    single_batch = utils.hash_single_batch(single_batch, hparams)
                    feed_dic[self.single_features] = single_batch

                # multi_features
                if hparams.multi_features is not None:
                    multi_batch = \
                        train_multi_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(train))]
                    multi_batch, multi_weights = utils.hash_multi_batch(multi_batch, hparams)
                    feed_dic[self.multi_features] = multi_batch
                    feed_dic[self.multi_weights] = multi_weights

                # dense_features
                if hparams.dense_features is not None:
                    feed_dic[self.dense_features] = \
                        train_dense_features[idx * hparams.batch_size: min((idx + 1) * hparams.batch_size, len(train))]

                # kv_features
                if hparams.kv_features is not None:
                    feed_dic[self.kv_features] = \
                        train_kv_features[idx * hparams.batch_size: min((idx + 1) * hparams.batch_size, len(train))]

                # cross_features
                if hparams.cross_features is not None:
                    cross_batch = \
                        train_cross_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(train))]
                    cross_batch = utils.hash_single_batch(cross_batch, hparams)
                    feed_dic[self.cross_features] = cross_batch

                # label
                label = train_label[idx * hparams.batch_size: min((idx + 1) * hparams.batch_size, len(train))]
                feed_dic[self.label] = label

                # setting
                feed_dic[self.use_norm] = True

                # train a batch
                loss, _, norm, preds = sess.run([self.loss, self.update, self.grad_norm, self.fake_label],
                                                feed_dict=feed_dic)

                info['loss'].append(loss)
                info['norm'].append(norm)

                # show results
                if (idx + 1) % hparams.num_display_steps == 0:
                    info['learning_rate'] = hparams.learning_rate
                    info["train_ppl"] = np.mean(info['loss'])
                    info["avg_grad_norm"] = np.mean(info['norm'])

                    # accuracy
                    # tmp_train = output_labels(train, preds)
                    # age_acc = sum((tmp_train.age == tmp_train.predicted_age).astype(np.int)) / len(tmp_train)
                    # gender_acc = sum((tmp_train.gender == tmp_train.predicted_gender).astype(np.int)) / len(tmp_train)
                    # info["age_acc"] = age_acc
                    # info["gender_acc"] = gender_acc

                    # print
                    utils.print_step_info("  ", epoch, idx + 1, info)

                    # clear
                    del info
                    info = {}
                    info['loss'] = []
                    info['norm'] = []

                # save model
                if (idx + 1) % hparams.num_save_steps == 0:
                    self.save(hparams.checkpoint_dir, (idx + 1))

                # evaluate model
                # if (idx + 1) % hparams.num_eval_steps == 0:
                #     self.evaluate()

        print('Finish Train')

    # =======================================================================================================================
    # =======================================================================================================================

    def infer(self, dev):
        """
        Infer with Pretrained Model

        """
        # init 
        hparams = self.hparams
        sess = self.sess

        preds = []
        a = hparams.batch_size
        hparams.batch_size = hparams.infer_batch_size

        # fetch data
        if hparams.single_features is not None:
            dev_single_features = dev[hparams.single_features].values
        if hparams.multi_features is not None:
            dev_multi_features = dev[hparams.multi_features].values
        if hparams.dense_features is not None:
            dev_dense_features = dev[hparams.dense_features].values
        if hparams.kv_features is not None:
            dev_kv_features = dev[hparams.kv_features].values
        if hparams.cross_features is not None:
            dev_cross_features = dev[hparams.cross_features].values

        # start inference
        print('Start Infer')
        for idx in tqdm(range(len(dev) // hparams.batch_size + 1), total=len(dev) // hparams.batch_size + 1):

            single_batch = dev_multi_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(dev))]
            if len(single_batch) == 0:
                break

            feed_dic = {}

            # single_features
            if hparams.single_features is not None:
                single_batch = \
                    dev_single_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(dev))]
                single_batch = utils.hash_single_batch(single_batch, hparams)
                feed_dic[self.single_features] = single_batch

            # multi_features
            if hparams.multi_features is not None:
                multi_batch = \
                    dev_multi_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(dev))]
                multi_batch, multi_weights = utils.hash_multi_batch(multi_batch, hparams)
                feed_dic[self.multi_features] = multi_batch
                feed_dic[self.multi_weights] = multi_weights

            # dense_features
            if hparams.dense_features is not None:
                feed_dic[self.dense_features] = \
                    dev_dense_features[idx * hparams.batch_size: min((idx + 1) * hparams.batch_size, len(dev))]

            # kv_features
            if hparams.kv_features is not None:
                feed_dic[self.kv_features] = \
                    dev_kv_features[idx * hparams.batch_size: min((idx + 1) * hparams.batch_size, len(dev))]

            # cross_features
            if hparams.cross_features is not None:
                cross_batch = \
                    dev_cross_features[idx * hparams.batch_size:min((idx + 1) * hparams.batch_size, len(dev))]
                cross_batch = utils.hash_single_batch(cross_batch, hparams)
                feed_dic[self.cross_features] = cross_batch

            # setting
            feed_dic[self.use_norm] = False

            # infer a batch
            pred = sess.run(self.fake_label, feed_dict=feed_dic)

            preds.append(pred)

        print('Finish Infer')
        preds = np.concatenate(preds)
        # dev['temp'] = preds
        hparams.batch_size = a

        return preds
