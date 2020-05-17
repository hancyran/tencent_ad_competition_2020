import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

from model.base_model import BaseModel
import utils.misc_utils as utils


class Model(BaseModel):
    def __init__(self, hparams):
        self.hparams = hparams
        if hparams.metric in ['softmax_loss']:
            self.best_score = 100000
        else:
            self.best_score = 0
        self.build_graph(hparams)
        self.optimizer(hparams)

        params = tf.trainable_variables()
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))

    def set_Session(self, sess):
        self.sess = sess

    # =======================================================================================================================
    # =======================================================================================================================

    def build_graph(self, hparams):
        """
        Build BiLstm Model

        New Features:

        """
        # init
        self.initializer = self._get_initializer(hparams)

        #############################################################################
        # Data placeholder
        ###############################################################################
        self.label = tf.placeholder(shape=(None, hparams.label_dim), dtype=tf.float32)
        self.input = tf.placeholder(tf.int32, [None, hparams.sequence_length], name="input")

        # embedding
        self.embed_dict = tf.get_variable("embed_dict", shape=[hparams.vocab_size, hparams.embed_size],
                                          initializer=self.initializer)
        self.embed_words = tf.nn.embedding_lookup(self.embed_dict, self.input)

        ###############################################################################
        # Forward
        ###############################################################################
        # Bi-LSTM layer
        # forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(hparams.hidden_size)
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=hparams.dropout_keep_prob)
        # backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(hparams.hidden_size)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=hparams.dropout_keep_prob)
        # dynamic rnn
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embed_words, dtype=tf.float32)

        # Pool
        output_rnn = tf.concat(outputs, axis=2)
        output_rnn_pooled = tf.reduce_mean(output_rnn, axis=1)

        # output fc
        self.W_projection = tf.get_variable("W_projection", shape=[hparams.hidden_size * 2, hparams.num_classes],
                                            initializer=self.initializer)  # [embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection", shape=[hparams.num_classes])  # [label_size]

        self.output_logits = tf.matmul(output_rnn_pooled, self.W_projection) + self.b_projection

        if hparams.label_dim > 2:
            self.fake_label = self.output_logits
        else:
            self.fake_label = tf.nn.sigmoid(self.output_logits)

        ###############################################################################
        # Loss - Softmax/Binary Cross Entropy
        ###############################################################################
        if hparams.label_dim > 2:
            self.normal_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.output_logits))
        else:
            self.normal_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.output_logits))

        self.l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * hparams.l2_lambda

        self.loss = self.l2_loss + self.normal_loss

        self.saver = tf.train.Saver()
