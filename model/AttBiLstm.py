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
        1) Add Attention Layer TODO Test
        
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

        # attention layer
        output_att = self._attention(output_rnn_pooled)

        # output fc
        self.W_projection = tf.get_variable("W_projection", shape=[hparams.hidden_size * 2, hparams.num_classes],
                                            initializer=self.initializer)  # [embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection", shape=[hparams.num_classes])  # [label_size]

        self.output_logits = tf.matmul(output_att, self.W_projection) + self.b_projection

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

    # =======================================================================================================================
    # =======================================================================================================================

    def _attention(self, H):
        """
        Build Attention Layer

        """
        hiddenSize = self.hparams.hidden_size[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.hparams.label_dim])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.hparams.label_dim, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.hparams.dropout)

        return output
