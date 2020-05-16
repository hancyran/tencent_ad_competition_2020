import json
import numpy as np
import pandas as pd
from utils import model_utils
import tensorflow as tf
import utils.misc_utils as utils
import os
import gc
from sklearn import metrics
from sklearn import preprocessing
import random

from utils.data_utils import read_all_feature_data, output_labels_v3, output_labels_v2
from utils.feature_utils import Features

from utils.config_utils import Config

cfg = Config()

np.random.seed(cfg.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg.gpu)

flags = tf.app.flags
flags.DEFINE_string("mode", "train", "type of operation [train, val, test, pred]")
flags.DEFINE_string("log_dir", "log", "path to save model")
FLAGS = flags.FLAGS

"""
Main Function

1) Remove Outliers
2) Train Separately (gender+age / age(gender=1) / gender(age=3))
3) Apply to BERT4Rec with text classification

"""


def main(_):
    ####################################################################################
    feats = Features()

    # hyper params
    hparam = tf.contrib.training.HParams(
        model=cfg.model,
        norm=True,  # use batch norm
        seed=cfg.seed,
        batch_norm_decay=0.9,
        hidden_size=[1024, 512],
        cross_layer_sizes=[128, 128],
        k=16,  # multi_features embedding dim
        single_k=16,  # single_features embedding dim
        max_length=100,  # hash length
        cross_hash_num=int(5e6),
        single_hash_num=int(5e6),
        multi_hash_num=int(1e6),
        batch_size=1024,
        infer_batch_size=2 ** 14,
        optimizer="adam",
        dropout=0,
        kv_batch_num=20,
        learning_rate=0.00005,
        num_display_steps=100,  # every number of steps to display results
        num_save_steps=1000,  # every number of steps to save model
        num_eval_steps=2000,  # every number of steps to evaluate model
        epoch=10,  # train epoch
        metric='softmax_loss',
        activation=['relu', 'relu', 'relu'],
        init_method='tnormal',
        cross_activation='relu',
        init_value=0.001,
        single_features=None,
        cross_features=None,
        multi_features=feats.multi_features,
        dense_features=feats.dense_features,
        kv_features=None,
        label=feats.label_features,
        label_dim=4,  # output label dim (gender - 1, age - 4, age_all - 10)
        label_name='age',
        model_name=cfg.model,
        checkpoint_dir=os.path.join(cfg.data_path, FLAGS.log_dir)
    )
    utils.print_hparams(hparam)

    ####################################################################################

    if FLAGS.mode == 'train':
        # read data
        train_log = read_all_feature_data(feats, label_name=hparam.label_name)

        # build model
        model = model_utils.build_model(hparam)

        # train model
        model.train(train_log, None)

    ####################################################################################
    elif FLAGS.mode == 'test':
        # read data
        test_log = read_all_feature_data(feats, mode='test')

        # build model
        model = model_utils.build_model(hparam)

        # infer model
        preds = model.infer(test_log)  # shape: [length, 20]

        if hparam.label_name == 'age':
            _ = output_labels_v2(test_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'preds.csv'))
        elif hparam.label_name == 'gender':
            _ = output_labels_v3(test_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'preds.csv'))

        # K_fold = []
        # for i in range(5):
        #     if i == 4:
        #         tmp = index
        #     else:
        #         tmp = random.sample(index, int(1.0 / 5 * train.shape[0]))
        #     index = index - set(tmp)
        #     print("Number:", len(tmp))
        #     K_fold.append(tmp)
        #
        # train_preds = np.zeros(len(train))
        # test_preds = np.zeros(len(test))
        # scores = []
        # train['gold'] = True
        # for i in range(5):
        #     print("Fold", i)
        #     dev_index = K_fold[i]
        #     train_index = []
        #     for j in range(5):
        #         if j != i:
        #             train_index += K_fold[j]
        #     for k in range(2):
        #         model = model_utils.build_model(hparam)
        #         score = model.train(train.loc[train_index], train.loc[dev_index])
        #         scores.append(score)
        #         train_preds[list(dev_index)] += model.infer(train.loc[list(dev_index)]) / 2
        #         test_preds += model.infer(test) / 10
        #         print(np.mean((np.exp(test_preds * 10 / (i * 2 + k + 1)) - 1)))
        #     try:
        #         del model
        #         gc.collect()
        #     except:
        #         pass
        # train_preds = np.exp(train_preds) - 1
        # test_preds = np.exp(test_preds) - 1

    ####################################################################################
    elif FLAGS.mode == 'val':
        # read data
        train_log, val_log = read_all_feature_data(feats, mode='val')

        # build model
        model = model_utils.build_model(hparam)

        # train model
        model.train(train_log, None, is_val=True)

        # infer model
        preds = model.infer(val_log)  # shape: [length, 20]

        if hparam.label_name == 'age':
            val_log = output_labels_v2(
                val_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'val_preds.csv'), is_train=True)
        elif hparam.label_name == 'gender':
            val_log = output_labels_v3(
                val_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'val_preds.csv'), is_train=True)

        # print results
        age_acc = sum((val_log.age == val_log.predicted_age).astype(np.int)) / len(val_log)
        gender_acc = sum((val_log.gender == val_log.predicted_gender).astype(np.int)) / len(val_log)

        print("Final Age Accuracy: %.4f" % age_acc)
        print("Final Gender Accuracy: %.4f" % gender_acc)


# ####################################################################################

if __name__ == '__main__':
    tf.app.run()
