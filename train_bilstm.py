# -*- coding: utf-8 -*-
# training the model.
# process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import tensorflow as tf
import numpy as np
from data_util_zhihu import load_data, create_voabulary  # ,create_voabulary_label
from tflearn.data_utils import pad_sequences  # to_categorical
import os
import word2vec

import utils.misc_utils as utils
from model.BiLstm import BiLstm
from utils import model_utils
from utils.data_utils import read_all_feature_data, output_labels_v2, output_labels_v3
from utils.feature_utils import Features
from utils.config_utils import Config

cfg = Config()

np.random.seed(cfg.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cfg.gpu)

flags = tf.app.flags
flags.DEFINE_string("mode", "train", "type of operation [train, val, test, pred]")
flags.DEFINE_string("log_dir", "log", "path to save model")
FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt_dir", "biLstm_text_relation_checkpoint/", "checkpoint location for the model")
flags.DEFINE_string("word2vec_model_path", "zhihu-word2vec.bin-100", "word2vec's vocabulary and vectors")


# TODO Fix it
def main(_):
    ####################################################################################
    feats = Features()

    # hyper params
    hparam = tf.contrib.training.HParams(
        num_classes=1999,  # number of label
        model=cfg.model,
        norm=True,  # use batch norm
        seed=cfg.seed,
        batch_norm_decay=0.9,
        hidden_size=[1024, 512],
        cross_layer_sizes=[128, 128],
        k=16,  # multi_features embedding dim
        single_k=16,  # single_features embedding dim
        sequence_length=100,  # max sentence length
        embed_size=100,  # embedding size
        cross_hash_num=int(5e6),
        single_hash_num=int(5e6),
        multi_hash_num=int(1e6),
        batch_size=1024,
        infer_batch_size=2 ** 14,
        optimizer="adam",
        dropout=0,
        kv_batch_num=20,
        learning_rate=0.01,
        decay_steps=12000,  # how many steps before decay learning rate
        decay_rate=0.9,  # Rate of decay for learning rate
        num_display_steps=1000,  # every number of steps to display results
        num_save_steps=1000,  # every number of steps to save model
        num_eval_steps=1000,  # every number of steps to evaluate model
        epoch=20,  # train epoch
        metric='softmax_loss',
        activation=['relu', 'relu', 'relu'],
        init_method='tnormal',
        cross_activation='relu',
        init_value=0.001,
        l2_lambda=0.0001,
        single_features=None,
        cross_features=None,
        multi_features=feats.multi_features,
        dense_features=feats.dense_features,
        kv_features=None,
        label=feats.label_features,
        label_dim=1,  # output label dim (gender - 1, age - 4, age_all - 10)
        label_name='gender',
        model_name=cfg.model,
        checkpoint_dir=os.path.join(cfg.data_path, FLAGS.log_dir)
    )
    utils.print_hparams(hparam)

    ####################################################################################
    if FLAGS.mode == 'train':
        # read train data
        train_log = read_all_feature_data(feats, label_name=hparam.label_name)

        # build model
        model = model_utils.build_model(hparam)

        # train model
        model.train(train_log, None)

        # read test data
        test_log = read_all_feature_data(feats, mode='test', label_name=hparam.label_name)

        # infer model
        preds = model.infer(test_log)  # shape: [length, 20]

        if hparam.label_name == 'age':
            _ = output_labels_v2(test_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'preds.csv'))
        elif hparam.label_name == 'gender':
            _ = output_labels_v3(test_log, preds, pred_path=os.path.join(cfg.data_path, FLAGS.log_dir, 'preds.csv'))

    ####################################################################################
    elif FLAGS.mode == 'val':
        # read data
        train_log, val_log = read_all_feature_data(feats, mode='val', label_name=hparam.label_name)

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
    # 1.load data(X:list of lint,y:int).
    # if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    # else:
    if 1 == 1:
        # 1.  get vocabulary of X and label.
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',
                                                                        word2vec_model_path=FLAGS.word2vec_model_path,
                                                                        name_scope="biLstmTextRelation")
        vocab_size = len(vocabulary_word2index)
        print("rnn_model.vocab_size:", vocab_size)
        # vocabulary_word2index_label,vocabulary_index2word_label = create_voabulary_label(name_scope="biLstmTextRelation")
        vocabulary_word2index_label = {'1': 1, '0': 0}
        vocabulary_index2word_label = {0: '0', 1: '1'}
        train, test, _ = load_data(vocabulary_word2index, vocabulary_word2index_label, valid_portion=0.005,
                                   training_data_path=FLAGS.traning_data_path)
        # train, test, _ =  load_data_multilabel_new_twoCNN(vocabulary_word2index, vocabulary_word2index_label,multi_label_flag=False,traning_data_path=FLAGS.traning_data_path) #,traning_data_path=FLAGS.traning_data_path
        # train, test, _ =  load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,multi_label_flag=False,traning_data_path=FLAGS.traning_data_path) #,traning_data_path=FLAGS.traning_data_path
        trainX, trainY = train
        testX, testY = test
        # 2.Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
        ###############################################################################################
        # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        ###############################################################################################
        print("trainX[0]:", trainX[0])  # ;print("trainY[0]:", trainY[0])
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        biLstmTR = BiLstm(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, biLstmTR,
                                                 word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(biLstmTR.epoch_step)

        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, hparam.epoch):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])  # ;print("trainY[start:end]:",trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([biLstmTR.loss_val, biLstmTR.accuracy, biLstmTR.train_op],
                                                  feed_dict={biLstmTR.input_x: trainX[start:end],
                                                             biLstmTR.input_y: trainY[start:end]
                                                      ,
                                                             biLstmTR.dropout_keep_prob: 1.0})  # curr_acc--->TextCNN.accuracy -->,textRNN.dropout_keep_prob:1
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 500 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, loss / float(counter),
                        acc / float(counter)))  # tTrain Accuracy:%.3f---》acc/float(counter)
            # epoch increment
            print("going to increment epoch counter....")
            sess.run(biLstmTR.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, biLstmTR, testX, testY, batch_size, vocabulary_index2word_label)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                if not os.path.exists(FLAGS.ckpt_dir):
                    os.mkdir(FLAGS.ckpt_dir)
                saver.save(sess, save_path, global_step=epoch)



def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textRNN, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")




if __name__ == "__main__":
    tf.app.run()
