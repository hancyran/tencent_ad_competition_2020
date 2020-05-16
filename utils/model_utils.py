from model import CIN, xdeepfm, BiLstm, AttBiLstm
import tensorflow as tf


def build_model(hparams):
    tf.reset_default_graph()
    if hparams.model == 'CIN':
        model = CIN.Model(hparams)
    elif hparams.model == 'BiLstm':
        model = BiLstm.Model(hparams)
    elif hparams.model == 'AttBiLstm':
        model = AttBiLstm.Model(hparams)
    elif hparams.model == 'bert':
        model = AttBiLstm.Model(hparams)
    else:
        raise Exception('[!] No Such Type of Model')

    config_proto = tf.ConfigProto(log_device_placement=0, allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    model.set_Session(sess)

    return model
