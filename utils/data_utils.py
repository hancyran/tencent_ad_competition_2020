import copy
import json
import os
import random

import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.outlier_utils import Outliers
from utils.config_utils import Config

cfg = Config()

random.seed(cfg.seed)


###############################################################################
# Preprocess Data
###############################################################################
def preprocess(is_train=True, is_split=False, log_path=None):
    """
    Preprocess to obtain training dataset

    """
    if is_train:
        if not log_path or not os.path.exists(os.path.join(cfg.data_path, log_path)):
            print('Read Raw Data')
            users, ads, log = read_train_raw_data()

            print('Combine User Log')
            user_log = combine_log(ads, log, users, is_train=is_train, sort_type='time', save_path=log_path)

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log)
        else:
            user_log = pd.read_pickle(os.path.join(cfg.data_path, log_path))

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log)

        if is_split:
            print('Split Dataset')
            train_log, val_log = split_dataset(user_log)

            return train_log, val_log
        else:
            return user_log

    else:
        if not log_path or not os.path.exists(os.path.join(cfg.data_path, log_path)):
            print('Read Raw Data')
            ads, log = read_test_raw_data()

            print('Combine User Log')
            user_log = combine_log(ads, log, is_train=is_train, sort_type='time', save_path=log_path)

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log, is_train=is_train)
        else:
            user_log = pd.read_pickle(os.path.join(cfg.data_path, log_path))

            print('Remove Outliers')
            user_log = remove_outlier_id(user_log, is_train=is_train)

        return user_log


def read_train_raw_data(root_path='train_preliminary', add_click=False):
    """
    Read Train Raw Data

    """
    users = pd.read_csv(os.path.join(cfg.data_path, root_path, 'user.csv'))
    ads = pd.read_csv(os.path.join(cfg.data_path, root_path, 'ad.csv'))
    log = pd.read_csv(os.path.join(cfg.data_path, root_path, 'click_log.csv'))

    # turn to null value
    ads.replace('\\N', 0, inplace=True)
    ads.product_id = ads.product_id.astype(np.int64)
    ads.industry = ads.industry.astype(np.int64)

    if add_click:
        log = process_log(log)

    return users, ads, log


def read_test_raw_data(root_path='test', add_click=False):
    """
    Read Test Raw Data

    """
    ads = pd.read_csv(os.path.join(cfg.data_path, root_path, 'ad.csv'))
    log = pd.read_csv(os.path.join(cfg.data_path, root_path, 'click_log.csv'))

    # turn to null value
    ads.replace('\\N', 0, inplace=True)
    ads.product_id = ads.product_id.astype(np.int64)
    ads.industry = ads.industry.astype(np.int64)

    if add_click:
        log = process_log(log)

    return ads, log


def process_log(log):
    """
    Append click_times*row

    """
    creative_id_list = [','.join([str(ids)] * time) for ids, time in
                        zip(log['creative_id'].values, log['click_times'].values)]
    ad_id_list = [','.join([str(ids)] * time) for ids, time in zip(log['ad_id'].values, log['click_times'].values)]
    product_id_list = [','.join([str(ids)] * time) for ids, time in
                       zip(log['product_id'].values, log['click_times'].values)]
    product_category_list = [','.join([str(ids)] * time) for ids, time in
                             zip(log['product_category'].values, log['click_times'].values)]
    advertiser_id_list = [','.join([str(ids)] * time) for ids, time in
                          zip(log['advertiser_id'].values, log['click_times'].values)]
    industry_list = [','.join([str(ids)] * time) for ids, time in
                     zip(log['industry'].values, log['click_times'].values)]

    log['creative_id'] = creative_id_list
    log['ad_id'] = ad_id_list
    log['product_id'] = product_id_list
    log['product_category'] = product_category_list
    log['advertiser_id'] = advertiser_id_list
    log['industry'] = industry_list

    del creative_id_list, ad_id_list, product_id_list, product_category_list, advertiser_id_list, industry_list
    gc.collect()

    return log


def combine_log(ads, log, users=None, is_train=True, save_path=None, sort_type='time'):
    """
    Combine Log into User-Primary Log

    """
    # merge df
    if is_train:
        merged_log = pd.merge(log, users, on='user_id')
        merged_log = pd.merge(merged_log, ads, on='creative_id')
    else:
        merged_log = pd.merge(log, ads, on='creative_id')

    def combine_log(merged_log):
        # combine id into one sequence
        def combine_id(x):
            col = list(set(x) - {0})
            col.sort(key=list(x).index)
            return ','.join([str(i) for i in col])

        def combine_id_origin(x):
            col = list(x)
            try:
                col.remove(0)
            except:
                pass
            col.sort(key=list(x).index)
            return ','.join([str(i) for i in col])

        # creative_id
        combined_log = merged_log[['user_id', 'creative_id']].groupby(['user_id']).agg(
            {'creative_id': combine_id_origin})
        user_log = combined_log.reset_index()
        # ad_id
        combined_log = merged_log[['user_id', 'ad_id']].groupby(['user_id']).agg(
            {'ad_id': combine_id_origin})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # product_id
        combined_log = merged_log[['user_id', 'product_id']].groupby(['user_id']).agg(
            {'product_id': combine_id_origin})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # product_category
        combined_log = merged_log[['user_id', 'product_category']].groupby(['user_id']).agg(
            {'product_category': combine_id_origin})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # advertiser_id
        combined_log = merged_log[['user_id', 'advertiser_id']].groupby(['user_id']).agg(
            {'advertiser_id': combine_id_origin})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # industry
        combined_log = merged_log[['user_id', 'industry']].groupby(['user_id']).agg(
            {'industry': combine_id_origin})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')

        return user_log

    if sort_type == 'time':
        # sort by time
        merged_log.sort_values(by='time', inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    elif sort_type == 'click_times':
        # count click times
        group = merged_log[['user_id', 'creative_id', 'click_times']].groupby(['user_id', 'creative_id']).sum()
        group.reset_index()

        # merge count times
        merged_log.drop(['click_times'], inplace=True, axis=1)
        merged_log = pd.merge(merged_log, group, on=['user_id', 'creative_id'])

        # sort by click times
        merged_log.sort_values(by='click_times', ascending=False, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    elif sort_type == 'both_combined':
        # sort by time
        merged_log.sort_values(by='time', ascending=True, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        time_log = combine_log(merged_log)

        # count click times
        group = merged_log[['user_id', 'creative_id', 'click_times']].groupby(['user_id', 'creative_id']).sum()
        group.reset_index()

        # merge count times
        merged_log.drop(['click_times'], inplace=True, axis=1)
        merged_log = pd.merge(merged_log, group, on=['user_id', 'creative_id'])

        # sort by click times
        merged_log.sort_values(by='click_times', ascending=False, inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        click_log = combine_log(merged_log)

        # merge both types
        user_log = pd.merge(time_log, click_log, on='user_id', suffixes=('_time', '_click'))

    elif sort_type == 'both_sorted':
        # sort by time(ascending) & click_times(descending)
        merged_log.sort_values(by=['time', 'click_times'], ascending=[True, False], inplace=True)
        merged_log.reset_index(inplace=True, drop=True)
        user_log = combine_log(merged_log)

    # merge labels
    if is_train:
        user_log = pd.merge(user_log, users, on=['user_id'])

    if save_path:
        user_log.to_pickle(os.path.join(cfg.data_path, save_path))

    return user_log


def combine_log_v2(ads, log, users=None, is_train=True, save_path=None):
    """
    Combine Log into User-Primary Log (Click + Time)

    """
    # merge df
    if is_train:
        merged_log = pd.merge(log, users, on='user_id')
        merged_log = pd.merge(merged_log, ads, on='creative_id')
        merged_log = process_log(merged_log)
    else:
        merged_log = pd.merge(log, ads, on='creative_id')
        merged_log = process_log(merged_log)

    def combine_log(merged_log):
        # combine id into one sequence
        def combine_id_time_remove_0(x):
            id_str = ','.join(x)
            id_list = id_str.split(',')
            while '0' in id_list:
                id_list.remove('0')
            random.shuffle(id_list)
            return ','.join(id_list)

        def combine_id_time(x):
            id_str = ','.join(x)
            id_list = id_str.split(',')
            random.shuffle(id_list)
            return ','.join(id_list)

        def combine_id_user(x):
            return ','.join(x)

        # creative_id
        combined_log = merged_log[['user_id', 'time', 'creative_id']].groupby(['user_id', 'time']).agg(
            {'creative_id': combine_id_time}).groupby(['user_id']).agg({'creative_id': combine_id_user})
        user_log = combined_log.reset_index()
        # ad_id
        combined_log = merged_log[['user_id', 'time', 'ad_id']].groupby(['user_id', 'time']).agg(
            {'ad_id': combine_id_time}).groupby(['user_id']).agg({'ad_id': combine_id_user})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # product_id
        combined_log = merged_log[['user_id', 'time', 'product_id']].groupby(['user_id', 'time']).agg(
            {'product_id': combine_id_time_remove_0}).groupby(['user_id']).agg({'product_id': combine_id_user})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # product_category
        combined_log = merged_log[['user_id', 'time', 'product_category']].groupby(['user_id', 'time']).agg(
            {'product_category': combine_id_time}).groupby(['user_id']).agg({'product_category': combine_id_user})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # advertiser_id
        combined_log = merged_log[['user_id', 'time', 'advertiser_id']].groupby(['user_id', 'time']).agg(
            {'advertiser_id': combine_id_time}).groupby(['user_id']).agg({'advertiser_id': combine_id_user})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')
        # industry
        combined_log = merged_log[['user_id', 'time', 'industry']].groupby(['user_id', 'time']).agg(
            {'industry': combine_id_time_remove_0}).groupby(['user_id']).agg({'industry': combine_id_user})
        user_log = pd.merge(user_log, combined_log.reset_index(), on='user_id')

        return user_log

    # sort by time
    merged_log.sort_values(by='time', inplace=True)
    merged_log.reset_index(inplace=True, drop=True)
    user_log = combine_log(merged_log)

    def remove_null(x):
        l = x.split(',')
        while '' in l:
            l.remove('')
        return ','.join(l)
    # remove comma
    user_log['product_id'] = user_log['product_id'].apply(lambda x: remove_null(x))
    user_log['industry'] = user_log['industry'].apply(lambda x: remove_null(x))

    # merge labels
    if is_train:
        user_log = pd.merge(user_log, users, on=['user_id'])

    if save_path:
        user_log.to_pickle(os.path.join(cfg.data_path, save_path))

    del ads, log, users, merged_log
    gc.collect()

    return user_log


def remove_outlier_id(log, is_train=True, save_path=None):
    """
    Remove Outliers with existing user_id

    """
    # fetch outlier ids
    outliers = Outliers()
    if is_train:
        outlier_ids = outliers.train_userid_outliers
    else:
        outlier_ids = outliers.test_userid_outliers

    # remove outliers
    for i in outlier_ids:
        log = log.drop(log[log.user_id == i].index)

    if save_path:
        log.to_pickle(os.path.join(cfg.data_path, save_path))

    return log


def split_dataset(log):
    """
    Split Data into Validation and Training Dataset

    """
    train_log, val_log = train_test_split(log, test_size=0.05, random_state=cfg.seed, shuffle=True)

    return train_log, val_log


###############################################################################
# Extend for Input
###############################################################################
def read_all_feature_data(feats, mode='train', label_name='age'):
    """
    Read All data (Dense Features + Multi Features)

    """
    if mode == 'train':
        if not os.path.exists(os.path.join(cfg.data_path, 'train_log_NN_v2.pkl')):
            train_log = preprocess(log_path='train_log_time_click_time_sequence.pkl')

            # add w2v features
            for multi_feat in feats.multi_features:
                dense_feats = [x for x in feats.dense_features if multi_feat and 'w2v' in x]
                print('read %s w2v embedding' % multi_feat)
                train_log = read_embedding_data(train_log, multi_feat, dense_feats,
                                                os.path.join(cfg.data_path, 'user_id_%s_test_w2v_128.pkl' % multi_feat))

            # add tfidf features
            for multi_feat in feats.multi_features:
                print('read %s tfidf embedding' % multi_feat)
                tfidf_df = pd.read_pickle(os.path.join(cfg.data_path, 'user_id_%s_test_tfidf.pkl' % multi_feat))
                train_log = pd.merge(train_log, tfidf_df, on='user_id')

            # process label
            print('read label')
            if label_name == 'age':
                train_log = read_labels_v2(train_log, feats.label_features)
            elif label_name == 'gender':
                train_log = read_labels_v3(train_log, feats.label_features)

            # save data
            print('save data')
            train_log.to_pickle(os.path.join(cfg.data_path, 'train_log_NN_v2.pkl'))

            train_log = train_log.fillna(0)
        else:
            print('read all data')
            train_log = pd.read_pickle(os.path.join(cfg.data_path, 'train_log_NN_v2.pkl'))
            train_log = train_log.fillna(0)

            print('read label')
            if label_name == 'age':
                train_log = read_labels_v2(train_log, feats.label_features)
            elif label_name == 'gender':
                train_log = read_labels_v3(train_log, feats.label_features)

        return train_log

    elif mode == 'test':
        if not os.path.exists(os.path.join(cfg.data_path, 'test_log_NN_v2.pkl')):
            test_log = preprocess(is_train=False, log_path='test_log_time_click_time_sequence.pkl')

            # add w2v features
            for multi_feat in feats.multi_features:
                dense_feats = [x for x in feats.dense_features if multi_feat and 'w2v' in x]
                print('read %s w2v embedding' % multi_feat)
                test_log = read_embedding_data(test_log, multi_feat, dense_feats,
                                               os.path.join(cfg.data_path, 'user_id_%s_test_w2v_128.pkl' % multi_feat))

            # add tfidf features
            for multi_feat in feats.multi_features:
                print('read %s tfidf embedding' % multi_feat)
                tfidf_df = pd.read_pickle(os.path.join(cfg.data_path, 'user_id_%s_test_tfidf.pkl' % multi_feat))
                test_log = pd.merge(test_log, tfidf_df, on='user_id')

            # save data
            print('save data')
            test_log.to_pickle(os.path.join(cfg.data_path, 'test_log_NN_v2.pkl'))

            test_log = test_log.fillna(0)
        else:
            print('read all data')
            test_log = pd.read_pickle(os.path.join(cfg.data_path, 'test_log_NN_v2.pkl'))
            test_log = test_log.fillna(0)

        return test_log

    elif mode == 'val':
        if not os.path.exists(os.path.join(cfg.data_path, 'train_train_log_NN_v2.pkl')):
            train_log, val_log = preprocess(log_path='train_log_time_click_time_sequence.pkl', is_split=True)

            # add w2v features
            for multi_feat in feats.multi_features:
                dense_feats = [x for x in feats.dense_features if multi_feat in x]
                print('read %s w2v embedding' % multi_feat)
                train_log = read_embedding_data(train_log, multi_feat, dense_feats,
                                                os.path.join(cfg.data_path, 'user_id_%s_val_w2v_128.pkl' % multi_feat))
                val_log = read_embedding_data(val_log, multi_feat, dense_feats,
                                              os.path.join(cfg.data_path, 'user_id_%s_val_w2v_128.pkl' % multi_feat))

            # add tfidf features
            for multi_feat in feats.multi_features:
                print('read %s tfidf embedding' % multi_feat)
                tfidf_df = pd.read_pickle(os.path.join(cfg.data_path, 'user_id_%s_val_tfidf.pkl' % multi_feat))
                train_log = pd.merge(train_log, tfidf_df, on='user_id')
                tfidf_df = pd.read_pickle(os.path.join(cfg.data_path, 'user_id_%s_val_tfidf.pkl' % multi_feat))
                val_log = pd.merge(val_log, tfidf_df, on='user_id')

            # process label
            print('read label')
            if label_name == 'age':
                train_log = read_labels_v2(train_log, feats.label_features)
                val_log = read_labels_v2(val_log, feats.label_features)
            elif label_name == 'gender':
                train_log = read_labels_v3(train_log, feats.label_features)
                val_log = read_labels_v3(val_log, feats.label_features)

            # save data
            print('save data')
            train_log.to_pickle(os.path.join(cfg.data_path, 'train_train_log_NN_v2.pkl'))
            val_log.to_pickle(os.path.join(cfg.data_path, 'train_val_log_NN_v2.pkl'))

            train_log = train_log.fillna(0)
            val_log = val_log.fillna(0)
        else:
            print('read all data')
            train_log = pd.read_pickle(os.path.join(cfg.data_path, 'train_train_log_NN_v2.pkl'))
            val_log = pd.read_pickle(os.path.join(cfg.data_path, 'train_val_log_NN_v2.pkl'))

            print('read label')
            if label_name == 'age':
                train_log = read_labels_v2(train_log, feats.label_features)
                val_log = read_labels_v2(val_log, feats.label_features)
            elif label_name == 'gender':
                train_log = read_labels_v3(train_log, feats.label_features)
                val_log = read_labels_v3(val_log, feats.label_features)

            train_log = train_log.fillna(0)
            val_log = val_log.fillna(0)

        return train_log, val_log

    else:
        raise Exception('[!] Such mode not available')


def read_embedding_data(log, in_feat, out_feats, dict_path):
    """
    Read Dense Features from Multi Features

    Improved High Speed I/O
    """
    values = log[in_feat].values
    embed_dic = pd.read_pickle(dict_path)
    embed_dic = embed_dic.set_index([in_feat])

    for f in out_feats:
        log[f] = np.nan

    dense_values = []
    ids_values = [x.split(',') for x in values]
    for ids in tqdm(ids_values):
        try:
            ids.remove('')
        except:
            pass
        ids = [int(x) for x in ids]
        rows = embed_dic.loc[ids].values
        dense_values.append(np.mean(rows, axis=0))

    log[out_feats] = np.array(dense_values)

    return log


def read_labels(log, label_features):
    """
    Read Labels from Log (Normal age + gender)

    Improved High Speed I/O
    """
    age_labels = log['age'].values
    gender_labels = log['gender'].values
    label_idx = age_labels + (gender_labels - 1) * 10 - 1

    for f in label_features:
        log[f] = np.nan

    labels = np.zeros([len(log), len(label_features)])
    labels[range(len(log)), label_idx] = 1
    log[label_features] = labels

    return log


def read_labels_v2(log, label_features):
    """
    Read Labels from Log ( Only age )

    Improved High Speed I/O
    """
    age_labels = log['age'].values
    age_labels[age_labels > 4] = 1
    label_idx = age_labels - 1

    for f in label_features:
        log[f] = np.nan

    labels = np.zeros([len(log), len(label_features)])
    labels[range(len(log)), label_idx] = 1
    log[label_features] = labels

    return log


def read_labels_v3(log, label_features):
    """
    Read Labels from Log ( Only gender )

    Improved High Speed I/O
    """
    log['label_0'] = log['gender'] - 1

    return log


def output_labels(log, preds, pred_path=None, is_train=False):
    """
    Decode Model Output into Age+Gender

    """
    label_idx = np.argmax(preds, axis=1)
    pred_age = label_idx % 10 + 1
    pred_gender = label_idx // 10 + 1

    log['predicted_age'] = pred_age
    log['predicted_gender'] = pred_gender

    if not is_train:
        log = add_outliers(log)
    if pred_path:
        log[['user_id', 'predicted_age', 'predicted_gender']].to_csv(pred_path, index=False)

    return log


def output_labels_v2(log, preds, pred_path=None, is_train=False):
    """
    Decode Model Output into Age

    """
    label_idx = np.argmax(preds, axis=1)
    pred_age = label_idx + 1
    pred_age[pred_age == 1] = 5
    pred_gender = 1

    log['predicted_age'] = pred_age
    log['predicted_gender'] = pred_gender

    if not is_train:
        log = add_outliers(log)
    if pred_path:
        log[['user_id', 'predicted_age', 'predicted_gender']].to_csv(pred_path, index=False)

    return log


def output_labels_v3(log, preds, pred_path=None, is_train=False):
    """
    Decode Model Output into Gender (0/1)

    """
    pred_age = 3
    pred_gender = np.around(preds) + 1

    log['predicted_age'] = pred_age
    log['predicted_gender'] = pred_gender

    if not is_train:
        log = add_outliers(log)
    if pred_path:
        log[['user_id', 'predicted_age', 'predicted_gender']].to_csv(pred_path, index=False)

    return log


def combine_labels(age_log_path, gender_log_path, output_path='preds.csv'):
    """
    Combine Both Age Preds and Gender Preds

    """
    age_df = pd.read_csv(os.path.join(cfg.data_path, age_log_path, 'preds.csv'))
    age_df.drop(['predicted_gender'], inplace=True, axis=1)

    gender_df = pd.read_csv(os.path.join(cfg.data_path, gender_log_path, 'preds.csv'))
    gender_df.drop(['predicted_age'], inplace=True, axis=1)

    combined_df = pd.merge(age_df, gender_df, on='user_id')

    combined_df.to_csv(os.path.join(cfg.data_path, output_path), index=False)


def add_outliers(log):
    """
    Add Outliers After Prediction

    """
    print('Add outliers')
    outliers = Outliers()
    outlier_id = outliers.test_idx_outliers
    log.loc[outlier_id, 'age'] = outliers.outlier_age
    log.loc[outlier_id, 'gender'] = outliers.outlier_gender

    # direct_preds = np.tile([outliers.outlier_age, outliers.outlier_gender], (len(outlier_id), 1))
    # direct_preds_df = pd.DataFrame(direct_preds, columns=['predicted_age', 'predicted_gender'])
    # direct_preds_df['user_id'] = outlier_id

    # log = pd.concat([log, direct_preds_df])

    return log


if __name__ == '__main__':
    # combine gender & age labels
    # combine_labels('log-5-13-8', 'log-5-13-9', 'preds.csv')
    print('Read Raw Data')
    users, ads, log = read_train_raw_data()

    print('Combine User Log')
    _ = combine_log_v2(ads, log, users, is_train=True, save_path='train_log_time_click_time_sequence.pkl')

    print('Read Raw Data')
    ads, log = read_test_raw_data()

    print('Combine User Log')
    _ = combine_log_v2(ads, log, is_train=False, save_path='test_log_time_click_time_sequence.pkl')
