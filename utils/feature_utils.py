import os
import pandas as pd
import numpy as np
import random
import json
import gc
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter
from sklearn import preprocessing
import scipy.special as special
from pandas import DataFrame, Series
from tqdm import tqdm
import time

# import sys
# sys.path.extend('../')

from utils.data_utils import preprocess
from utils.config_utils import Config

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io as scio

cfg = Config()

np.random.seed(cfg.seed)
random.seed(cfg.seed)

"""
Feature Extraction Tools

TF-IDF + W2V + Multi-label + Onehot + Click multiply + Time Sequence + Shuffle

"""


def tfidf(log, pivot_key, out_key, flag, max_df=0.99):
    """
    TF-IDF Features

    creative_id - (1899993, 2086586) -> (1899993, 309296) over20 -> (1899993, 218991) over30
    ad_id - (1899993, 1977177) -> (1899993, 309179) over20 -> (1899993, 219129) over30
    product_id - (1899993, 39014)
    product_category - (1899993, 18)
    advertiser_id - (1899993, 57841)
    industry - (1899993, 331)

    """

    print('word2vec %s on %s' % (out_key, pivot_key))
    # Fetch sentences
    sentences = log[out_key].values

    # Word bag
    if out_key == 'ad_id':
        values = cfg.excluded_ad_id
    elif out_key == 'creative_id':
        values = cfg.excluded_creative_id
    else:
        values = None

    # Train TF-IDF Model
    model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=max_df, min_df=0, vocabulary=values).fit(sentences)

    # Output
    tfidf_results = model.transform(sentences)

    # Save
    print(tfidf_results.shape)
    scio.savemat('%s_%s_%s_tfidf.mat' % (pivot_key, out_key, flag), {'mat': tfidf_results})


def w2v(log, pivot_key, out_key, flag, size=128, window=10, iter=10):
    """
    Walk2Vector Algorithm for Embedding
    window - 10
    window - 5

    Features:
    product_category_embedding
    industry_embedding
    advertiser_id_embedding
    product_id_embedding
    ad_id_embedding
    (Deprecated)creative_id_embedding

    (Deprecated)ad_id_creative_id_embedding
    TODO product_category_industry_embedding
    TODO industry_advertiser_id_embedding
    TODO product_category_product_id_embedding

    TODO apply to data without droping duplicate words
    """

    print('word2vec %s on %s' % (out_key, pivot_key))
    # Fetch sentences
    sentences = []
    content = log[out_key].values
    for s in content:
        words = s.split(',')
        if '' in words:
            words.remove('')
        sentences.append(words)

    # Build Word Bag
    if out_key == 'ad_id':
        values = set(cfg.excluded_ad_id)
        # Train Word2Vec Model
        print('training...')
        random.shuffle(sentences)
        model = Word2Vec(sentences, size=size, window=window, min_count=21, workers=10, iter=iter)
    elif out_key == 'creative_id':
        values = set(cfg.excluded_creative_id)
        # Train Word2Vec Model
        print('training...')
        random.shuffle(sentences)
        model = Word2Vec(sentences, size=size, window=window, min_count=21, workers=10, iter=iter)
    else:
        values = []
        for i in sentences:
            values.extend(i)
        values = set(values)
        # Train Word2Vec Model
        print('training...')
        random.shuffle(sentences)
        model = Word2Vec(sentences, size=size, window=window, min_count=1, workers=10, iter=iter)

    # Output
    print('outputing...')
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model[str(v)])
            w2v.append(a)
        except:
            pass

    # Save
    print('saving...')
    out_df = pd.DataFrame(w2v)
    names = [out_key]
    for i in range(size):
        names.append(pivot_key + '_w2v_embedding_' + out_key + '_' + str(size) + '_' + str(i))
    out_df.columns = names
    out_df.to_pickle(
        os.path.join(cfg.data_path,
                     pivot_key + '_' + out_key + '_' + flag + '_w2v_' + str(size) + '_window_' + str(window) + '.pkl'))


def deepwalk(log, pivot_key, out_key, flag, size):
    """
    DeepWalk for Graph Embedding

    TODO Fix / Test
    """
    print('deepwalk %s on %s' % (out_key, pivot_key))
    # 构建图
    dic = {}
    for item in log[[pivot_key, out_key]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_' + str(int(item[1]))].add('user_' + str(int(item[0])))
        except:
            dic['item_' + str(int(item[1]))] = set(['user_' + str(int(item[0]))])
        try:
            dic['user_' + str(int(item[0]))].add('item_' + str(int(item[1])))
        except:
            dic['user_' + str(int(item[0]))] = set(['item_' + str(int(item[1]))])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])
        dic_cont[key] = len(dic[key])
    print("creating")
    # 构建路径
    path_length = 10
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            key = dic[sentence[-1]][random.randint(0, dic_cont[sentence[-1]] - 1)]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    # 训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=size, window=4, min_count=1, sg=1, workers=10, iter=20)
    print('outputing...')
    # 输出
    values = set(log[pivot_key].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model['user_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [pivot_key]
    for i in range(size):
        names.append(pivot_key + '_' + out_key + '_' + names[0] + '_deepwalk_embedding_' + str(size) + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(
        'data/' + pivot_key + '_' + out_key + '_' + pivot_key + '_' + flag + '_deepwalk_' + str(size) + '.pkl')
    ########################
    values = set(log[out_key].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model['item_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [out_key]
    for i in range(size):
        names.append(pivot_key + '_' + out_key + '_' + names[0] + '_deepwalk_embedding_' + str(size) + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(
        'data/' + pivot_key + '_' + out_key + '_' + out_key + '_' + flag + '_deepwalk_' + str(size) + '.pkl')


def gae_encode(log, pivot_key, out_key, flag, size):
    """
    Graph AutoEncoder Embedding

    TODO Apply it!
    """
    return


def get_agg_features(train_df, test_df, pivot_key, out_key, agg, log=None):
    """
    Statistical Feature

    """
    if type(pivot_key) == str:
        pivot_key = [pivot_key]
    if log is None:
        if agg != 'size':
            data = train_df[pivot_key + [out_key]].append(
                test_df.drop_duplicates(pivot_key + [out_key])[pivot_key + [out_key]])
        else:
            data = train_df[pivot_key].append(test_df.drop_duplicates(pivot_key)[pivot_key])
    else:
        if agg != 'size':
            data = log[pivot_key + [out_key]]
        else:
            data = log[pivot_key]
    if agg == "size":
        tmp = pd.DataFrame(data.groupby(pivot_key).size()).reset_index()
    elif agg == "count":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].count()).reset_index()
    elif agg == "mean":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].mean()).reset_index()
    elif agg == "unique":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].nunique()).reset_index()
    elif agg == "max":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].max()).reset_index()
    elif agg == "min":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].min()).reset_index()
    elif agg == "sum":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].sum()).reset_index()
    elif agg == "std":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].std()).reset_index()
    elif agg == "median":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].median()).reset_index()
    elif agg == "skew":
        tmp = pd.DataFrame(data.groupby(pivot_key)[out_key].skew()).reset_index()
    elif agg == "unique_mean":
        group = data.groupby(pivot_key)
        group = group.apply(lambda x: np.mean(list(Counter(list(x[out_key])).values())))
        tmp = pd.DataFrame(group.reset_index())
    elif agg == "unique_var":
        group = data.groupby(pivot_key)
        group = group.apply(lambda x: np.var(list(Counter(list(x[out_key])).values())))
        tmp = pd.DataFrame(group.reset_index())
    else:
        raise Exception("agg error")
    if log is None:
        tmp.columns = pivot_key + ['_'.join(pivot_key) + "_" + out_key + "_" + agg]
        print('_'.join(pivot_key) + "_" + out_key + "_" + agg)
    else:
        tmp.columns = pivot_key + ['_'.join(pivot_key) + "_" + out_key + "_log_" + agg]
        print('_'.join(pivot_key) + "_" + out_key + "_log_" + agg)
    try:
        del test_df['_'.join(pivot_key) + "_" + out_key + "_" + agg]
        del train_df['_'.join(pivot_key) + "_" + out_key + "_" + agg]
    except:
        pass
    test_df = test_df.merge(tmp, on=pivot_key, how='left')
    train_df = train_df.merge(tmp, on=pivot_key, how='left')
    del tmp
    del data
    gc.collect()
    print(train_df.shape, test_df.shape)
    return train_df, test_df


def kfold_static(train_df, test_df, pivot_key, label):
    """
    Five-Folds Statistical Features

    TODO Fix / Test
    """
    print("K-fold static:", pivot_key + '_' + label)

    # K-fold positive and negative num
    avg_rate = train_df[label].mean()
    num = len(train_df) // 5
    index = [0 for i in range(num)] + [1 for i in range(num)] + [2 for i in range(num)] + \
            [3 for i in range(num)] + [4 for i in range(len(train_df) - 4 * num)]
    random.shuffle(index)
    train_df['index'] = index
    # 五折统计
    dic = [{} for i in range(5)]
    dic_all = {}
    for item in train_df[['index', pivot_key, label]].values:
        try:
            dic[item[0]][item[1]].append(item[2])
        except:
            dic[item[0]][item[1]] = []
            dic[item[0]][item[1]].append(item[2])
    print("static done!")
    # 构造训练集的五折特征，均值，中位数等
    mean = []
    median = []
    std = []
    Min = []
    Max = []
    cache = {}
    for item in train_df[['index', pivot_key]].values:
        if tuple(item) not in cache:
            temp = []
            for i in range(5):
                if i != item[0]:
                    try:
                        temp += dic[i][item[1]]
                    except:
                        pass
            if len(temp) == 0:
                cache[tuple(item)] = [-1] * 5
            else:
                cache[tuple(item)] = [np.mean(temp), np.median(temp), np.std(temp), np.min(temp), np.max(temp)]
        temp = cache[tuple(item)]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])
    del cache
    train_df[pivot_key + '_' + label + '_mean'] = mean
    train_df[pivot_key + '_' + label + '_median'] = median
    train_df[pivot_key + '_' + label + '_std'] = std
    train_df[pivot_key + '_' + label + '_min'] = Min
    train_df[pivot_key + '_' + label + '_max'] = Max
    print("train done!")

    # 构造测试集的五折特征，均值，中位数等
    mean = []
    median = []
    std = []
    Min = []
    Max = []
    cache = {}
    for uid in test_df[pivot_key].values:
        if uid not in cache:
            temp = []
            for i in range(5):
                try:
                    temp += dic[i][uid]
                except:
                    pass
            if len(temp) == 0:
                cache[uid] = [-1] * 5
            else:
                cache[uid] = [np.mean(temp), np.median(temp), np.std(temp), np.min(temp), np.max(temp)]
        temp = cache[uid]
        mean.append(temp[0])
        median.append(temp[1])
        std.append(temp[2])
        Min.append(temp[3])
        Max.append(temp[4])

    test_df[pivot_key + '_' + label + '_mean'] = mean
    test_df[pivot_key + '_' + label + '_median'] = median
    test_df[pivot_key + '_' + label + '_std'] = std
    test_df[pivot_key + '_' + label + '_min'] = Min
    test_df[pivot_key + '_' + label + '_max'] = Max
    print("test done!")
    del train_df['index']
    print(pivot_key + '_' + label + '_mean')
    print(pivot_key + '_' + label + '_median')
    print(pivot_key + '_' + label + '_std')
    print(pivot_key + '_' + label + '_min')
    print(pivot_key + '_' + label + '_max')
    print('avg of mean', np.mean(train_df[pivot_key + '_' + label + '_mean']),
          np.mean(test_df[pivot_key + '_' + label + '_mean']))
    print('avg of median', np.mean(train_df[pivot_key + '_' + label + '_median']),
          np.mean(test_df[pivot_key + '_' + label + '_median']))
    print('avg of std', np.mean(train_df[pivot_key + '_' + label + '_std']),
          np.mean(test_df[pivot_key + '_' + label + '_std']))
    print('avg of min', np.mean(train_df[pivot_key + '_' + label + '_min']),
          np.mean(test_df[pivot_key + '_' + label + '_min']))
    print('avg of max', np.mean(train_df[pivot_key + '_' + label + '_max']),
          np.mean(test_df[pivot_key + '_' + label + '_max']))


def crowd_uid(train_df, test_df, pivot_key, minor_key, log, size):
    """
    Second-order Statistical Features

    多值特征，提取以pivot_key为主键，minor_key在log中出现Topk的ID
    如pivot_key=aid, minor_key=uid, size=100,则表示访问该广告最多的前100名用户

    TODO Fix / Test
    """

    print("crowd_uid features", pivot_key, minor_key)
    dic = {}
    log[pivot_key] = log[pivot_key].fillna(-1).astype(int)
    train_df[pivot_key] = train_df[pivot_key].fillna(-1).astype(int)
    test_df[pivot_key] = test_df[pivot_key].fillna(-1).astype(int)
    for item in tqdm(log[[pivot_key, minor_key, 'request_day']].values, total=len(log)):
        try:
            dic[item[0]][0][item[1]] += 1
        except:
            dic[item[0]] = [Counter(), Counter()]
            dic[item[0]][0][item[1]] = 1

    items = []
    for key in tqdm(dic, total=len(dic)):
        conter = dic[key][0]
        item = [str(x[0]) for x in conter.most_common(size)]
        if len(item) == 0:
            item = ['-1']
        items.append([key, ' '.join(item)])

    df = pd.DataFrame(items)
    df.columns = [pivot_key, pivot_key + '_' + minor_key + 's']
    df = df.drop_duplicates(pivot_key)
    try:
        del train_df[pivot_key + '_' + minor_key + 's']
        del test_df[pivot_key + '_' + minor_key + 's']
    except:
        pass
    train_df = train_df.merge(df, on=pivot_key, how='left')
    test_df = test_df.merge(df, on=pivot_key, how='left')
    train_df[pivot_key + '_' + minor_key + 's'] = train_df[pivot_key + '_' + minor_key + 's'].fillna('-1')
    test_df[pivot_key + '_' + minor_key + 's'] = test_df[pivot_key + '_' + minor_key + 's'].fillna('-1')
    del df
    del items
    del dic
    gc.collect()
    return train_df, test_df


def history(train_df, test_df, log, pivot_key, minor_key):
    """
    History Features - Yesterday / Last range of days / Overall History

    TODO Fix / Test
    """
    # 以pivot_key为主键，统计最近一次minor_key的值
    print("history", pivot_key, minor_key)
    nan = log[minor_key].median()
    dic = {}
    temp_log = log[[pivot_key, 'request_day', minor_key, 'aid']].drop_duplicates(['aid', 'request_day'], keep='last')
    for item in log[[pivot_key, 'request_day', minor_key]].values:
        if (item[0], item[1]) not in dic:
            dic[(item[0], item[1])] = [item[2]]
        else:
            dic[(item[0], item[1])].append(item[2])
    for key in dic:
        dic[key] = np.mean(dic[key])
    # 统计训练集的特征
    items = []
    cont = 0
    day = log['request_day'].min()
    for item in train_df[[pivot_key, 'request_day']].values:
        flag = False
        for i in range(item[1] - 1, day - 1, -1):
            if (item[0], i) in dic:
                items.append(dic[(item[0], i)])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    train_df['history_' + pivot_key + '_' + minor_key] = items
    # 统计测试集的特征
    items = []
    cont = 0
    day_min = log['request_day'].min()
    day_max = log['request_day'].max()
    for item in test_df[pivot_key].values:
        flag = False
        for i in range(day_max, day_min - 1, -1):
            if (item, i) in dic:
                items.append(dic[(item, i)])
                flag = True
                cont += 1
                break
        if flag is False:
            items.append(nan)
    test_df['history_' + pivot_key + '_' + minor_key] = items

    print(train_df['history_' + pivot_key + '_' + minor_key].mean())
    print(test_df['history_' + pivot_key + '_' + minor_key].mean())
    del items
    del dic
    gc.collect()
    return train_df, test_df


class Features:
    def __init__(self):
        """
        creative_id: 2086586
        ad_id: 1977177
        advertiser_id: 50544
        product_id: 31581

        """
        # label
        self.label_features = ['label_' + str(i) for i in range(1)]  # gender
        # self.label_features = ['label_' + str(i) for i in range(4)]  # age

        # embedding
        self.single_features = []

        # fm
        self.cross_features = []

        # fm
        # normal feats
        self.multi_features = ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']

        # click_times topN feats
        # self.multi_features += ['product_category_click_top3', 'product_category_click_top6']
        # self.multi_features += ['industry_click_top3', 'industry_click_top6', 'industry_click_top10']
        # self.multi_features += ['advertiser_id_click_top2', 'advertiser_id_click_top5', 'advertiser_id_click_top10']
        # self.multi_features += ['product_id_click_top5', 'product_id_click_top9', 'product_id_click_top15']
        # self.multi_features += ['ad_id_click_top2', 'ad_id_click_top8']
        # self.multi_features += ['creative_id_click_top6', 'creative_id_click_top10']

        # word2vec
        self.dense_features = ['user_id_w2v_embedding_creative_id_128_' + str(i) for i in range(128)] + \
                              ['user_id_w2v_embedding_ad_id_128_' + str(i) for i in range(128)] + \
                              ['user_id_w2v_embedding_product_id_128_' + str(i) for i in range(128)] + \
                              ['user_id_w2v_embedding_product_category_128_' + str(i) for i in range(128)] + \
                              ['user_id_w2v_embedding_advertiser_id_128_' + str(i) for i in range(128)] + \
                              ['user_id_w2v_embedding_industry_128_' + str(i) for i in range(128)]

        # tfidf
        self.dense_features += ['user_id_tfidf_embedding_creative_id_' + str(i) for i in range(309296)] + \
                               ['user_id_tfidf_embedding_ad_id_' + str(i) for i in range(309296)] + \
                               ['user_id_tfidf_embedding_product_id_' + str(i) for i in range(39014)] + \
                               ['user_id_tfidf_embedding_product_category_' + str(i) for i in range(18)] + \
                               ['user_id_tfidf_embedding_advertiser_id_' + str(i) for i in range(57841)] + \
                               ['user_id_tfidf_embedding_industry_' + str(i) for i in range(331)]

        # onehot
        self.multi_features += ['product_category_onehot', 'industry_onehot']

        # self.dense_features += ['user_id_w2v_embedding_ad_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_w2v_embedding_product_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_w2v_embedding_product_category_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_w2v_embedding_advertiser_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_w2v_embedding_industry_64_' + str(i) for i in range(64)]

        # deep walk
        # self.dense_features += ['user_id_deepwalk_embedding_creative_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_deepwalk_embedding_ad_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_deepwalk_embedding_product_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_deepwalk_embedding_product_category_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_deepwalk_embedding_advertiser_id_64_' + str(i) for i in range(64)] + \
        #                        ['user_id_deepwalk_embedding_industry_64_' + str(i) for i in range(64)]

        # key-values memory (TopN/time_range + click_times/click_rate/click_count + key)
        # self.kv_features = ['creative_id_kv_click_times', 'ad_id_kv_click_times', 'product_id_kv_click_times',
        #                     'product_category_kv_click_times', 'advertiser_id_kv_click_times',
        #                     'industry_kv_click_times']


if __name__ == "__main__":
    # for path1, path2, log_path, flag, wday, day in \
    #         [('data/train_dev.pkl', 'data/dev.pkl', 'data/user_log_dev.pkl', 'dev', 1, 17974),
    #          ('data/train.pkl', 'data/test.pkl', 'data/user_log_test.pkl', 'test', 3, 17976)]:
    #
    #     # 拼接静态特征
    #     print(path1, path2, log_path, flag)
    #     train_df = pd.read_pickle(path1)
    #     test_df = pd.read_pickle(path2)
    #     log = pd.read_pickle(log_path)
    #     print(train_df.shape, test_df.shape, log.shape)
    #     df = pd.read_pickle('data/testA/ad_static_feature.pkl')
    #     log = log.merge(df, on='aid', how='left')
    #     del df
    #     gc.collect()
    #     print(train_df.shape, test_df.shape, log.shape)
    #
    #     # 多值特征
    #     train_df, test_df = crowd_uid(train_df, test_df, 'good_id', 'advertiser', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'good_id', 'request_day', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'good_id', 'position', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'good_id', 'period_id', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'good_id', 'wday', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'advertiser', 'good_id', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'advertiser', 'request_day', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'advertiser', 'position', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'advertiser', 'period_id', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'advertiser', 'wday', log, 100)
    #     train_df, test_df = crowd_uid(train_df, test_df, 'aid', 'uid', log, 20)
    #
    #     # 历史特征
    #     for pivot in ['aid']:
    #         for f in ['imp', 'bid', 'pctr', 'quality_ecpm', 'totalEcpm']:
    #             history(train_df, test_df, log, pivot, f)
    #
    #     # 五折特征
    #     kfold_static(train_df, test_df, 'aid', 'imp')
    #     kfold_static(train_df, test_df, 'good_id', 'imp')
    #     kfold_static(train_df, test_df, 'advertiser', 'imp')
    #
    #     # 统计特征
    #     train_df, test_df = get_agg_features(train_df, test_df, ["good_id"], 'advertiser', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["good_id"], 'aid', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["good_id"], 'ad_size', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["good_id"], 'ad_type_id', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["good_id"], 'good_id', "size")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'good_id', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'aid', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'ad_size', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'ad_type_id', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'good_type', "count")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["advertiser"], 'advertiser', "size")
    #     train_df, test_df = get_agg_features(train_df, test_df, ['good_type'], 'good_type', "size")
    #     train_df, test_df = get_agg_features(train_df, test_df, ["aid"], 'aid', "size")
    #
    #     # 保存数据
    #     print(train_df.shape, test_df.shape, log.shape)
    #     train_df.to_pickle(path1)
    #     test_df.to_pickle(path2)
    #     print(list(train_df))
    #     print("*" * 80)
    #     print("save done!")
    #
    #     # Word2vec
    #     w2v(log, 'uid', 'good_id', flag, 64)
    #     w2v(log, 'uid', 'advertiser', flag, 64)
    #     w2v(log, 'uid', 'aid', flag, 64)
    #
    #     # Deepwalk
    #     deepwalk(log, 'uid', 'aid', flag, 64)
    #     deepwalk(log, 'uid', 'good_id', flag, 64)
    #
    #     del train_df
    #     del test_df
    #     del log
    #     gc.collect()

    # Word2vec
    print('preprocess train_log')
    train_log = preprocess(log_path='train_log_time_click_time_sequence.pkl')
    print('preprocess test_log')
    test_log = preprocess(is_train=False, log_path='test_log_time_click_time_sequence.pkl')
    log = pd.concat([train_log, test_log])
    log.reset_index(drop=True, inplace=True)
    flag = 'test'

    # print('preprocess train_log')
    # train_log = preprocess(log_path='train_log.pkl')
    # log = train_log
    # flag = 'val'

    w2v(log, 'user_id', 'creative_id', flag, 128, window=10)
    w2v(log, 'user_id', 'ad_id', flag, 128, window=10)
    w2v(log, 'user_id', 'product_id', flag, 128, window=10)
    w2v(log, 'user_id', 'product_category', flag, 128, window=10)
    w2v(log, 'user_id', 'advertiser_id', flag, 128, window=10)
    w2v(log, 'user_id', 'industry', flag, 128, window=10)

    # tfidf(log, 'user_id', 'creative_id', flag, max_df=0.99)
    # tfidf(log, 'user_id', 'ad_id', flag, max_df=0.99)
    # tfidf(log, 'user_id', 'product_id', flag, max_df=0.99)
    # tfidf(log, 'user_id', 'product_category', flag, max_df=0.99)
    # tfidf(log, 'user_id', 'advertiser_id', flag, max_df=0.99)
    # tfidf(log, 'user_id', 'industry', flag, max_df=0.99)
