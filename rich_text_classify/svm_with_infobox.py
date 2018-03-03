# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib  #也可以选择pickle等保存模型，请随意
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import json


reload(sys)
sys.setdefaultencoding('utf8')

def load_infobox_dict():
    """
    从抽取好的json文件加载infobox的内容
    :return:
    """
    pos_file_list = os.listdir('./page_infobox/pos_case/')
    pos_info_dict_list = []
    for file_name in pos_file_list:
        with open('./page_infobox/pos_case/'+file_name) as in_file:
            info_dict_tmp = json.load(in_file)
            info_dict = {}
            for k in info_dict_tmp:
                info_dict[k.replace('    ', '')] = info_dict_tmp[k]
            pos_info_dict_list.append(info_dict)

    neg_file_list = os.listdir('./page_infobox/neg_case/')
    neg_info_dict_list = []
    for file_name in neg_file_list:
        with open('./page_infobox/neg_case/' + file_name) as in_file:
            info_dict_tmp = json.load(in_file)
            info_dict = {}
            for k in info_dict_tmp:
                info_dict[k.replace('    ', '')] = info_dict_tmp[k]
            neg_info_dict_list.append(info_dict)
    
    info_dict = {'pos_case': pos_info_dict_list, 'neg': neg_info_dict_list}
    return info_dict


def build_property_set(info_dict, n_property):
    """
    构建正负例属性词表
    :param info_dict: 由load_infobox_dict产生
    :param n_property: 正负样例top_n属性词数
    :return: 
    """
    pos_info_dict_list = info_dict['pos_case']
    pos_property_dict = {}
    for item_info_dict in pos_info_dict_list:
        for k in item_info_dict:
            if pos_property_dict.get(k) is None:
                pos_property_dict[k] = 1
            else:
                pos_property_dict[k] += 1

    pos_property_list = [p[0] for p in sorted(pos_property_dict.items(), key=lambda d:d[1], reverse=True)]

    neg_info_dict_list = info_dict['neg']
    neg_property_dict = {}
    for item_info_dict in neg_info_dict_list:
        for k in item_info_dict:
            if neg_property_dict.get(k) is None:
                neg_property_dict[k] = 1
            else:
                neg_property_dict[k] += 1

    neg_property_list = [p[0] for p in sorted(neg_property_dict.items(), key=lambda d:d[1], reverse=True)]
    property_list = pos_property_list[0:n_property]+neg_property_list[0:n_property]

    return property_list


def build_case_vec(info_dict, property_list):
    """
    利用属性词表和词条infobox的属性,构建向量,one-hot形式
    :param info_dict: 正负样例
    :param property_list: 属性词表
    :return: 
    """
    pos_case_vec_list = []
    pos_info_list = info_dict['pos_case']
    for case in pos_info_list:
        case_vec = [0]*len(property_list)
        for i in range(len(property_list)):
            if case.get(property_list[i]) is not None:
                case_vec[i] = 1
        case_vec.append(1)
        pos_case_vec_list.append(case_vec)

    neg_case_vec_list = []
    neg_info_list = info_dict['neg']
    for case in neg_info_list:
        case_vec = [0] * len(property_list)
        for i in range(len(property_list)):
            if case.get(property_list[i]) is not None:
                case_vec[i] = 1
        case_vec.append(0)
        neg_case_vec_list.append(case_vec)

    case_vec_dict = {'pos_case':pos_case_vec_list, 'neg':neg_case_vec_list}
    return case_vec_dict



def svm_classify_with_info_vec(n_property):
    """
    利用infobox的onehot特征向量做svm分类
    :param n_property:
    :return:
    """
    info_dict = load_infobox_dict()
    property_list = build_property_set(info_dict, n_property)
    case_vec_dict = build_case_vec(info_dict, property_list)
    pos_vec_list = case_vec_dict['pos_case']
    neg_vec_list = case_vec_dict['neg']

    neg_set = np.array(neg_vec_list)
    x_neg, y_neg = np.split(neg_set, (n_property*2,), axis=1)
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, random_state=11, test_size=200)

    pos_set = np.array(pos_vec_list)
    x_pos, y_pos = np.split(pos_set, (n_property*2,), axis=1)
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, random_state=11, test_size=200)

    # 拼接训练集,测试集
    x_train = np.concatenate((x_neg_train, x_pos_train), axis=0)
    y_train = np.concatenate((y_neg_train, y_pos_train), axis=0)
    x_test = np.concatenate((x_neg_test, x_pos_test), axis=0)
    y_test = np.concatenate((y_neg_test, y_pos_test), axis=0)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
    clf.fit(x_train, y_train.ravel())

    y_hat = clf.predict(x_train)
    train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
    with open('./infobox_classify_result/train.txt', 'a') as out_file:
        out_file.write(train_socre)

    y_hat = clf.predict(x_test)
    test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
    with open('./infobox_classify_result/test.txt', 'a') as out_file:
        out_file.write(test_score)




svm_classify_with_info_vec(200)























    
    