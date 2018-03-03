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


reload(sys)
sys.setdefaultencoding('utf8')

def textPrecessing(text, denoise=True):
    """
    文本预处理,对摘要文本做分词,分词用结巴,去除标点和停用词处理
    :param text: 待处理词条摘要文本
    :param denoise: 是否去除停用词
    :return:
    """
    # 停用词文件是utf8编码
    stoplist = {}.fromkeys([line.strip() for line in open("./assist_file/stopword.txt")])

    segs = jieba.lcut(text, cut_all='True')
    segs = [word.encode('utf-8') for word in segs]
    if denoise:
        segs = [word for word in list(segs) if word not in stoplist and word != '']

    return " ".join(segs)




def load_doc_list(case_type= 'all', denoise=True):
    """
    加载词条摘要文本,返回预处理过的文本列表
    :param case_type: 表示加载那部分样例的摘要文本,pos_case,neg或all
    :param denoise: 是否去除停用词
    :return:
    """
    if case_type == 'pos_case':
        pos_file_list = os.listdir('./page_abstract/pos_case/')
        pos_text_list = []
        for file in pos_file_list:
            with open('./page_abstract/pos_case/'+file, 'r') as in_file:
                text = in_file.read()
                pos_text_list.append(textPrecessing(text, denoise))

        return pos_text_list

    elif case_type == 'neg':
        neg_file_list = os.listdir('./page_abstract/neg_case/')
        neg_text_list = []
        for file in neg_file_list:
            with open('./page_abstract/neg_case/'+file, 'r') as in_file:
                text = in_file.read()
                neg_text_list.append(textPrecessing(text, denoise))

        return neg_text_list

    elif case_type == 'all':
        pos_file_list = os.listdir('./page_abstract/pos_case/')
        pos_text_list = []
        for file in pos_file_list:
            with open('./page_abstract/pos_case/' + file, 'r') as in_file:
                text = in_file.read()
                pos_text_list.append(textPrecessing(text, denoise))
        neg_file_list = os.listdir('./page_abstract/neg_case/')
        neg_text_list = []
        for file in neg_file_list:
            with open('./page_abstract/neg_case/' + file, 'r') as in_file:
                text = in_file.read()
                neg_text_list.append(textPrecessing(text, denoise))

        all_text_list = pos_text_list+neg_text_list
        return all_text_list

    else:
        print 'Bad Usage with load_doc_list!'
        return None




def count_vectorizer(case_type= 'all', n_features=100, denoise=True):
    """
    词频统计,模型保存,只训练一次
    :param denoise: 是否去除停用词
    :return:
    """
    doc_list = load_doc_list(case_type=case_type, denoise=denoise)
    if os.path.exists('./assist_file/tf_%s_%s.model' % (str(n_features), str(int(denoise)))):
        tf_vectorizer = joblib.load('./assist_file/tf_%s_%s.model' % (str(n_features), str(int(denoise))))
        tf = tf_vectorizer.transform(doc_list)
    else:
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,)
        tf = tf_vectorizer.fit_transform(doc_list)
        joblib.dump(tf_vectorizer, './assist_file/tf_%s_%s.model' % (str(n_features), str(int(denoise))))

    return tf


def lda_train(n_topics=30, n_features=100, denoise=True):
    """
    lda模型训练
    :param n_topics: 主题数
    :param n_features: 特征词数
    :param denoise: 是否去除停用词
    :return:lda模型
    """

    if os.path.exists('./assist_file/lda_%s_%s_%s.model' % (str(n_topics), str(n_features), str(int(denoise)))):
        lda = joblib.load('./assist_file/lda_%s_%s_%s.model' % (str(n_topics), str(n_features), str(int(denoise))))
    else:
        tf_matrix = count_vectorizer(n_features=n_features, denoise=denoise)
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50, learning_method='batch')
        lda.fit(tf_matrix)  # tf_matrix即为Document_word Sparse Matrix
        joblib.dump(lda, './assist_file/lda_%s_%s_%s.model' % (str(n_topics), str(n_features), str(int(denoise))))


    return lda


def convert_doc_lda_feature_vec(lda_model, n_features=100, denoise=True):
    """
    将正负样例词条摘要文本转换成lda主题向量
    :return:
    """
    pos_tf_matrix = count_vectorizer(case_type='pos_case', n_features=n_features, denoise=denoise)
    pos_doc_topic_list = lda_model.transform(pos_tf_matrix).tolist()
    for doc_topic in pos_doc_topic_list:
        doc_topic.append('1')

    neg_tf_matrix = count_vectorizer(case_type='neg', n_features=n_features, denoise=denoise)
    neg_doc_topic_list = lda_model.transform(neg_tf_matrix).tolist()
    for doc_topic in neg_doc_topic_list:
        doc_topic.append('0')

    doc_topic_dict = {'pos_case': pos_doc_topic_list, 'neg': neg_doc_topic_list}

    return doc_topic_dict


def svm_classify_with_abstract_topic_vec(svm_kernel='linear', n_features=500, n_topics=200, denoise=False):
    """
    利用lda模型生成的文档主题向量作为特征,进行svm分类
    :return:
    """
    lda_model = lda_train(n_topics=n_topics, n_features=n_features, denoise=denoise)
    doc_topic_list = convert_doc_lda_feature_vec(lda_model, n_features=n_features, denoise=denoise)
    pos_vec_list = doc_topic_list['pos_case']
    neg_vec_list = doc_topic_list['neg']

    neg_set = np.array(neg_vec_list)
    x_neg, y_neg = np.split(neg_set, (n_topics,), axis=1)
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, random_state=11, test_size=200)

    pos_set = np.array(pos_vec_list)
    x_pos, y_pos = np.split(pos_set, (n_topics,), axis=1)
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, random_state=11, test_size=200)

    # 拼接训练集,测试集
    x_train = np.concatenate((x_neg_train, x_pos_train), axis=0)
    y_train = np.concatenate((y_neg_train, y_pos_train), axis=0)
    x_test = np.concatenate((x_neg_test, x_pos_test), axis=0)
    y_test = np.concatenate((y_neg_test, y_pos_test), axis=0)


    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    y_hat = clf.predict(x_train)
    train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
    with open('./lda_abstract_classify_result/_%s_%s_%s_%s'%(str(svm_kernel), str(n_feature), str(n_topics), str(denois))+'_train.txt', 'a') as out_file:
        out_file.write(train_socre)

    y_hat = clf.predict(x_test)
    test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
    with open('./lda_abstract_classify_result/_%s_%s_%s_%s'%(str(svm_kernel), str(n_feature), str(n_topics), str(denois))+'_test.txt', 'a') as out_file:
        out_file.write(test_score)



n_feature_list = [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
n_topic_list = [10, 20, 30, 40, 50, 60, 80, 100, 140, 180, 200, 250, 300]
denois_list = [True, False]
svm_kernel_list = ['linear', 'rbf']

for n_feature in n_feature_list:
    for n_topic in n_topic_list:
        for denois in denois_list:
            for svm_kernel in svm_kernel_list:
                svm_classify_with_abstract_topic_vec(svm_kernel, n_feature, n_topic, denois)












