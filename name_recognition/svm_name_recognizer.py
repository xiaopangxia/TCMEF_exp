# -*- coding: utf-8 -*-
import sys
from gensim.models import word2vec
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from pyltp import Segmentor
import numpy as np
import random

reload(sys)
sys.setdefaultencoding('utf8')

# 实验人名识别
# 数据说明:
# name.txt中有5000条人名词条
# opendomain_noname.txt里有5000条开放域词条,不包含人名


class FaeturePrepare():
    """
    特征准备
    """
    @classmethod
    def wiki_item_2_vec(cls, item_list):
        """
        每个字50维,词条由字向量拼接构成,每个词条500维
        认为词条最多10个字,超过10个字截断,不足的以0补
        :param item_list:
        :return:
        """
        word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/wiki_zi_with_name_word2vec.model')

        item_vec_list = []
        for item in item_list:
            case_zi_line = unicode(item)
            case_vec = []
            is_useful = 0
            for zi in case_zi_line:
                try:
                    # 拼接
                    case_vec.extend(word_vec_model[unicode(zi)].tolist())
                    is_useful = 1
                except Exception, e:
                    with open("not_in_zi_table.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')

            # 多退少补
            if len(case_vec) > 500:
                case_vec = case_vec[0:500]
            else:
                while (len(case_vec) < 500):
                    case_vec.append(0)
            if is_useful:
                item_vec_list.append(case_vec)

        return item_vec_list



class SVMClassify():
    """
    svm分类方法类
    """


    @classmethod
    def svm_classify_with_zi_vec(cls, svm_kernel='linear'):
        """
        利用人名词条数据和非人名词条数据训练svm模型进行分类
        :return:
        """
        pos_case_list = []
        with open('./data/name.txt', 'r') as in_file:
            for line in in_file.readlines():
                pos_case_list.append(line.strip())

        pos_case_vec_list = FaeturePrepare.wiki_item_2_vec(pos_case_list)
        for item in pos_case_vec_list:
            item.append(1)


        # 负样本
        neg_case_list = []
        with open('./data/open_domain_noname.txt', 'r') as in_file:
            for line in in_file.readlines():
                neg_case_list.append(line.strip())

        neg_case_vec_list = FaeturePrepare.wiki_item_2_vec(neg_case_list)
        for item in neg_case_vec_list:
            item.append(0)

        case_vec_list = pos_case_vec_list + neg_case_vec_list
        # 列表转np类型
        case_train_set = np.array(case_vec_list)
        x, y = np.split(case_train_set, (500,), axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

        # svm分类,linear先不用核函数,rbf带卷积核
        clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel())

        y_hat = clf.predict(x_train)
        train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./result/%s_train.txt' % svm_kernel, 'a') as out_file:
            out_file.write(train_socre)

        y_hat = clf.predict(x_test)
        test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./result/%s_test.txt' % svm_kernel, 'a') as out_file:
            out_file.write(test_score)


    @classmethod
    def svm_classify_with_zi_vec_for_C(cls, svm_kernel='linear', para_C=0.8):
        """
        利用人名词条数据和非人名词条数据训练svm模型进行分类
        :return:
        """
        pos_case_list = []
        with open('./data/name.txt', 'r') as in_file:
            for line in in_file.readlines():
                pos_case_list.append(line.strip())

        pos_case_vec_list = FaeturePrepare.wiki_item_2_vec(pos_case_list)
        for item in pos_case_vec_list:
            item.append(1)


        # 负样本
        neg_case_list = []
        with open('./data/open_domain_noname.txt', 'r') as in_file:
            for line in in_file.readlines():
                neg_case_list.append(line.strip())

        neg_case_vec_list = FaeturePrepare.wiki_item_2_vec(neg_case_list)
        for item in neg_case_vec_list:
            item.append(0)

        case_vec_list = pos_case_vec_list + neg_case_vec_list
        # 列表转np类型
        case_train_set = np.array(case_vec_list)
        x, y = np.split(case_train_set, (500,), axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

        # svm分类,linear先不用核函数,rbf带卷积核
        clf = svm.SVC(C=para_C, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel())

        y_hat = clf.predict(x_train)
        train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./result/%s_for_C_train.txt' % svm_kernel, 'a') as out_file:
            out_file.write(train_socre)

        y_hat = clf.predict(x_test)
        test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./result/%s_for_C_test.txt' % svm_kernel, 'a') as out_file:
            out_file.write(test_score)


# SVMClassify.svm_classify_with_zi_vec(svm_kernel='linear')
# SVMClassify.svm_classify_with_zi_vec(svm_kernel='rbf')

for i in range(1, 50):
    para_C = 0.02*i
    SVMClassify.svm_classify_with_zi_vec_for_C(para_C=para_C)


