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

# 人名过滤
# 数据说明:
#   训练数据:
#       name.txt中有5000条人名词条
#       opendomain_noname.txt里有5000条开放域词条,不包含人名
#   用于过滤的目标数据:
#       7400条正例数据
#       open_domain_neg_large_with_name

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
            item_vec_list.append(case_vec)

        return item_vec_list


class PersonNameFilter():
    """
    svm人名分类方法类
    """
    @classmethod
    def svm_classify_with_zi_vec(cls, svm_kernel='linear'):
        """
        利用人名词条数据和非人名词条数据训练svm模型进行分类
        此函数内的pos,neg指是否为人名
        :return:分离器模型
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
        x_train, y_train = np.split(case_train_set, (500,), axis=1)


        # svm分类,linear先不用核函数,rbf带卷积核
        clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel())

        return clf


    @classmethod
    def separate_target_data(cls):
        """
        分割目标数据,将pos,neg分成pos_name,pos_noname,neg_name,neg_noname
        此函数内的pos,neg指是否为TCM相关
        :return:
        """
        # 得到模型
        pnf_model = cls.svm_classify_with_zi_vec()
        
        # 正样本分割
        pos_case_list = []
        with open('./data/pos_case_list_with_name.txt', 'r') as in_file:
            for line in in_file.readlines():
                pos_case_list.append(line.strip())
        pos_case_vec_list = FaeturePrepare.wiki_item_2_vec(pos_case_list)
        pos_target_set = np.array(pos_case_vec_list)
        y_predict = pnf_model.predict(pos_target_set)
        print len(pos_case_list), len(y_predict) # 看长度是否一致
        with open('./pnf_result/pos_name.txt', 'a') as out_file_name:
            with open('./pnf_result/pos_noname.txt', 'a') as out_file_noname:
                y_list = y_predict.tolist()
                for i in range(len(y_list)):
                    if y_list[i] < 0.5:
                        out_file_noname.write(pos_case_list[i]+'\n')
                    else:
                        out_file_name.write(pos_case_list[i]+'\n')
        # 负样本分割
        neg_case_list = []
        with open('./data/open_domain_neg_large_12000.txt', 'r') as in_file:
            for line in in_file.readlines():
                neg_case_list.append(line.strip())
        neg_case_vec_list = FaeturePrepare.wiki_item_2_vec(neg_case_list)
        neg_target_set = np.array(neg_case_vec_list)
        y_predict = pnf_model.predict(neg_target_set)
        print len(neg_case_list), len(neg_case_vec_list), len(y_predict)  # 看长度是否一致
        with open('./pnf_result/neg_name.txt', 'a') as out_file_name:
            with open('./pnf_result/neg_noname.txt', 'a') as out_file_noname:
                y_list = y_predict.tolist()
                for i in range(len(y_list)):
                    if y_list[i] < 0.5:
                        out_file_noname.write(neg_case_list[i] + '\n')
                    else:
                        out_file_name.write(neg_case_list[i] + '\n')


PersonNameFilter.separate_target_data()

