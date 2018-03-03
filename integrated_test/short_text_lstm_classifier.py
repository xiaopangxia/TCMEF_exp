# -*- coding: utf-8 -*-
import codecs
import sys
import numpy as np
import yaml
from sklearn.metrics import classification_report
import copy

np.random.seed(1337)
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.cross_validation import train_test_split
from pyltp import Segmentor
from gensim.models import word2vec
import random

reload(sys)
sys.setdefaultencoding('utf8')


# lstm做分类,采用经pnf过滤过的不带人名的数据
# lstm输入层要求示3D,所以词条的特征与此前直接拼接不同
# input_shape中第一维样本数不用设,第二维时间步也就是词条字数,第三维特征即单字或单词特征向量长度


class FeaturePrepare():
    """
    特征向量准备类
    """

    @classmethod
    def zi_2_stroke_count_dict(cls):
        """
        由笔画库构建字典
        将单个汉子转换成五种笔画数构成的5维向量
        :return: 一个dict
        """
        stroke_dict = {}
        with open('./data/zi2stroke.txt', 'r') as in_file:
            for line in in_file.readlines():
                zi = line.split()[0]
                stroke_str = line.split()[2]
                stroke_vec = []
                stroke_vec.append(stroke_str.count('1'))
                stroke_vec.append(stroke_str.count('2'))
                stroke_vec.append(stroke_str.count('3'))
                stroke_vec.append(stroke_str.count('4'))
                stroke_vec.append(stroke_str.count('5'))
                stroke_dict[zi] = stroke_vec
        return stroke_dict

    @classmethod
    def zi_2_stroke_seq_dict(cls):
        """
        由笔画库构建字典
        将单个汉子转换成五种笔画的序列的30维向量
        字库中超过30维的只有12个字
        :return: 一个dict
        """
        stroke_dict = {}
        with open('./data/zi2stroke.txt', 'r') as in_file:
            for line in in_file.readlines():
                zi = line.split()[0]
                stroke_str = line.split()[2]
                stroke_vec = []
                for stroke in stroke_str:
                    stroke_vec.append(int(stroke))
                while len(stroke_vec) < 30:
                    stroke_vec.append(0)
                stroke_vec = stroke_vec[0:30]
                stroke_dict[zi] = stroke_vec
        return stroke_dict

    @classmethod
    def load_case_set(cls):
        """
        从文件读取训练数据样本
        :param with_name: 正样例是否包含人名
        :return: dict
        """
        case_dict = {'pos_case': [], 'neg': []}  # 这里有个命名失误,pos与neg没有一致,后面多处引用牵涉较多
        with open('./pnf_result/pos_noname.txt', 'r') as in_file:
            for line in in_file.readlines():
                case_dict['pos_case'].append(line.strip())

        with open('./pnf_result/neg_noname.txt', 'r') as in_file:
            for line in in_file.readlines():
                case_dict['neg'].append(line.strip())
        random.shuffle(case_dict['pos_case'])
        return case_dict



    @classmethod
    def zi_vec_case_set(cls, word_model_file):
        """
        获取单字向量特征集,认为每个词条最多20个字,多的退,少的以0补
        每个词条被表示为50*20的二维列表
        :param word_model_file: 字向量模型文件
        :param with_name: 正样例是否包含人名
        :return: 一个字典{pos_case:{正例},neg:{负例}}
        """
        word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/' + word_model_file)
        case_dict = cls.load_case_set()
        zi_vec_case_dict = {}
        # 以字向量拼接的方式构建词条表示,1000维
        pos_case_list = case_dict['pos_case']
        pos_case_vec_dict = {}
        for pos_case in pos_case_list:
            case_words = unicode(pos_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(word_vec_model[unicode(zi)].tolist())
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_zi_table.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 50)

            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 50)
            if is_useful:
                pos_case_vec_dict[pos_case] = case_vec

        # 负样本
        neg_case_list = case_dict['neg']
        neg_case_vec_dict = {}
        for neg_case in neg_case_list:
            case_words = unicode(neg_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(word_vec_model[unicode(zi)].tolist())
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_zi_table.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 50)
            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 50)
            if is_useful:
                neg_case_vec_dict[neg_case] = case_vec

        zi_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        return zi_vec_case_dict

    @classmethod
    def stroke_count_vec_case_set(cls):
        """
        五种笔画数向量
        每个词条按最多二十个字算,不够的补零,超长的截断
        每个字表示成5维向量,每一维示对应笔画数,没有笔画数的就全0
        每个词条向量为5*20的二维列表
        :param with_name_False: 正样例是否包含人名
        :return: 一个字典
        """
        case_dict = cls.load_case_set()
        stroke_count_dict = cls.zi_2_stroke_count_dict()

        # 正样本
        pos_case_list = case_dict['pos_case']
        pos_case_vec_dict = {}
        for pos_case in pos_case_list:
            case_words = unicode(pos_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(stroke_count_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 5)

            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 5)
            if is_useful:
                pos_case_vec_dict[pos_case] = case_vec

        # 负样本
        neg_case_list = case_dict['neg']
        neg_case_vec_dict = {}
        for neg_case in neg_case_list:
            case_words = unicode(neg_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(stroke_count_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 5)
            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 5)
            if is_useful:
                neg_case_vec_dict[neg_case] = case_vec

        stroke_count_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        return stroke_count_vec_case_dict

    @classmethod
    def stroke_seq_vec_case_set(cls):
        """
        笔画序列向量
        每个词条按最多二十个字算,不够的补零,超长的截断
        每个字表示成30维向量,每一维示对应笔画,不足30划的补0
        每个词条向量为30*20的二维列表
        :param with_name: 正样例是否包含人名
        :return: 一个字典
        """
        case_dict = cls.load_case_set()
        stroke_seq_dict = cls.zi_2_stroke_seq_dict()
        # 正样本
        pos_case_list = case_dict['pos_case']
        pos_case_vec_dict = {}
        for pos_case in pos_case_list:
            case_words = unicode(pos_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(stroke_seq_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 30)

            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 30)
            if is_useful:
                pos_case_vec_dict[pos_case] = case_vec

        # 负样本
        neg_case_list = case_dict['neg']
        neg_case_vec_dict = {}
        for neg_case in neg_case_list:
            case_words = unicode(neg_case)
            case_vec = []
            is_useful = 0
            for zi in case_words:
                try:
                    # 拼接
                    case_vec.append(stroke_seq_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
                    case_vec.append([0] * 30)
            # 多退少补
            if len(case_vec) > 20:
                case_vec = case_vec[0:20]
            else:
                while (len(case_vec) < 20):
                    case_vec.append([0] * 30)
            if is_useful:
                neg_case_vec_dict[neg_case] = case_vec

        stroke_seq_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        return stroke_seq_vec_case_dict


class lstmClassify():
    """
    lstm分类,用的keras神经网络模块
    """

    ##定义网络结构
    @classmethod
    def train_lstm(cls, x_train, y_train, x_test, y_test, time_step, feature_d, log_file, batch_size=10, n_epoch=5):
        """
        lstm训练模块
        :return:
        """
        print 'Defining a Simple Keras Model...'
        model = Sequential()
        # 特征已经整理完了,所以不加embedding
        model.add(LSTM(output_dim=50, input_shape=(time_step, feature_d), activation='sigmoid',
                       inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        print 'Compiling the Model...'
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print "Train..."
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1,
                  validation_data=(x_train, y_train))

        print "Evaluate..."
        score = model.evaluate(x_test, y_test, batch_size=batch_size)

        yaml_string = model.to_yaml()
        with open('model_file/' + log_file + 'lstm.yml', 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights('model_file/' + log_file + '_lstm.h5')
        print 'Test score:', score

        return model

    @classmethod
    def lstm_classify_core(cls, pos_vec_list, neg_vec_list_large, time_step, feature_d, log_file):
        """
        lstm分类部件
        :param pos_vec_list:
        :param neg_vec_list_large:
        :param time_step:时间步,也就是默认的词条中字数或词数
        :param feature_d:特征向量长度
        :param log_file:
        :return:
        """
        neg_vec_list = neg_vec_list_large
        case_vec_list = pos_vec_list + neg_vec_list
        # 上面按0.8采样还是有问题,测试集正负样例极不均衡,统一用1500个
        x_neg = np.array(neg_vec_list)
        y_neg = np.array([0] * len(neg_vec_list))
        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, random_state=11,
                                                                            test_size=0.2)
        x_pos = np.array(pos_vec_list)
        y_pos = np.array([1] * len(pos_vec_list))
        x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, random_state=11,
                                                                            test_size=0.2)

        # 拼接训练集,测试集
        x_train = np.concatenate((x_neg_train, x_pos_train), axis=0)
        y_train = np.concatenate((y_neg_train, y_pos_train), axis=0)
        x_test = np.concatenate((x_neg_test, x_pos_test), axis=0)
        y_test = np.concatenate((y_neg_test, y_pos_test), axis=0)

        print x_train.shape

        # svm分类,linear先不用核函数,rbf带卷积核
        lstm_model = cls.train_lstm(x_train, y_train, x_test, y_test, time_step=time_step, feature_d=feature_d,
                                    log_file=log_file, batch_size=10, n_epoch=10)
        y_hat = lstm_model.predict(x_train)
        for y in y_hat:
            if y[0] >= 0.5:
                y[0] = 1
            else:
                y[0] = 0
        train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./stlc_result/' + log_file + '_train.txt', 'a') as out_file:
            out_file.write(train_socre)

        y_hat = lstm_model.predict(x_test)
        for y in y_hat:
            if y[0] >= 0.5:
                y[0] = 1
            else:
                y[0] = 0
        test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
        with open('./stlc_result/' + log_file + '_test.txt', 'a') as out_file:
                out_file.write(test_score)


    @classmethod
    def lstm_classify_with_zi_stroke_vec(cls, log_file):
        """
        联合单字向量和笔画特征做分类,
        每个字feature_d为50+30+5
        :param cls:
        :param log_file:
        :param with_name:
        :return:
        """
        stroke_seq_vec_case_set = FeaturePrepare.stroke_seq_vec_case_set()
        stroke_count_vec_case_set = FeaturePrepare.stroke_count_vec_case_set()
        zi_vec_case_set = FeaturePrepare.zi_vec_case_set('wiki_hans_hant_single_zi_word2vec.model')
        pos_vec_list = []
        for item in stroke_seq_vec_case_set['pos_case']:
            try:
                item_strok_seq = copy.deepcopy(stroke_seq_vec_case_set['pos_case'][item])
                item_strok_count = copy.deepcopy(stroke_count_vec_case_set['pos_case'][item])
                item_zi = copy.deepcopy(zi_vec_case_set['pos_case'][item])
                item_feature = []
                for i in range(20):
                    item_feature.append(item_zi[i] + item_strok_seq[i] + item_strok_count[i])
                pos_vec_list.append(item_feature)
            except Exception, e:
                pass
        neg_vec_list_large = []
        for item in stroke_seq_vec_case_set['neg']:
            try:
                item_strok_seq = copy.deepcopy(stroke_seq_vec_case_set['neg'][item])
                item_strok_count = copy.deepcopy(stroke_count_vec_case_set['neg'][item])
                item_zi = copy.deepcopy(zi_vec_case_set['neg'][item])
                item_feature = []
                for i in range(20):
                    item_feature.append(item_zi[i] + item_strok_seq[i] + item_strok_count[i])
                neg_vec_list_large.append(item_feature)
            except Exception, e:
                pass

        cls.lstm_classify_core(pos_vec_list, neg_vec_list_large, time_step=20, feature_d=85, log_file=log_file)


lstmClassify.lstm_classify_with_zi_stroke_vec(log_file='stlc_report_2')





