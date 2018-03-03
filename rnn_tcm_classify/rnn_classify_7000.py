# -*- coding: utf-8 -*-
import sys
from gensim.models import word2vec
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from pyltp import Segmentor
import numpy as np
import random
import yaml
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation


reload(sys)
sys.setdefaultencoding('utf8')


# rnn做分类,特征提取的代码同svm一样
# 采用了7000条正样例数据集

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
    def load_case_set(cls, with_name=False):
        """
        从文件读取训练数据样本
        :param with_name: 正样例是否包含人名
        :return: dict
        """
        case_dict = {'pos_case': [], 'neg': []}  # 这里有个命名失误,pos与neg没有一致,后面多处引用牵涉较多
        if with_name:
            with open('../data_prepare/manual_filtered_data/pos_data_7000/pos_case_list_with_name.txt', 'r') as in_file:
                for line in in_file.readlines():
                    case_dict['pos_case'].append(line.strip())
        else:
            with open('../data_prepare/manual_filtered_data/pos_data_7000/pos_case_list_without_name.txt', 'r') as in_file:
                for line in in_file.readlines():
                    case_dict['pos_case'].append(line.strip())
        if with_name:
            with open('../data_prepare/manual_filtered_data/open_domain_neg_large_with_name.txt', 'r') as in_file:
                for line in in_file.readlines():
                    case_dict['neg'].append(line.strip())
        else:
            with open('../data_prepare/manual_filtered_data/open_domain_neg_large.txt', 'r') as in_file:
                for line in in_file.readlines():
                    case_dict['neg'].append(line.strip())
        random.shuffle(case_dict['pos_case'])
        return case_dict

    @classmethod
    def word_vec_case_set(cls, word_model_file, with_name=False, merge_by='mosaic'):
        """
        获取词向量特征集,认为词条最多10个词
        如果以mosaic方式,每个词条被表示为50*10=500维
        如果以sum方式,每个词条被表示为50维
        :param word_model_file: 词向量模型文件
        :param with_name: 正样例是否包含人名
        :param merge_by: 词条中词项量的结合方式,mosaic或sum
        :return: 一个字典{pos_case:{正例},neg:{负例}}
        """
        segmentor = Segmentor()
        segmentor.load("../word2vec_process/model/cws.model")
        word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/' + word_model_file)
        case_dict = cls.load_case_set(with_name)
        word_vec_case_dict = {}

        if merge_by == 'mosaic':
            # 以词向量拼接的方式构建词条表示,500维
            pos_case_list = case_dict['pos_case']
            pos_case_vec_dict = {}
            for pos_case in pos_case_list:
                case_words = segmentor.segment(pos_case)
                case_vec = []
                is_useful = 0
                for word in case_words:
                    try:
                        # 拼接
                        case_vec.extend(word_vec_model[unicode(word)].tolist())
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_vocabulary.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(word + '\n')
                # 多退少补
                if len(case_vec) > 500:
                    case_vec = case_vec[0:500]
                else:
                    while (len(case_vec) < 500):
                        case_vec.append(0)
                if is_useful:
                    pos_case_vec_dict[pos_case] = case_vec
            # 负样本
            neg_case_list = case_dict['neg']
            neg_case_vec_dict = {}
            for neg_case in neg_case_list:
                case_words = segmentor.segment(neg_case)
                case_vec = []
                is_useful = 0
                for word in case_words:
                    try:
                        # 拼接
                        case_vec.extend(word_vec_model[unicode(word)].tolist())
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_vocabulary.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(word + '\n')
                # 多退少补
                if len(case_vec) > 500:
                    case_vec = case_vec[0:500]
                else:
                    while (len(case_vec) < 500):
                        case_vec.append(0)
                if is_useful:
                    neg_case_vec_dict[neg_case] = case_vec
            word_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        elif merge_by == 'sum':
            # 以词向量加和的方式构建词条表示,50维
            pos_case_list = case_dict['pos_case']
            pos_case_vec_dict = {}
            for pos_case in pos_case_list:
                case_words = segmentor.segment(pos_case)
                case_vec = np.zeros(50)
                is_useful = 0
                for word in case_words:
                    try:
                        case_vec += word_vec_model[unicode(word)]
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_vocabulary.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(word + '\n')

                case_vec = case_vec.tolist()
                if is_useful:
                    pos_case_vec_dict[pos_case] = case_vec
            # 负样本
            neg_case_list = case_dict['neg']
            neg_case_vec_dict = {}
            for neg_case in neg_case_list:
                case_words = segmentor.segment(neg_case)
                case_vec = np.zeros(50)
                is_useful = 0
                for word in case_words:
                    try:
                        case_vec += word_vec_model[unicode(word)]
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_vocabulary.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(word + '\n')

                case_vec = case_vec.tolist()
                if is_useful:
                    neg_case_vec_dict[neg_case] = case_vec

            word_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}
        return word_vec_case_dict

    @classmethod
    def zi_vec_case_set(cls, word_model_file, with_name=False, merge_by='mosaic'):
        """
        获取单字向量特征集,认为每个词条最多20个字,多的退,少的以0补
        如果以mosaic方式,每个词条被表示为50*20=1000维
        如果以sum方式,每个词条被表示为50维
        :param word_model_file: 字向量模型文件
        :param with_name: 正样例是否包含人名
        :param merge_by: 词条中词项量的结合方式,mosaic或sum
        :return: 一个字典{pos_case:{正例},neg:{负例}}
        """
        word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/' + word_model_file)
        case_dict = cls.load_case_set(with_name)
        zi_vec_case_dict = {}
        if merge_by == 'mosaic':
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
                        case_vec.extend(word_vec_model[unicode(zi)].tolist())
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_zi_table.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(zi + '\n')

                # 多退少补
                if len(case_vec) > 1000:
                    case_vec = case_vec[0:1000]
                else:
                    while (len(case_vec) < 1000):
                        case_vec.append(0)
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
                        case_vec.extend(word_vec_model[unicode(zi)].tolist())
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_zi_table.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(zi + '\n')

                # 多退少补
                if len(case_vec) > 1000:
                    case_vec = case_vec[0:1000]
                else:
                    while (len(case_vec) < 1000):
                        case_vec.append(0)
                if is_useful:
                    neg_case_vec_dict[neg_case] = case_vec

            zi_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        elif merge_by == 'sum':
            # 以字向量加和的方式构建词条表示,50维
            pos_case_list = case_dict['pos_case']
            pos_case_vec_dict = {}
            for pos_case in pos_case_list:
                case_words = unicode(pos_case)
                case_vec = np.zeros(50)
                is_useful = 0
                for zi in case_words:
                    try:
                        case_vec += word_vec_model[unicode(zi)]
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_zi_table.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(zi + '\n')

                case_vec = case_vec.tolist()
                if is_useful:
                    pos_case_vec_dict[pos_case] = case_vec
            # 负样本
            neg_case_list = case_dict['neg']
            neg_case_vec_dict = {}
            for neg_case in neg_case_list:
                case_words = unicode(neg_case)
                case_vec = np.zeros(50)
                is_useful = 0
                for zi in case_words:
                    try:
                        case_vec += word_vec_model[unicode(zi)]
                        is_useful = 1
                    except Exception, e:
                        with open("./data/not_in_zi_table.txt", 'a') as out_file:
                            # 记录缺失词汇
                            out_file.write(zi + '\n')
                case_vec = case_vec.tolist()
                if is_useful:
                    neg_case_vec_dict[neg_case]=case_vec

            zi_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}


        return zi_vec_case_dict

    @classmethod
    def stroke_count_vec_case_set(cls, with_name=False):
        """
        五种笔画数向量
        每个词条按最多二十个字算,不够的补零,超长的截断
        每个字表示成5维向量,每一维示对应笔画数,没有笔画数的就全0
        每个词条向量为5*20维,总共100维
        :param with_name_False: 正样例是否包含人名
        :return: 一个字典
        """
        case_dict = cls.load_case_set(with_name)
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
                    case_vec.extend(stroke_count_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./data/not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
            # 多退少补
            if len(case_vec) > 100:
                case_vec = case_vec[0:100]
            else:
                while (len(case_vec) < 100):
                    case_vec.append(0)
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
                    case_vec.extend(stroke_count_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./data/not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
            # 多退少补
            if len(case_vec) > 100:
                case_vec = case_vec[0:100]
            else:
                while (len(case_vec) < 100):
                    case_vec.append(0)
            if is_useful:
                neg_case_vec_dict[neg_case] = case_vec

        stroke_count_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        return stroke_count_vec_case_dict

    @classmethod
    def stroke_seq_vec_case_set(cls, with_name=False):
        """
        笔画序列向量
        词条表示用字向量拼接,
        每个词条按最多二十个字算,不够的补零,超长的截断
        每个字表示成30维向量,每一维示对应笔画,不足30划的补0
        每个词条向量为30*20维,总共600维
        :param with_name: 正样例是否包含人名
        :return: 一个字典
        """
        case_dict = cls.load_case_set(with_name)
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
                    case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./data/not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')

            # 多退少补
            if len(case_vec) > 600:
                case_vec = case_vec[0:600]
            else:
                while (len(case_vec) < 600):
                    case_vec.append(0)
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
                    case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                    is_useful = 1
                except Exception, e:
                    with open("./data/not_in_stroke_dict.txt", 'a') as out_file:
                        # 记录缺失词汇
                        out_file.write(zi + '\n')
            # 多退少补
            if len(case_vec) > 600:
                case_vec = case_vec[0:600]
            else:
                while (len(case_vec) < 600):
                    case_vec.append(0)
            if is_useful:
                neg_case_vec_dict[neg_case] = case_vec

        stroke_seq_vec_case_dict = {'pos_case': pos_case_vec_dict, 'neg': neg_case_vec_dict}

        return stroke_seq_vec_case_dict


class rnnClassify():
    """
    rnn分类,用的keras神经网络模块
    """

    ##定义网络结构
    @classmethod
    def train_rnn(cls, n_dim, x_train, y_train, x_test, y_test, log_file, batch_size=10, n_epoch=5):
        """
        lstm训练模块
        :return:
        """
        print 'Defining a Simple Keras Model...'
        model = Sequential()
        # 特征已经整理完了,所以不加embedding
        model.add(Dense(n_dim/2, input_shape=(n_dim,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_dim/4, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_dim/8, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        print 'Compiling the Model...'
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print "Train..."
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_train, y_train))

        print "Evaluate..."
        score = model.evaluate(x_test, y_test, batch_size=batch_size)

        yaml_string = model.to_yaml()
        with open('rnn_data/'+log_file+'lstm.yml', 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights('rnn_data/'+log_file+'_lstm.h5')
        print 'Test score:', score

        return model


    @classmethod
    def rnn_classify_core(cls, pos_vec_list, neg_vec_list_large, feature_d, log_file):
        """
        lstm分类部件
        :param pos_vec_list:
        :param neg_vec_list_large:
        :param feature_d:
        :param log_file:
        :return:
        """
        for i in range(10):
            # 逐轮增加负样本数,从7000到12000
            neg_vec_list = neg_vec_list_large[0:7000 + i * 500]
            case_vec_list = pos_vec_list + neg_vec_list


            # 上面按0.8采样还是有问题,测试集正负样例极不均衡,统一用1500个
            neg_set = np.array(neg_vec_list)
            x_neg, y_neg = np.split(neg_set, (feature_d,), axis=1)
            x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, random_state=11,
                                                                                test_size=1500)

            pos_set = np.array(pos_vec_list)
            x_pos, y_pos = np.split(pos_set, (feature_d,), axis=1)
            x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, random_state=11,
                                                                                test_size=1500)

            # 拼接训练集,测试集
            x_train = np.concatenate((x_neg_train, x_pos_train), axis=0)
            y_train = np.concatenate((y_neg_train, y_pos_train), axis=0)
            x_test = np.concatenate((x_neg_test, x_pos_test), axis=0)
            y_test = np.concatenate((y_neg_test, y_pos_test), axis=0)


            # svm分类,linear先不用核函数,rbf带卷积核
            lstm_model = cls.train_rnn(feature_d, x_train, y_train, x_test, y_test, log_file, batch_size=10, n_epoch=5)
            y_hat = lstm_model.predict(x_train)
            for y in y_hat:
                if y[0]>=0.5:
                    y[0] = 1
                else:
                    y[0] = 0
            train_socre = str(classification_report(y_train, y_hat, target_names=['neg', 'pos_case'], digits=4))
            with open('./classify_result_7000/' + log_file + '_train.txt', 'a') as out_file:
                out_file.write(train_socre)


            y_hat = lstm_model.predict(x_test)
            for y in y_hat:
                if y[0]>=0.5:
                    y[0] = 1
                else:
                    y[0] = 0
            test_score = str(classification_report(y_test, y_hat, target_names=['neg', 'pos_case'], digits=4))
            with open('./classify_result_7000/' + log_file + '_test.txt', 'a') as out_file:
                out_file.write(test_score)




    @classmethod
    def rnn_classify_with_word_vec(cls, feature_d, log_file, with_name=False, merge_by='mosaic'):
        """
        利用词语向量特征做dt分类
        :param feature_d:特征向量维度
        :param with_name:
        :param merge_by:
        :return:
        """
        word_vec_case_set = FeaturePrepare.word_vec_case_set('wiki_hans_simple_word2vec.model', with_name=with_name,
                                                             merge_by=merge_by)
        pos_vec_list = []
        for item in word_vec_case_set['pos_case']:
            pos_vec_list.append(word_vec_case_set['pos_case'][item])
        neg_vec_list_large = []
        for item in word_vec_case_set['neg']:
            neg_vec_list_large.append(word_vec_case_set['neg'][item])
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)

    @classmethod
    def rnn_classify_with_zi_vec(cls, feature_d, log_file, with_name=False, merge_by='mosaic'):
        """
        利用单字向量特征做rnn分类
        :param feature_d: 
        :param log_file: 
        :param with_name: 
        :param merge_by: 
        :return: 
        """
        zi_vec_case_set = FeaturePrepare.zi_vec_case_set('wiki_hans_hant_single_zi_word2vec.model', with_name=with_name,
                                                         merge_by=merge_by)
        pos_vec_list = []
        for item in zi_vec_case_set['pos_case']:
            pos_vec_list.append(zi_vec_case_set['pos_case'][item])
        neg_vec_list_large = []
        for item in zi_vec_case_set['neg']:
            neg_vec_list_large.append(zi_vec_case_set['neg'][item])
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)

    @classmethod
    def rnn_classify_with_stroke_count_vec(cls, feature_d, log_file, with_name=False):
        """
        利用单字笔画数特征做rnn分类
        :param feature_d: 
        :param log_file: 
        :param with_name: 
        :return: 
        """
        stroke_count_vec_case_set = FeaturePrepare.stroke_count_vec_case_set(with_name)
        pos_vec_list = []
        for item in stroke_count_vec_case_set['pos_case']:
            pos_vec_list.append(stroke_count_vec_case_set['pos_case'][item])
        neg_vec_list_large = []
        for item in stroke_count_vec_case_set['neg']:
            neg_vec_list_large.append(stroke_count_vec_case_set['neg'][item])
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)

    @classmethod
    def rnn_classify_with_stroke_seq_vec(cls, feature_d, log_file, with_name=False):
        """
        利用单字笔画序列特征做rnn分类
        :param feature_d: 
        :param log_file: 
        :param with_name: 
        :return: 
        """
        stroke_seq_vec_case_set = FeaturePrepare.stroke_seq_vec_case_set(with_name)
        pos_vec_list = []
        for item in stroke_seq_vec_case_set['pos_case']:
            pos_vec_list.append(stroke_seq_vec_case_set['pos_case'][item])
        neg_vec_list_large = []
        for item in stroke_seq_vec_case_set['neg']:
            neg_vec_list_large.append(stroke_seq_vec_case_set['neg'][item])
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)

    @classmethod
    def rnn_classify_with_stroke_joint_vec(cls, feature_d, log_file, with_name=False):
        """
        结合笔画数与笔画序列特征,feature_d为100+600=700
        :param feature_d: 
        :param log_file: 
        :param with_name: 
        :return: 
        """
        stroke_seq_vec_case_set = FeaturePrepare.stroke_seq_vec_case_set(with_name)
        stroke_count_vec_case_set = FeaturePrepare.stroke_count_vec_case_set(with_name)
        pos_vec_list = []
        for item in stroke_seq_vec_case_set['pos_case']:
            try:
                stroke_seq_vec_case_set['pos_case'][item].extend(stroke_count_vec_case_set['pos_case'][item])
                pos_vec_list.append(stroke_seq_vec_case_set['pos_case'][item])
            except Exception, e:
                pass
        neg_vec_list_large = []
        for item in stroke_seq_vec_case_set['neg']:
            try:
                stroke_seq_vec_case_set['neg'][item].extend(stroke_count_vec_case_set['neg'][item])
                neg_vec_list_large.append(stroke_seq_vec_case_set['neg'][item])
            except Exception, e:
                pass
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)

    @classmethod
    def rnn_classify_with_zi_stroke_vec(cls, feature_d, log_file, with_name=False, merge_by='mosaic'):
        """
        联合单字向量和笔画特征做分类,
        字向量拼接1000+600+100=1700维,字向量加和50+600+100=750维
        :param cls:
        :param feature_d:
        :param log_file:
        :param with_name:
        :param merge_by:
        :return:
        """
        stroke_seq_vec_case_set = FeaturePrepare.stroke_seq_vec_case_set(with_name)
        stroke_count_vec_case_set = FeaturePrepare.stroke_count_vec_case_set(with_name)
        zi_vec_case_set = FeaturePrepare.zi_vec_case_set('wiki_hans_hant_single_zi_word2vec.model', with_name=with_name,
                                                         merge_by=merge_by)
        pos_vec_list = []
        for item in stroke_seq_vec_case_set['pos_case']:
            try:
                stroke_seq_vec_case_set['pos_case'][item].extend(stroke_count_vec_case_set['pos_case'][item])
                stroke_seq_vec_case_set['pos_case'][item].extend(zi_vec_case_set['pos_case'][item])
                pos_vec_list.append(stroke_seq_vec_case_set['pos_case'][item])
            except Exception, e:
                pass
        neg_vec_list_large = []
        for item in stroke_seq_vec_case_set['neg']:
            try:
                stroke_seq_vec_case_set['neg'][item].extend(stroke_count_vec_case_set['neg'][item])
                stroke_seq_vec_case_set['neg'][item].extend(zi_vec_case_set['neg'][item])
                neg_vec_list_large.append(stroke_seq_vec_case_set['neg'][item])
            except Exception, e:
                pass
        # 加标签
        for vec in pos_vec_list:
            vec.append(1)
        for vec in neg_vec_list_large:
            vec.append(0)

        cls.rnn_classify_core(pos_vec_list, neg_vec_list_large, feature_d, log_file)
    





# rnnClassify.rnn_classify_with_word_vec(feature_d=500, log_file='word_noname_mosaic')
# rnnClassify.rnn_classify_with_word_vec(feature_d=500, log_file='word_name_mosaic', with_name=True)
# rnnClassify.rnn_classify_with_word_vec(feature_d=50, log_file='word_noname_sum', merge_by='sum')
# rnnClassify.rnn_classify_with_word_vec(feature_d=50, log_file='word_name_sum', with_name=True, merge_by='sum')


# rnnClassify.rnn_classify_with_zi_vec(feature_d=1000, log_file='zi_noname_mosaic')
# rnnClassify.rnn_classify_with_zi_vec(feature_d=1000, log_file='zi_name_mosaic', with_name=True)
# rnnClassify.rnn_classify_with_zi_vec(feature_d=50, log_file='zi_noname_sum', merge_by='sum')
# rnnClassify.rnn_classify_with_zi_vec(feature_d=50, log_file='zi_name_sum', with_name=True, merge_by='sum')



# rnnClassify.rnn_classify_with_stroke_count_vec(feature_d=100, log_file='stroke_count_noname_mosaic')
# rnnClassify.rnn_classify_with_stroke_count_vec(feature_d=100, log_file='stroke_count_name_mosaic', with_name=True)
#
#
#
# rnnClassify.rnn_classify_with_stroke_seq_vec(feature_d=600, log_file='stroke_seq_noname_mosaic')
# rnnClassify.rnn_classify_with_stroke_seq_vec(feature_d=600, log_file='stroke_seq_name_mosaic', with_name=True)
#
#
# rnnClassify.rnn_classify_with_stroke_joint_vec(feature_d=700, log_file='stroke_joint_noname_mosaic')
# rnnClassify.rnn_classify_with_stroke_joint_vec(feature_d=700, log_file='stroke_joint_name_mosaic', with_name=True)



# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=1700, log_file='zi_stroke_noname_mosaic')
# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=1700, log_file='zi_stroke_name_mosaic', with_name=True)
# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=750, log_file='zi_stroke_noname_sum', merge_by='sum')
# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=750, log_file='zi_stroke_name_sum', with_name=True, merge_by='sum')



# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=1700, log_file='zi_stroke_name_mosaic_special', with_name=True)
# rnnClassify.rnn_classify_with_zi_stroke_vec(feature_d=750, log_file='zi_stroke_name_sum_special', with_name=True, merge_by='sum')


