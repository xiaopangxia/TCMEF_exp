# -*- coding: utf-8 -*-
import sys
from gensim.models import word2vec
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from pyltp import Segmentor
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')


def svm_classify_with_word_sum(word_model_file, svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    这里的词项量模型用分词预料训练得
    词条表示用分词结果的词向量求和
    :return:
    """
    segmentor = Segmentor()
    segmentor.load("../word2vec_process/model/cws.model")

    word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/' + word_model_file)

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
    for pos_case in pos_case_list:
        case_words = segmentor.segment(pos_case)
        case_vec = np.zeros(50)
        is_useful = 0
        for word in case_words:
            try:
                case_vec += word_vec_model[unicode(word)]
                is_useful = 1
            except Exception, e:
                with open("not_in_vocabulary.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(word + '\n')

        case_vec = case_vec.tolist()
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
    for neg_case in neg_case_list:
        case_words = segmentor.segment(neg_case)
        case_vec = np.zeros(50)
        is_useful = 0
        for word in case_words:
            try:
                case_vec += word_vec_model[unicode(word)]
                is_useful = 1
            except Exception, e:
                with open("not_in_vocabulary.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(word + '\n')

        case_vec = case_vec.tolist()
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)

    x, y = np.split(case_train_set, (50,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')


def svm_classify_with_word_mosaic(word_model_file, svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    这里的词项量模型用分词预料训练得
    词条表示用分词结果的词向量拼接,
    每个词条按最多十个词算,不够的补零,超长的截断
    每个词条向量为50*10维+标签1维,总共501维
    :return:
    """
    segmentor = Segmentor()
    segmentor.load("../word2vec_process/model/cws.model")

    word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/' + word_model_file)

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
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
                with open("not_in_vocabulary.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(word + '\n')
        # 多退少补
        if len(case_vec) > 500:
            case_vec = case_vec[0:500]
        else:
            while (len(case_vec) < 500):
                case_vec.append(0)
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
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
                with open("not_in_vocabulary.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(word + '\n')

        # 多退少补
        if len(case_vec) > 500:
            case_vec = case_vec[0:500]
        else:
            while (len(case_vec) < 500):
                case_vec.append(0)
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)

    x, y = np.split(case_train_set, (500,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')


def svm_classify_with_zi_vec_sum(svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    这里用单字向量
    词条表示用字向量求和
    :return:
    """
    word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/wiki_hans_hant_single_zi_word2vec.model')

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
    for pos_case in pos_case_list:
        case_words = unicode(pos_case)
        case_vec = np.zeros(50)
        is_useful = 0
        for zi in case_words:
            try:
                case_vec += word_vec_model[unicode(zi)]
                is_useful = 1
            except Exception, e:
                with open("not_in_zi_table.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        case_vec = case_vec.tolist()
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
    for neg_case in neg_case_list:
        case_words = unicode(neg_case)
        case_vec = np.zeros(50)
        is_useful = 0
        for zi in case_words:
            try:
                case_vec += word_vec_model[unicode(zi)]
                is_useful = 1
            except Exception, e:
                with open("not_in_zi_table.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        case_vec = case_vec.tolist()
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)

    x, y = np.split(case_train_set, (50,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    # show_accuracy(y_hat, y_test, '测试集')


def svm_classify_with_zi_vec_mosaic(svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    这里用单字向量
    词条表示用字向量拼接,
    每个词条按最多二十个词算,不够的补零,超长的截断
    每个词条向量为50*20维+标签1维,总共1001维
    :return:
    """
    word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/wiki_hans_hant_single_zi_word2vec.model')

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
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
                with open("not_in_zi_table.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 1000:
            case_vec = case_vec[0:1000]
        else:
            while (len(case_vec) < 1000):
                case_vec.append(0)
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
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
                with open("not_in_zi_table.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 1000:
            case_vec = case_vec[0:1000]
        else:
            while (len(case_vec) < 1000):
                case_vec.append(0)
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)

    x, y = np.split(case_train_set, (1000,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)


def zi_2_stroke_count_dict():
    """
    由笔画库构建字典
    将单个汉子转换成五种笔画数构成的5维向量
    :return: 一个dict
    """
    stroke_dict = {}
    with open('zi2stroke.txt', 'r') as in_file:
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


def zi_2_stroke_seq_dict():
    """
    由笔画库构建字典
    将单个汉子转换成五种笔画的序列的30维向量
    字库中超过30维的只有12个字
    :return: 一个dict
    """
    stroke_dict = {}
    with open('zi2stroke.txt', 'r') as in_file:
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


def svm_classify_with_stroke_count_vec(svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    词条表示用字向量拼接,
    每个词条按最多二十个字算,不够的补零,超长的截断
    每个字表示成5维向量,每一维示对应笔画数,没有笔画数的就全0
    每个词条向量为5*20维+标签1维,总共101维
    :param svm_kernel:
    :return:
    """

    stroke_count_dict = zi_2_stroke_count_dict()

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
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
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 100:
            case_vec = case_vec[0:100]
        else:
            while (len(case_vec) < 100):
                case_vec.append(0)
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)


    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
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
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')
        # 多退少补
        if len(case_vec) > 100:
            case_vec = case_vec[0:100]
        else:
            while (len(case_vec) < 100):
                case_vec.append(0)
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)


    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)
    x, y = np.split(case_train_set, (100,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)



def svm_classify_with_stroke_seq_vec(svm_kernel='linear'):
    """
    用svm训练分类,得出分类准确率
    这里用单字向量
    词条表示用字向量拼接,
    每个词条按最多二十个字算,不够的补零,超长的截断
    每个字表示成30维向量,每一维示对应笔画,不足30划的补0
    每个词条向量为30*20维+标签1维,总共601维
    :param svm_kenel:
    :return:
    """
    stroke_seq_dict = zi_2_stroke_seq_dict()

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
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
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 600:
            case_vec = case_vec[0:600]
        else:
            while (len(case_vec) < 600):
                case_vec.append(0)
        case_vec.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
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
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')
        # 多退少补
        if len(case_vec) > 600:
            case_vec = case_vec[0:600]
        else:
            while (len(case_vec) < 600):
                case_vec.append(0)
        case_vec.append(0)
        if is_useful:
            neg_case_vec_list.append(case_vec)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)
    x, y = np.split(case_train_set, (600,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)


def svm_classify_with_joint_stroke(svm_kernel='linear'):
    """
    以拼接方式融合两种笔画特征向量
    做svm分类
    每个词条向量为30*20+5*20+1维,总共701维
    :return:
    """
    stroke_seq_dict = zi_2_stroke_seq_dict()
    stroke_count_dict = zi_2_stroke_count_dict()
    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
    for pos_case in pos_case_list:
        case_words = unicode(pos_case)
        case_vec = []
        case_vec_2 = []  # 笔画数特征
        is_useful = 0
        for zi in case_words:
            try:
                # 拼接
                case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                case_vec_2.extend(stroke_count_dict[zi.encode('utf8')])
                is_useful = 1
            except Exception, e:
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 600:
            case_vec = case_vec[0:600]
        else:
            while (len(case_vec) < 600):
                case_vec.append(0)
        if len(case_vec_2) > 100:
            case_vec_2 = case_vec_2[0:100]
        else:
            while (len(case_vec_2) < 100):
                case_vec_2.append(0)
        case_vec_joint = case_vec+case_vec_2
        case_vec_joint.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec_joint)


    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
    for neg_case in neg_case_list:
        case_words = unicode(neg_case)
        case_vec = []
        case_vec_2 = []
        is_useful = 0
        for zi in case_words:
            try:
                # 拼接
                case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                case_vec_2.extend(stroke_count_dict[zi.encode('utf8')])
                is_useful = 1
            except Exception, e:
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')
                    # 多退少补
                if len(case_vec) > 600:
                    case_vec = case_vec[0:600]
                else:
                    while (len(case_vec) < 600):
                        case_vec.append(0)
                if len(case_vec_2) > 100:
                    case_vec_2 = case_vec_2[0:100]
                else:
                    while (len(case_vec_2) < 100):
                        case_vec_2.append(0)
                case_vec_joint = case_vec + case_vec_2
                case_vec_joint.append(0)
                if is_useful:
                    neg_case_vec_list.append(case_vec_joint)


    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)
    x, y = np.split(case_train_set, (700,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    print classification_report(y_train, y_hat, target_names=['pos_case', 'neg'])
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    print classification_report(y_test, y_hat, target_names=['pos_case', 'neg'])


def svm_classify_joint_zi_stroke(svm_kernel='linear'):
    """
    以拼接方式融合两种笔画特征向量与单字向量
    做svm分类
    每个词条向量为30*20+5*20+50*20+1维,总共1701维
    :return:
    """
    stroke_seq_dict = zi_2_stroke_seq_dict()
    stroke_count_dict = zi_2_stroke_count_dict()
    word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/wiki_hans_hant_single_zi_word2vec.model')

    # 正样例词条不包括医家人名,负样例词条为开放域
    pos_case_list = []
    for i in range(1, 9):
        if i != 3:
            with open('../data_prepare/manual_filtered_data/pos_data/pos_case_list_%s.txt' % str(i), 'r') as in_file:
                for line in in_file.readlines():
                    pos_case_list.append(line.strip())
    # 正样例词向量列表
    pos_case_vec_list = []
    for pos_case in pos_case_list:
        case_words = unicode(pos_case)
        case_vec = []
        case_vec_2 = []  # 笔画数特征
        case_vec_3 = []  # 单字向量
        is_useful = 0
        for zi in case_words:
            try:
                # 拼接
                case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                case_vec_2.extend(stroke_count_dict[zi.encode('utf8')])
                case_vec_3.extend(word_vec_model[unicode(zi)].tolist())
                is_useful = 1
            except Exception, e:
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')

        # 多退少补
        if len(case_vec) > 600:
            case_vec = case_vec[0:600]
        else:
            while (len(case_vec) < 600):
                case_vec.append(0)
        if len(case_vec_2) > 100:
            case_vec_2 = case_vec_2[0:100]
        else:
            while (len(case_vec_2) < 100):
                case_vec_2.append(0)
        if len(case_vec_3) > 1000:
            case_vec_3 = case_vec_3[0:1000]
        else:
            while (len(case_vec_3) < 1000):
                case_vec_3.append(0)
        case_vec_joint = case_vec + case_vec_2+case_vec_3
        case_vec_joint.append(1)
        if is_useful:
            pos_case_vec_list.append(case_vec_joint)

    neg_case_list = []
    with open('../data_prepare/manual_filtered_data/open_domain_neg.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    # 负样例词训练列表
    neg_case_vec_list = []
    for neg_case in neg_case_list:
        case_words = unicode(neg_case)
        case_vec = []
        case_vec_2 = []
        case_vec_3 = []
        is_useful = 0
        for zi in case_words:
            try:
                # 拼接
                case_vec.extend(stroke_seq_dict[zi.encode('utf8')])
                case_vec_2.extend(stroke_count_dict[zi.encode('utf8')])
                case_vec_3.extend(word_vec_model[unicode(zi)].tolist())
                is_useful = 1
            except Exception, e:
                with open("not_in_stroke_dict.txt", 'a') as out_file:
                    # 记录缺失词汇
                    out_file.write(zi + '\n')
                    # 多退少补
                if len(case_vec) > 600:
                    case_vec = case_vec[0:600]
                else:
                    while (len(case_vec) < 600):
                        case_vec.append(0)
                if len(case_vec_2) > 100:
                    case_vec_2 = case_vec_2[0:100]
                else:
                    while (len(case_vec_2) < 100):
                        case_vec_2.append(0)
                if len(case_vec_3) > 1000:
                    case_vec_3 = case_vec_3[0:1000]
                else:
                    while (len(case_vec_3) < 1000):
                        case_vec_3.append(0)
                case_vec_joint = case_vec + case_vec_2+case_vec_3
                case_vec_joint.append(0)
                if is_useful:
                    neg_case_vec_list.append(case_vec_joint)

    case_vec_list = pos_case_vec_list + neg_case_vec_list
    # 列表转np类型
    case_train_set = np.array(case_vec_list)
    x, y = np.split(case_train_set, (1700,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    # svm分类,linear先不用核函数,rbf带卷积核
    clf = svm.SVC(C=0.8, kernel=svm_kernel, gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    print classification_report(y_train, y_hat, target_names=['neg', 'pos_case'])
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    print classification_report(y_test, y_hat, target_names=['neg', 'pos_case'])




# word_vec_model = word2vec.Word2Vec.load('../word2vec_process/model/wiki_hans_simple_word2vec.model')
# print type(word_vec_model[u'上火'])

# svm_classify_with_word_sum('wiki_hans_simple_word2vec.model')
# svm_classify_with_word_mosaic('wiki_hans_simple_word2vec.model')
# svm_classify_with_zi_vec_sum()
# svm_classify_with_zi_vec_mosaic()
# svm_classify_with_stroke_count_vec()
# svm_classify_with_stroke_seq_vec()
# svm_classify_with_joint_stroke()
# svm_classify_joint_zi_stroke()


