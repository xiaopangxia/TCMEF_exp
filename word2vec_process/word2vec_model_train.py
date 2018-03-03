# -*- coding: utf-8 -*-
import sys
from gensim.models import word2vec

reload(sys)
sys.setdefaultencoding('utf8')


# 用分好词的语料训练wordvec模型

def train_wiki_hans_simple():
    """
    用wiki简体中文语料ltp简单分词后的结果做词向量训练
    :return:
    """
    sentences = word2vec.Text8Corpus(u'corpus/wiki_hans_seg.txt')
    model = word2vec.Word2Vec(sentences, min_count=5, size=50)
    model.save('model/wiki_hans_simple_word2vec.model')


def train_wiki_hant_simple():
    """
    用wiki繁体中文语料ltp简单分词后的结果做词向量训练
    :return:
    """
    sentences = word2vec.Text8Corpus(u'corpus/wiki_hant_seg.txt')
    model = word2vec.Word2Vec(sentences, min_count=5, size=50)
    model.save('model/wiki_hant_simple_word2vec.model')


def train_wiki_hans_hant_simple():
    """
    用wiki简体繁体中文语料ltp简单分词后的结果做词向量训练
    :return:
    """
    sentences = word2vec.Text8Corpus(u'corpus/wiki_hans_hant_seg.txt')
    model = word2vec.Word2Vec(sentences, min_count=5, size=50)
    model.save('model/wiki_hans_hant_simple_word2vec.model')



def train_wiki_hans_hant_single_zi_simple():
    """
    用wiki简体繁体单字划分结果做字向量训练
    :return:
    """
    sentences = word2vec.Text8Corpus(u'corpus/wiki_hans_hant_single_zi.txt')
    model = word2vec.Word2Vec(sentences, min_count=5, size=50)
    model.save('model/wiki_hans_hant_single_zi_word2vec.model')


def train_wiki_zi_with_name():
    """
    维基预料,带大量中文人名,用于人名识别
    :return:
    """
    sentences = word2vec.Text8Corpus(u'corpus/wiki_zi_with_name.txt')
    model = word2vec.Word2Vec(sentences, min_count=5, size=50)
    model.save('model/wiki_zi_with_name_word2vec.model')


# train_wiki_hans_simple()
# train_wiki_hant_simple()
# train_wiki_hans_hant_simple()
# train_wiki_hans_hant_single_zi_simple()
train_wiki_zi_with_name()


