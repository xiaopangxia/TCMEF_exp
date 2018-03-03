# -*- coding: utf-8 -*-
from pyltp import Segmentor
import jieba
import sys

segmentor = Segmentor()
segmentor.load("model/cws.model")

reload(sys)
sys.setdefaultencoding('utf8')

# words = segmentor.segment("元芳你怎么看")
# print words
# print "|".join(words)
# segmentor.release()



def simple_word_segment():
    """
    对wiki语料简体中文内容按照ltp默认方式分词,
    也就是未添加词典也未使用个性化分词
    :return:
    """
    with open('corpus/wiki_hans.txt', 'r') as in_file:
        count = 0
        for line in in_file.readlines():
            count += 1
            if count % 1000 == 0:
                print count
            try:
                words = segmentor.segment(line.strip())
                out_line = " ".join(words)
                with open('corpus/wiki_hans_seg.txt', 'a') as out_file:
                    out_file.write(out_line+'\n')
            except Exception, e:
                print e


def jieba_tight_word_segment():
    """
    用jieba对维基语料做细粒度分词
    :return:
    """
    with open('corpus/wiki_hans.txt', 'r') as in_file:
        count = 0
        for line in in_file.readlines():
            count += 1
            if count % 1000 == 0:
                print count
            try:
                words = jieba.lcut(unicode(line.strip()), cut_all=True, HMM=True)
                out_line = " ".join(words)
                with open('corpus/wiki_hans_jieba_tight_seg.txt', 'a') as out_file:
                    out_file.write(out_line+'\n')
            except Exception, e:
                print e



def single_zi_segment():
    """
    再ltp分词基础上对每一个单字分割
    :return:
    """
    # with open('corpus/wiki_hans_seg.txt', 'r') as in_file:
    #     count = 0
    #     for line in in_file.readlines():
    #         count += 1
    #         if count % 1000 == 0:
    #             print count
    #         try:
    #             seg_list = unicode(line).strip().split()
    #             new_line = ''
    #             for seg in seg_list:
    #                 for char in seg:
    #                     new_line += (char+' ')
    #
    #             with open('corpus/wiki_hans_single_zi_seg.txt', 'a') as out_file:
    #                 out_file.write(new_line.strip() + '\n')
    #         except Exception, e:
    #             print e
    #
    # with open('corpus/wiki_hant_seg.txt', 'r') as in_file:
    #     count = 0
    #     for line in in_file.readlines():
    #         count += 1
    #         if count % 1000 == 0:
    #             print count
    #         try:
    #             seg_list = unicode(line).strip().split()
    #             new_line = ''
    #             for seg in seg_list:
    #                 for char in seg:
    #                     new_line += (char + ' ')
    #
    #             with open('corpus/wiki_hant_single_zi_seg.txt', 'a') as out_file:
    #                 out_file.write(new_line.strip() + '\n')
    #         except Exception, e:
    #             print e

    with open('corpus/ancient_name_corpus.txt', 'r') as in_file:
        count = 0
        for line in in_file.readlines():
            count += 1
            if count % 1000 == 0:
                print count
            try:
                seg_list = unicode(line).strip().split()
                new_line = ''
                for seg in seg_list:
                    for char in seg:
                        new_line += (char+' ')

                with open('corpus/ancient_name_corpus_seg.txt', 'a') as out_file:
                    out_file.write(new_line.strip() + '\n')
            except Exception, e:
                print e

    with open('corpus/chinese_name_corpus.txt', 'r') as in_file:
        count = 0
        for line in in_file.readlines():
            count += 1
            if count % 1000 == 0:
                print count
            try:
                seg_list = unicode(line).strip().split()
                new_line = ''
                for seg in seg_list:
                    for char in seg:
                        new_line += (char + ' ')

                with open('corpus/chinese_name_corpus_seg.txt', 'a') as out_file:
                    out_file.write(new_line.strip() + '\n')
            except Exception, e:
                print e


# simple_word_segment()
# jieba_tight_word_segment()
single_zi_segment()



