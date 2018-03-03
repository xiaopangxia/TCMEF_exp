# -*- coding: utf-8 -*-
import sys
from langconv import *
reload(sys)
sys.setdefaultencoding('utf8')

# 在训练词向量过程中简体和繁体都要用到
# 需要将繁体wiki预料转化为简体中文

# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line

def wiki_cht2chs():
    with open('corpus/wiki_hant.txt', 'r') as in_file:
        for line in in_file.readlines():
            try:
                new_line = cht_to_chs(unicode(line)).strip()
                with open('corpus/wiki_hans.txt', 'a') as out_file:
                    out_file.write(new_line+'\n')
            except Exception, e:
                print e
                print line


def wiki_seg_chs2cht():
    with open('corpus/wiki_hans_seg.txt', 'r') as in_file:
        for line in in_file.readlines():
            try:
                new_line = chs_to_cht(unicode(line)).strip()
                with open('corpus/wiki_seg_hant.txt', 'a') as out_file:
                    out_file.write(new_line+'\n')
            except Exception, e:
                print e
                print line

# wiki_cht2chs()
wiki_seg_chs2cht()



