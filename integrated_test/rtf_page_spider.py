# -*- coding: utf-8 -*-
import sys
import os
import requests
import re
from bs4 import BeautifulSoup
import time
import random
import json

reload(sys)
sys.setdefaultencoding('utf8')


"""
从百度百科抓取正例词条与负例词条的页面,
解析出词条摘要文本存为文本,
目标词条都是由pnf过滤出的可能为人名的词条
"""

def person_name_raw_page_spide():
    """
    目标词条都是由pnf过滤出的可能为人名的词条
    存储到rtf_raw_page目录下
    :return: 
    """
    neg_case_list = []
    with open('./pnf_result/neg_name.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    pos_case_list = []
    with open('./pnf_result/pos_name.txt', 'r') as in_file:
        for line in in_file.readlines():
            pos_case_list.append(line.strip())

    my_session = requests.session()
    my_session.headers[
        'User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'

    for item in pos_case_list:
        try:
            res = my_session.get('https://baike.baidu.com/item/' + item)
            with open('./rtf_raw_page/pos_case/%s.html' % item, 'wb') as out_file:
                out_file.write(res.content)
                time.sleep(random.random())
            print item
        except Exception, e:
            print e
            with open('./log_file/raw_page_from_baidu_failed.txt', 'a') as out_file:
                out_file.write(item + '\n')

    for item in neg_case_list:
        try:
            res = my_session.get('https://baike.baidu.com/item/' + item)
            with open('./rtf_raw_page/neg_case/%s.html' % item, 'wb') as out_file:
                out_file.write(res.content)
                time.sleep(random.random())
            print item
        except Exception, e:
            print e
            with open('./log_file/raw_page_from_baidu_failed.txt', 'a') as out_file:
                out_file.write(item + '\n')
    



def extract_fulltext_from_raw_page():
    """
    从百科词条页面抽取全文本
    :return: 
    """
    # 来自百度的负例
    page_list = os.listdir('./rtf_raw_page/neg_case/')
    for page_name in page_list:
        try:
            with open('./rtf_raw_page/neg_case/' + page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                fulltext_soup = my_soup.find_all('div', attrs={'class': 'main-content'})

                fulltext_text = ''
                for text in fulltext_soup[0].stripped_strings:
                    fulltext_text += text

                # 保存非空的词条摘要
                if len(fulltext_text) > 10:
                    with open('./rtf_data_fulltext/neg_case/%s.txt' % page_name, 'w') as out_file:
                        out_file.write(fulltext_text)

                print 'succ:' + page_name
        except Exception, e:
            print e
            print 'Failed:' + page_name

    # 来自互动的正例
    page_list = os.listdir('./rtf_raw_page/pos_case/')
    for page_name in page_list:
        try:
            with open('./rtf_raw_page/pos_case/' + page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                fulltext_soup = my_soup.find_all('div', attrs={'class': 'main-content'})

                fulltext_text = ''
                for text in fulltext_soup[0].stripped_strings:
                    fulltext_text += text

                # 保存非空的词条摘要
                if len(fulltext_text) > 10:
                    with open('./rtf_data_fulltext/pos_case/%s.txt' % page_name, 'w') as out_file:
                        out_file.write(fulltext_text)

                print 'succ:' + page_name
        except Exception, e:
            print e
            print 'Failed:' + page_name

# person_name_raw_page_spide()
# extract_fulltext_from_raw_page()




