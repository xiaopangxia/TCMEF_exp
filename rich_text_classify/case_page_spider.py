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
抓取正例词条与负例词条的页面,
解析出词条摘要文本存为文本,
解析出infobox内容,存为json
"""

def open_domain_raw_page_from_baidu():
    """
    先把开放域的负样例词条抓下来
    :return:
    """
    neg_case_list = []
    with open('./case_list/open_domain_neg_large.txt', 'r') as in_file:
        for line in in_file.readlines():
            neg_case_list.append(line.strip())

    my_session = requests.session()
    my_session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'

    for item in neg_case_list:
        try:
            res = my_session.get('https://baike.baidu.com/item/'+item)
            with open('./raw_page/neg_case/%s.html' % item, 'wb') as out_file:
                out_file.write(res.content)
                time.sleep(random.random())
            print item
        except Exception, e:
            print e
            with open('./log_file/open_domain_from_baidu_failed.txt', 'a') as out_file:
                out_file.write(item+'\n')




def extract_abstract_infobox_from_baidu_raw_neg_page():
    """
    从保存的百度百科负样例页面中抽取词条摘要文本和infobox信息
    :return:
    """

    page_list = os.listdir('./raw_page/neg_case/')
    for page_name in page_list:
        try:
            with open('./raw_page/neg_case/'+page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                info_box_left = my_soup.find_all('dl', attrs={'class': 'basicInfo-block basicInfo-left'})
                info_box_right = my_soup.find_all('dl', attrs={'class': 'basicInfo-block basicInfo-right'})
                info_box_list = info_box_left + info_box_right
                abstract_soup = my_soup.find_all('div', attrs={'class':'lemma-summary'})


                abstract_text = ''
                for text in abstract_soup[0].stripped_strings:
                    abstract_text += text

                # 保存非空的词条摘要
                if len(abstract_text) > 10:
                    with open('./page_abstract/neg_case/%s.txt' % page_name, 'w') as out_file:
                        out_file.write(abstract_text)

                info_box_dict = {}
                info_item_list = []
                for box in info_box_list:
                    for item in box.stripped_strings:
                        info_item_list.append(item)
                for i in range(len(info_item_list)/2):
                    info_box_dict[info_item_list[i*2]]=info_item_list[i*2+1]

                with open('./page_infobox/neg_case/%s.json' % page_name, 'w') as out_file:
                    out_file.write(json.dumps(info_box_dict))

                print 'succ:'+page_name
        except Exception, e:
            print e
            print 'Failed:'+page_name



def extract_abstract_infobox_from_hudong_raw_pos_page():
    """
    从互动百科正例页面中抽取词条摘要与infobox信息
    :return:
    """
    page_list = os.listdir('./raw_page/pos_case/')
    for page_name in page_list:
        try:
            with open('./raw_page/pos_case/'+page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                try:
                    abstract_soup = my_soup.find_all('div', attrs={'class': 'summary'})[0]
                    abstract_text = ''
                    for text in abstract_soup.p.stripped_strings:
                        abstract_text += text

                    # 保存非空的词条摘要
                    if len(abstract_text) > 10:
                        with open('./page_abstract/pos_case/%s.txt' % page_name, 'w') as out_file:
                            out_file.write(abstract_text)
                except Exception, e:
                    print 'no_abstract:'+page_name

                infobox_name_list = my_soup.find_all('div', attrs={'id': 'datamodule'})[0].find_all('strong')
                infobox_value_list = my_soup.find_all('div', attrs={'id': 'datamodule'})[0].find_all('span')

                info_box_dict = {}
                for i in range(len(infobox_name_list)):
                    info_name = infobox_name_list[i].text
                    info_value = infobox_value_list[i].text
                    info_box_dict[info_name.replace('：', '')] = info_value

                with open('./page_infobox/pos_case/%s.json' % page_name, 'w') as out_file:
                    out_file.write(json.dumps(info_box_dict))

                print 'succ:' + page_name

        except Exception, e:
            print 'no_infobox:' + page_name


def extract_fulltext_from_raw_page():
    """
    从百科词条页面抽取全文本
    :return: 
    """
    # 来自百度的负例
    page_list = os.listdir('./raw_page/neg_case/')
    for page_name in page_list:
        try:
            with open('./raw_page/neg_case/'+page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                fulltext_soup = my_soup.find_all('div', attrs={'class': 'main-content'})

                fulltext_text = ''
                for text in fulltext_soup[0].stripped_strings:
                    fulltext_text += text

                # 保存非空的词条摘要
                if len(fulltext_text) > 10:
                    with open('./page_fulltext/neg_case/%s.txt' % page_name, 'w') as out_file:
                        out_file.write(fulltext_text)

                print 'succ:' + page_name
        except Exception, e:
            print e
            print 'Failed:' + page_name
    
    # 来自互动的正例
    page_list = os.listdir('./raw_page/pos_case/')
    for page_name in page_list:
        try:
            with open('./raw_page/pos_case/' + page_name, 'r') as in_file:
                page_text = in_file.read()
                my_soup = BeautifulSoup(page_text, 'lxml')
                fulltext_soup = my_soup.find_all('div', attrs={'class': 'l w-640'})

                fulltext_text = ''
                for text in fulltext_soup[0].stripped_strings:
                    fulltext_text += text

                # 保存非空的词条摘要
                if len(fulltext_text) > 10:
                    with open('./page_fulltext/pos_case/%s.txt' % page_name, 'w') as out_file:
                        out_file.write(fulltext_text)

                print 'succ:' + page_name
        except Exception, e:
            print e
            print 'Failed:' + page_name
    
    


# open_domain_raw_page_from_baidu()
# extract_abstract_infobox_from_baidu_raw_neg_page()
# extract_abstract_infobox_from_hudong_raw_pos_page()

extract_fulltext_from_raw_page()





