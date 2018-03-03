# -*- coding: utf-8 -*-
import requests
import re
from lxml import etree
import time
import random
import json
import datetime
from bs4 import BeautifulSoup
import Queue
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def pos_case_prepare():

    # 整理出800条中医药相关术语用作训练正例
    # 可能有重复,故从各中医药相关词表打乱随机抽出120条*8
    # 医书,医家,疾病由于数据来源问题,抽出的样例还需人工校验筛选一遍


    # 基础概念
    base_term_list = []
    with open('semi_finished_data/term_list.txt', 'r') as in_file:
        for line in in_file.readlines():
            base_term_list.append(line.strip())

    # 医书
    yishu_list = []
    with open('semi_finished_data/yishu_list_json.txt', 'r') as in_file:
        yishu_list_tmp = json.load(in_file)
        for book in yishu_list_tmp:
            # 以50%的概率带书名号
            float_tmp = random.random()
            if float_tmp > 0.5:
                book = book.replace('《', '').replace('》', '')
            yishu_list.append(book)

    # 医家
    yijia_list = []
    with open('semi_finished_data/yijia_list_json.txt', 'r') as in_file:
        yijia_list = json.load(in_file)

    # 方剂
    fangji_list = []
    with open('semi_finished_data/方剂.txt', 'r') as in_file:
        for line in in_file.readlines():
            fangji_list.append(line.strip())

    # 药材
    yaocai_list = []
    with open('semi_finished_data/yaocai_spider_log.txt', 'r') as in_file:
        for line in in_file.readlines():
            if 'succ' in line:
                yaocai_list.append(line.split('succ:')[1].strip())

    # 治法
    zhifa_list = []
    with open('semi_finished_data/zhifa_json.txt', 'r') as in_file:
        zhifa_dict_list = json.load(in_file)
        for item in zhifa_dict_list:
            zhifa_list.append(item['name'])

    # 疾病
    # 疾病名当初为了爬虫召回率做了窗口拆词,
    # 这里不太好用,还需人工筛选
    jibing_list = []
    with open('semi_finished_data/succ_jibing_page.json', 'r') as in_file:
        jibing_dict = json.load(in_file)
        for k in jibing_dict:
            if len(k) > 0:
                jibing_list.append(k)

    # 证候期度型
    zhenghou_list = []
    with open('semi_finished_data/zhenghou.txt', 'r') as in_file:
        for line in in_file.readlines():
            # 以50%的概率带证字和不带证字
            if random.random() > 0.5:
                line = line.replace('证', '')
            zhenghou_list.append(line.strip())


    # 将这八个列表打乱随机各抽出120个
    the_eight_list = [base_term_list, yishu_list, yijia_list, fangji_list, yaocai_list, zhifa_list, jibing_list,
                      zhenghou_list]

    list_num = 1
    for a_term_list in the_eight_list:
        random.shuffle(a_term_list)
        with open('semi_finished_data/pos_case_list_%s.txt' % str(list_num), 'w') as out_file:
            for i in range(len(a_term_list)):
                out_file.write(str(a_term_list[i])+'\n')
        list_num += 1



def open_domain_neg_spider():
    """
    从百度百科按照随机id抓取词条,一千条左右,
    并人工筛选出800条与中医药领域无关的作为反例数据
    https://baike.baidu.com/view/{id}
    :return:
    """

    view_id_set = set()
    for i in range(100000):
        fetch_id = (int(random.random()*111101998))%15000000
        view_id_set.add(fetch_id)
    view_id_list = list(view_id_set)

    open_domain_word_set = set()
    for id in view_id_list:
        item_url = 'https://baike.baidu.com/view/'+str(id)
        # 百度百科的链接重定向,直接get会报错,用session家agent
        my_session = requests.session()
        my_session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
        res = my_session.get(item_url)
        mysoup = BeautifulSoup(res.content, 'lxml')
        open_domain_word = str(mysoup.title.text).replace('_百度百科', '')
        open_domain_word_set.add(open_domain_word)
        if len(open_domain_word_set) % 10==0:
            print len(open_domain_word_set)
        if len(open_domain_word_set)>20000:
            break
    with open('semi_finished_data/open_domain_neg_large.txt', 'a') as out_file:
        for item in list(open_domain_word_set):
            out_file.write(item+'\n')


def modern_med_neg_prepare():
    """
    从百度医疗分类词库中随机抽了一部分词条,现代医学为主
    现从html里剥离出词条,再经人工过滤,去除与中医药相关的,
    作为训练负样本.
    :return:
    """
    term_pattern = re.compile('<div class="waterFall_content_title">(.*?)</div>')
    with open('semi_finished_data/modern_med_page_element.txt') as in_file:
        file_content = in_file.read()
        term_list = re.findall(term_pattern, file_content)
    with open('semi_finished_data/modern_med_neg.txt', 'a') as out_file:
        for item in term_list:
            out_file.write(item+'\n')
            print item



pos_case_prepare()
# open_domain_neg_spider()
# modern_med_neg_prepare()








