# -*- coding: utf-8 -*-

import sys

reload(sys)
sys.setdefaultencoding('utf8')

# 统计词条长度分布

length_count = [0]*80

with open('./manual_filtered_data/open_domain_neg_large.txt', 'r') as in_file:
    for line in in_file.readlines():
        length = len(unicode(line.strip()))
        if length<80:
            length_count[length] += 1



with open('./manual_filtered_data/pos_data_7000/pos_case_list_with_name.txt', 'r') as in_file:
    for line in in_file.readlines():
        length = len(unicode(line.strip()))
        if length<80:
            length_count[length] += 1

for cnt in length_count:
    print cnt


