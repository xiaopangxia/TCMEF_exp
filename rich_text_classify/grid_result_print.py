# -*- coding: utf-8 -*-
import sys
import os

class GridReault():
    @classmethod
    def load_reports_from_file(cls, file_path):
        """
        从结果文件,按照六行为一个report,加载所有reports
        :param file_path: 实验结果文件
        :return: 每个report为一个字典,返回report的列表[{},{},{}...]
        """
        line_list = []
        with open(file_path, 'r') as in_file:
            for line in in_file.readlines():
                line_list.append(line)

        report_list = []
        for i in range(len(line_list) / 6):
            report = {}
            neg_list = line_list[i * 6 + 2].split()
            report['neg_p'] = float(neg_list[1])
            report['neg_c'] = float(neg_list[2])
            report['neg_f'] = float(neg_list[3])

            pos_list = line_list[i * 6 + 3].split()
            report['pos_p'] = float(pos_list[1])
            report['pos_c'] = float(pos_list[2])
            report['pos_f'] = float(pos_list[3])

            avg_list = line_list[i * 6 + 5].split()
            report['avg_p'] = float(avg_list[3])
            report['avg_c'] = float(avg_list[4])
            report['avg_f'] = float(avg_list[5])

            report_list.append(report)

        return report_list

    @classmethod
    def get_score_from_file(cls, file_path, score_name):
        """
        从结果文件获取多轮实验某项结果指标的list
        :param file_path:
        :return:
        """
        report_list = cls.load_reports_from_file(file_path)
        score_list = []
        for res in report_list:
            score_list.append(res[score_name])
        return score_list

    @classmethod
    def get_avg_f_from_file(cls, file_path):
        """
        总f1_score
        :param file_path:
        :return:
        """
        return cls.get_score_from_file(file_path, 'avg_f')

    @classmethod
    def grid_result_print(cls):
        file_list = os.listdir('./lda_fulltext_classify_result')
        for result_file in file_list:
            if 'linear' in result_file and 'True' in result_file and 'test' in result_file:
                n_feature = result_file.split('_')[2]
                n_topic = result_file.split('_')[3]
                f_score = cls.get_avg_f_from_file('./lda_fulltext_classify_result/'+result_file)
                print n_feature, n_topic, f_score[0]




GridReault.grid_result_print()
