# -*- coding: utf-8 -*-
import sys
import os

reload(sys)
sys.setdefaultencoding('utf8')



class ResultExtractor():
    """
    实验结果抽取类, 按需求从运行结果文件中抽取相应数值项
    将相同含义的多组实验数值整理到一起,也方便载excel里观察
    """
    
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
        for i in range(len(line_list)/6):
            report = {}
            neg_list = line_list[i*6+2].split()
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
    def get_neg_p_from_file(cls, file_path):
        """
        负样本准确率
        :param file_path: 
        :return: 
        """
        return cls.get_score_from_file(file_path, 'neg_p')

    
    @classmethod
    def get_pos_p_from_file(cls, file_path):
        """
        正样本准确率
        :param file_path: 
        :return: 
        """
        return cls.get_score_from_file(file_path, 'pos_p')
        

    @classmethod
    def get_pos_c_from_file(cls, file_path):
        """
        正样本召回率
        :return: 
        """
        return cls.get_score_from_file(file_path, 'pos_c')

    @classmethod
    def get_avg_f_from_file(cls, file_path):
        """
        总f1_score
        :param file_path:
        :return:
        """
        return cls.get_score_from_file(file_path, 'avg_f')

    @classmethod
    def get_all_train_score_from_dir(cls, dir_path):
        """
        获取目录下所有训练集结果中的neg_p, pos_p, pos_c, avg_f
        :param dir_path: 
        :return: 
        """
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            if 'train' in file_name:
                with open(dir_path+'report_select/train/neg_p.txt', 'a') as out_file:
                    neg_p_list = cls.get_neg_p_from_file(dir_path+file_name)
                    out_str = file_name+'\t'+'\t'.join([str(item) for item in neg_p_list])+'\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train/pos_p.txt', 'a') as out_file:
                    pos_p_list = cls.get_pos_p_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_p_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train/pos_c.txt', 'a') as out_file:
                    pos_c_list = cls.get_pos_c_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_c_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train/avg_f.txt', 'a') as out_file:
                    avg_f_list = cls.get_avg_f_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in avg_f_list]) + '\n'
                    out_file.write(out_str)


    @classmethod
    def get_all_test_score_from_dir(cls, dir_path):
        """
        获取目录下所有测试集结果中的neg_p, pos_p, pos_c, avg_f
        :param dir_path: 
        :return: 
        """
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            if 'test' in file_name:
                with open(dir_path+'report_select/test/neg_p.txt', 'a') as out_file:
                    neg_p_list = cls.get_neg_p_from_file(dir_path+file_name)
                    out_str = file_name+'\t'+'\t'.join([str(item) for item in neg_p_list])+'\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/test/pos_p.txt', 'a') as out_file:
                    pos_p_list = cls.get_pos_p_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_p_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/test/pos_c.txt', 'a') as out_file:
                    pos_c_list = cls.get_pos_c_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_c_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/test/avg_f.txt', 'a') as out_file:
                    avg_f_list = cls.get_avg_f_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in avg_f_list]) + '\n'
                    out_file.write(out_str)

    @classmethod
    def get_train_test_score_together_from_dir(cls, dir_path):
        """
        将训练集预测结果与测试集预测结果放一起方便比较
        :param dir_path: 
        :return: 
        """
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            try:
                with open(dir_path+'report_select/train_test/neg_p.txt', 'a') as out_file:
                    neg_p_list = cls.get_neg_p_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in neg_p_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train_test/pos_p.txt', 'a') as out_file:
                    pos_p_list = cls.get_pos_p_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_p_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train_test/pos_c.txt', 'a') as out_file:
                    pos_c_list = cls.get_pos_c_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in pos_c_list]) + '\n'
                    out_file.write(out_str)
                with open(dir_path+'report_select/train_test/avg_f.txt', 'a') as out_file:
                    avg_f_list = cls.get_avg_f_from_file(dir_path + file_name)
                    out_str = file_name + '\t' + '\t'.join([str(item) for item in avg_f_list]) + '\n'
                    out_file.write(out_str)
            except Exception, e:
                print e
        


# ResultExtractor.get_all_train_score_from_dir('../svm_tcm_classify/classify_result_7000/')
# ResultExtractor.get_all_test_score_from_dir('../svm_tcm_classify/classify_result_7000/')
# ResultExtractor.get_train_test_score_together_from_dir('../svm_tcm_classify/classify_result_7000/')
#
#
# ResultExtractor.get_all_train_score_from_dir('../rnn_tcm_classify/classify_result_7000/')
# ResultExtractor.get_all_test_score_from_dir('../rnn_tcm_classify/classify_result_7000/')
# ResultExtractor.get_train_test_score_together_from_dir('../rnn_tcm_classify/classify_result_7000/')

# ResultExtractor.get_all_train_score_from_dir('../lstm_tcm_classify/classify_result_7000/')
# ResultExtractor.get_all_test_score_from_dir('../lstm_tcm_classify/classify_result_7000/')
# ResultExtractor.get_train_test_score_together_from_dir('../lstm_tcm_classify/classify_result_7000/')

# ResultExtractor.get_all_train_score_from_dir('../dt_tcm_classify/classify_result_7000/')
# ResultExtractor.get_all_test_score_from_dir('../dt_tcm_classify/classify_result_7000/')
# ResultExtractor.get_train_test_score_together_from_dir('../dt_tcm_classify/classify_result_7000/')


# ResultExtractor.get_all_train_score_from_dir('../name_recognition/result/')
# ResultExtractor.get_all_test_score_from_dir('../name_recognition/result/')
# ResultExtractor.get_train_test_score_together_from_dir('../name_recognition/result/')





