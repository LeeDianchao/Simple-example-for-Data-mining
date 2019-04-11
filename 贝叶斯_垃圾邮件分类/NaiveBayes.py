# coding:UTF-8
import re
import os
import numpy as np


def text_format(filepath):
    # 根据用户输入地址，读取文件
    with open(filepath, "r") as file:
        text = file.readlines()
    for i in range(len(text)):
        text[i] = text[i].strip()
    result_text = []
    splitter = re.compile(r"\W+|\d+")  # 正则匹配，去除文本中的符号、数字等元素
    for i in range(len(text)):
        text[i] = splitter.split(text[i].lower())  # 都变为小写
        # 每条文本已经被分为一段一段的句子，每条文本此时是一个list，先去除其中字段长度小于等于0的单词
        text[i] = [word for word in text[i] if len(word) > 0]
        if len(text[i]) > 0:     # 转为一维
            result_text += text[i]
    return result_text


def read_text(dirpath):
    n = 0   # 样本个数
    sample_data = [] # 所有的样本数据
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            filepath = dirpath+"\\"+file
            # print filepath
            text = text_format(filepath)
            sample_data += text
            n += 1
    # 单词表
    kinds = list(set(sample_data))   # 样本中单词种类个数
    return sample_data, n, kinds


def classify_count(data):
    Classified_count = {}  # 类别及每一类的数量
    index = 0
    while index < len(data):
        num = data[index]
        if Classified_count.get(str(num)) is None:
            Classified_count.update({str(num): 1})
        else:
            Classified_count[str(num)] += 1
        index += 1
    return Classified_count


def trainmodel(hampath, spampath):
    ham_data, ham_num, ham_kinds = read_text(hampath)
    # 非垃圾邮件中单词的种类以及出现的次数
    ham_class = classify_count(ham_data)

    spam_data, spam_num, spam_kinds = read_text(spampath)
    # 垃圾邮件中单词的种类以及出现的次数
    spam_class = classify_count(spam_data)

    all_kinds = ham_kinds + spam_kinds
    all_kinds = list(set(all_kinds))  # 样本单词表（样本中单词种类）
    # 先验概率（Prior probability）
    h_Prior = float(ham_num) / (float(ham_num + spam_num)) # 先验概率
    s_Prior = 1 - h_Prior    # 先验概率
    all_class_num = len(all_kinds)  # 样本中单词种类的数目

    # 后验概率（Posterior probability）
    h_Posterior = {}  # 非垃圾邮件中每种单词的后验概率
    ham_data_num = len(ham_data)  # 非垃圾邮件集大小
    s_Posterior = {}  # 垃圾邮件中每种单词的后验概率
    spam_data_num = len(spam_data)  # 垃圾邮件集大小
    for i in range(0, len(all_kinds)):
        s = all_kinds[i]
        if ham_class.get(s) is None:  # 加一平滑
            p = float(1) / (float(ham_data_num) + float(all_class_num))
            h_Posterior.update({s: p})
        else:
            p = float(ham_class.get(s) + 1) / (float(ham_data_num + all_class_num))
            h_Posterior.update({s: p})
        if spam_class.get(s) is None:  # 加一平滑
            p = float(1) / (float(spam_data_num) + float(all_class_num))
            s_Posterior.update({s: p})
        else:
            p = float(spam_class.get(s) + 1) / (float(spam_data_num + all_class_num))
            s_Posterior.update({s: p})
    print h_Posterior
    print len(h_Posterior)
    print s_Posterior
    print len(s_Posterior)
    return h_Prior, h_Posterior, s_Prior, s_Posterior, ham_data_num, spam_data_num


def text(path, h_Prior, h_Posterior, s_Prior, s_Posterior, ham_data_num, spam_data_num):
    text = text_format(path)
    texth = h_Prior
    texts = s_Prior
    all_class_num = len(h_Posterior)
    for i in range(0, len(text)):
        if h_Posterior.get(text[i]) is not None:
            texth = texth * h_Posterior.get(text[i])
        else :
            texth = texth * (float(1) / (float(ham_data_num) + float(all_class_num + 1)))  # 有问题？？？？
        if s_Posterior.get(text[i]) is not None:
            texts = texts * s_Posterior.get(text[i])
        else :
            texts = texts * (float(1) / (float(spam_data_num) + float(all_class_num + 1)))
    result = (texth > texts)  # 预测结果
    return texth, texts, result


if __name__ == '__main__':
    hampath = ".\\email\\ham"
    spampath = ".\\email\\spam"
    h_Prior, h_Posterior, s_Prior, s_Posterior, ham_data_num, spam_data_num = trainmodel(hampath, spampath)

    t = text_format(".\\email\\ham\\1.txt")
    '''
    th, ts, result = text(".\\email\\spam\\25.txt", hc, hh, sc, ss)
    print "非垃圾邮件预测概率：" + str(th)
    print "垃圾邮件预测概率：" + str(ts)
    print "预测结果:" + str(result)
    '''

    n = 0  # 样本个数
    dirpath = ".\\email\\test"
    sample_data = []  # 所有的样本数据
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            filepath = dirpath + "\\" + file
            # print filepath
            th, ts, result = text(filepath, h_Prior, h_Posterior, s_Prior, s_Posterior, ham_data_num, spam_data_num)
            n += 1
            print str(filepath)+":"+str(result)