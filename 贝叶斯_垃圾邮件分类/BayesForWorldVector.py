#-*-coding:utf-8-*-
import re
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def load_data(folder_path):
    datalist = datasets.load_files(folder_path)
    # datalist是一个Bunch类
    return datalist


# 构造字典
def create_dictionary(ori_data):
    word_dic = set([])
    # 词典的构造
    for doc in ori_data:
        doc = str(doc)
        doc = re.sub(r"\W+|\d+", " ", doc)  # 用正则表达式将特殊符号、数字去除
        # 按空格将文本分隔开，然后转化为小写字母
        word_dic = word_dic | set(doc.lower().split())  # 去重
    return list(word_dic)


# 文本向量化
def create_vector(wordDic, ori_data):
    # 创建一个文档数（行）* 词向量（列）长度的二维数组
    doc_vector = np.zeros((len(ori_data), len(wordDic)), dtype = np.int)
    # 计数器
    count = 0
    for doc in ori_data:
        doc = str(doc)
        doc = re.sub(r"\W+|\d+", " ", doc)  # 用正则表达式将特殊符号、数字去除
        for word in doc.lower().split():
            if word in wordDic:
                doc_vector[count][wordDic.index(word)] += 1  # 将对应词向量位置加 1
        count = count + 1
    return doc_vector


# 先验概率
def pre_probabilty(ham_num, spam_num):
    s_pre_pro = []
    # 正常邮件的先验概率
    P_normal = float(ham_num)/float(ham_num + spam_num)
    s_pre_pro.append(P_normal)
    # 垃圾邮件的先验概率
    P_spam = 1 - P_normal
    s_pre_pro.append(P_spam)
    return s_pre_pro


# 计算每个词在正常邮件垃圾邮件中的数目 以及两类文档集的大小
def wordNum(text_vector, wordDic, class_label):
    num_word = np.zeros((2, len(wordDic)), dtype = np.int)
    ham_text_size = 0   # 正常邮件文档集的大小
    spam_text_size = 0  # 垃圾邮件文档集的大小
    for j in range(len(class_label)):
        # 在正常邮件的数目
        if class_label[j] == 1:
            ham_text_size += sum(text_vector[j])   # 统计正常邮件文本集大小
            for i in range(len(wordDic)):
                num_word[0][i] += text_vector[j][i]
        # 在垃圾邮件中的数目
        else:
            spam_text_size += sum(text_vector[j])  # 统计垃圾邮件文本集大小
            for i in range(len(wordDic)):
                num_word[1][i] += text_vector[j][i]
    return num_word, ham_text_size, spam_text_size


# 后验概率
def con_probabilty(text_vector, wordDic, class_label):
    # 得到每个词汇在正常邮件、垃圾邮件中的数目 以及两类文档集的大小
    word_num, ham_text_size, spam_text_size = wordNum(text_vector, wordDic, class_label)
    word_pro = np.zeros((2, len(wordDic)), dtype = np.double)
    class_num = len(wordDic)  # 单词集种类
    for i in range(len(wordDic)):  # 加一平滑
        word_pro[0][i] = float(word_num[0][i] + 1)/float(ham_text_size + class_num)
        word_pro[1][i] = float(word_num[1][i] + 1)/float(spam_text_size + class_num)
    return word_pro


# 得到每个类别中的文档数
def class_num(class_label):
    ham_num, spam_num = 0, 0   # 两类邮件的数量
    for i in range(len(class_label)):
        if class_label[i] == 1:   # 正常邮件
            ham_num += 1
        else:
            spam_num += 1         # 垃圾邮件
    return ham_num, spam_num


# 训练
def trainmodel(WordVector, WordDictionary, class_label, ham_num, spam_num):
    # 训练文本向量  词典  训练文本的标签  正常邮件数量  垃圾邮件数量
    # 计算先验概率
    prePro = pre_probabilty(ham_num, spam_num)
    # 计算后验概率
    conPro = con_probabilty(WordVector, WordDictionary, class_label)
    print("preProbablity:", prePro)
    print("conProbablity:", conPro)
    return prePro, conPro


# 测试
def test(test_vector, pre_pro, con_pro):
    text_pro = np.zeros((len(test_vector), 2), dtype = np.double)
    text_judge = []
    ham_num = 0   # 正常邮件数量
    spam_num = 0  # 垃圾邮件数量
    for i in range(len(test_vector)):
        text_pro[i][0] = pre_pro[0]  # 先验概率
        text_pro[i][1] = pre_pro[1]
        for j in range(len(test_vector[0])):
            for k in range(0, test_vector[i][j]):  # 按出现的单词，将对应后验概率相乘
                text_pro[i][0] *= con_pro[0][j]  # 正常邮件
                text_pro[i][1] *= con_pro[1][j]  # 垃圾邮件
        # 正常邮件
        if text_pro[i][0] > text_pro[i][1]:
            text_judge.append(1)
            ham_num += 1
        else:  # 垃圾邮件
            text_judge.append(0)
            spam_num += 1
    print "text_judge"
    print text_judge
    return ham_num, spam_num, text_pro


if __name__ == "__main__":
    # 数据集的路径
    file_path = ".\\train"
    data_list = load_data(file_path)
    # 分割数据集（分为训练集和测试集，同时分割属性标签）
    data_train, data_test, label_train, label_test = train_test_split(data_list.data, data_list.target, test_size=0.2)
    # 正常邮件的数目 # 垃圾邮件的数目
    ham_train_num, spam_train_num = class_num(label_train)
    # 建立词汇表
    WordDictionary = create_dictionary(data_train)
    # 将训练数据进行向量表示
    train_vector = create_vector(WordDictionary, data_train)
    # 训练模型，得到先验概率和后验概率
    prePro, conPro = trainmodel(train_vector, WordDictionary, label_train, ham_train_num, spam_train_num)

    # 测试数据的向量表示
    test_vector = create_vector(WordDictionary, data_test)
    # 正常邮件的数目   # 垃圾邮件的数目
    ham_test, spam_test = class_num(label_test)
    print "test label"
    print list(label_test)
    # 测试数据的准确率
    ham_test_num, spam_test_num, email_pro = test(test_vector, prePro, conPro)
    print email_pro
    print "test accuracy"
    tpr = float(ham_test_num) / float(ham_test)
    tnr = float(spam_test_num) / float(spam_test)
    print "True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）:" + str(tpr)
    print "True Negative Rate（真负率 , TNR）或特指度（specificity）:" + str(tnr)

    X = np.array(train_vector)
    Y = np.array(label_train)


    print "\n\n --sklearn.naive_bayes--"
    # sk_nb = GaussianNB()
    sk_nb = MultinomialNB()
    # sk_nb = BernoulliNB()
    # 拟合数据
    sk_nb.fit(X, Y)
    print "==Predict result by predict=="
    print(sk_nb.predict(test_vector))
    print "==Predict result by predict_proba=="
    print(sk_nb.predict_proba(test_vector))
    print "==Predict result by predict_log_proba=="
    print(sk_nb.predict_log_proba(test_vector))



