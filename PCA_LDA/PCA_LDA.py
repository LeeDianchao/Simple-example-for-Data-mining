# coding:UTF-8
import scipy.io as scio
from numpy import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA as SKPCA


def LoadDataMat(FilePath):
    dataFile = FilePath #'C:\Users\\12294\Desktop\BU3D_feature.mat'
    loaddata = scio.loadmat(dataFile)
    data = loaddata['data']
    x = data[:, :414]   #数据
    y = data[:, 414]    #标签
    return x, y, data

def PCA(dataMat, Rate=1.0):
    meanVals = mean(dataMat, axis=0) #求均值
    #print meanVals.shape
    #print meanVals
    DataAdjust = dataMat - meanVals #减去平均值
    covMat = cov(DataAdjust, rowvar=0) #求协方差
    eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
    print "特征值"
    print eigVals
    print "特征向量"
    print eigVects
    # 从大到小的索引
    eigValIndex = argsort(eigVals, )[::-1]
    # 所有的特征值、特征向量根据特征值从从大到小排序
    eigVals = eigVals[eigValIndex]
    eigVects = eigVects[:, eigValIndex]
    pca_data = matmul(dataMat, eigVects)
    n = 1
    while n < eigValIndex.size:
        contribution = sum(eigVals[:n]) / sum(eigVals)  # 贡献率
        print "选取的特征值及特征向量的下标 " + str(eigValIndex[n - 1])
        print eigVals[n - 1]  # 特征值
        print eigVects[n - 1]  # 特征向量
        if contribution <= Rate:
            n += 1
        elif contribution > Rate:
            break
    redEigVects = eigVects[:n, :]  # 对应的特征向量
    print redEigVects
    lowDDataMat = DataAdjust * redEigVects.T  # 得到低维度的矩阵，将数据转换到低维新空间
    resultMat = (lowDDataMat * redEigVects) + meanVals  # 重构数据，得到降维后的矩阵
    pca_release = pca_data[:, :n]
    print "=================================="
    print "Enter ContributionRate = " + str(Rate)
    print "PCA num:" + str(n)
    print "Contribution:" + str(contribution)
    return pca_data, pca_release, n, lowDDataMat, resultMat

def class_mean(dataSet):  # 计算每类的均值
    means = mean(dataSet, axis=0)
    mean_vectors = mat(means)
    return mean_vectors

def within_class_S(dataSet):  # 计算类内散度
    m = shape(dataSet[1])[1]
    class_S = mat(zeros((m, m)))
    mean = class_mean(dataSet)
    for line in dataSet:
        x = line - mean
        class_S += x.T * x
    return class_S

def sort_by_col(a, col_index): #按列对行进行排序
    a1 = a.T
    col_max = a.shape[-1] - 1
    if col_index < col_max:
        # 两行互换
        a1[col_index] = a1[col_index] + a1[col_max]
        a1[col_max] = a1[col_index] - a1[col_max]
        a1[col_index] = a1[col_index] - a1[col_max]
        a2 = lexsort(a1)
        # 因为a1的行交换影响了a，得到序列结果后再换回来，保持行不变
        a1[col_index] = a1[col_index] + a1[col_max]
        a1[col_max] = a1[col_index] - a1[col_max]
        a1[col_index] = a1[col_index] - a1[col_max]
    else:
        a2 = lexsort(a1)
    return a[a2]

def LDA(data, classes, Rate=1.0):
    AllU = mean(data, axis=0)  # 整体的均值
    y = classes
    class_type = unique(y)
    Sw = []  # 类内散度
    Sb = []  # 类间散度
    Ui = []  # 每个类的均值
    for c in class_type:
        class1 = data[y == c, :]
        class_mean = mean(class1, axis=0)
        Ui.append(class_mean)
        # 类内散度
        Sw.append((class1 - class_mean).T.dot(class1 - class_mean))
        class_num = class1.shape[0]#样本数量
        # 类间散度
        Sb.append(class_num * (class_mean - AllU).dot((class_mean - AllU).T))

    SW = sum(Sw, axis=0)# 类内散度
    SB = sum(Sb, axis=0)# 类间散度
    SW_SB = linalg.inv(SW).dot(SB)
    eigVals, eigVects = linalg.eig(SW_SB)  # 计算特征值和特征向量
    # 从大到小的索引
    eigValIndex = argsort(eigVals, )[::-1]
    # 所有的特征值、特征向量根据特征值从从大到小排序
    eigVals = eigVals[eigValIndex]
    eigVects = eigVects[:, eigValIndex]
    lda_data = matmul(data, eigVects)
    n = 1
    while n < eigValIndex.size:
        contribution = sum(eigVals[:n]) / sum(eigVals)  # 贡献率
        if contribution <= Rate:
            n += 1
        elif contribution > Rate:
            break
    lda_release = lda_data[:, :n]
    print "=================================="
    print "Enter ContributionRate = " + str(Rate)
    print "n:" + str(n)
    print "Contribution:" + str(contribution)
    return lda_data, lda_release, n

def plot_embedding(X, y, title=None):
    x_min, x_max = X.min(0), X.max(0) #移动到第一象限
    X = (X - x_min) / (x_max - x_min) #标准化
    classes = unique(y)
    styles = ["r.", 'g*', 'b+', 'k^', "mv", "yx"]
    print zip(classes, styles)
    for c, mark in zip(classes, styles):
        class1 = X[y == c]
        plt.plot(class1[:, 0], class1[:, 1], mark, label="Class "+ str(c))
    plt.legend()  # 设置图例
    if title is not None:
        plt.title(title)
    plt.show()


def myLDA(data, Rate=1.0):
    yclass = data[:, 414]
    data = data[:, std(data, axis=0) != 0]  # 去除标准差为0的特征
    DATA = data[:, :-1]          # 原数据
    data = sort_by_col(data, data.shape[1])  # 排序（按标签列进行排序）
    data = data[:, :-1]  # 去除标签列
    AllU = mean(data, axis=0) # 所有数据的均值

    label_num = {}  # 统计类别及每一类的数量
    n = 0
    while n < yclass.shape[0]:
        label = int(yclass[n])
        if label_num.get(str(label)) is None:
            label_num.update({str(label): 1})
        else:
            label_num[str(label)] += 1
        n += 1
    class_label = label_num.keys()  # 所有类别标签
    class_label = sorted(class_label) # 对标签进行排序
    index = 0
    Sw = []  # 类内散度
    Sb = []  # 类间散度
    Ui = []  # 每个类的均值
    for i in range(0, len(class_label)):
        s = class_label[i]
        class_len = label_num.get(s)  # 得到该类数据的个数
        class_one = data[index:index+class_len, :] # 截取该类数据
        index = index + class_len
        class_mean = mean(class_one, axis=0)  # 该类数据的均值
        Ui.append(class_mean)
        # 类内散度
        Sw.append((class_one - class_mean).T.dot(class_one - class_mean))
        class1_num = class_one.shape[0]  # 样本数量
        # 类间散度
        Sb.append(class1_num * (class_mean - AllU).dot((class_mean - AllU).T))
    SW = sum(Sw, axis=0)  # 类内散度
    SB = sum(Sb, axis=0)  # 类间散度
    SW_SB = linalg.inv(SW).dot(SB)
    eigVals, eigVects = linalg.eig(SW_SB)  # 计算特征值和特征向量
    # 从大到小的索引
    eigValIndex = argsort(eigVals, )[::-1]
    # 所有的特征值、特征向量根据特征值从从大到小排序
    eigVals = eigVals[eigValIndex]
    eigVects = eigVects[:, eigValIndex]
    lda_data = matmul(DATA, eigVects)
    n = 1
    while n < eigValIndex.size:
        contribution = sum(eigVals[:n]) / sum(eigVals)  # 贡献率
        if contribution <= Rate:
            n += 1
        elif contribution > Rate:
            break
    lda_release = lda_data[:, :n]
    print "=================================="
    print "Enter ContributionRate = " + str(Rate)
    print "n:" + str(n)
    print "Contribution:" + str(contribution)
    return lda_data, lda_release, n


if __name__ == '__main__':
    Filepath = '.\\BU3D_feature.mat'
    x, yclass, data = LoadDataMat(Filepath)
    set_printoptions(suppress=True)
    x = x[:, std(x, axis=0) != 0]  # 去除标准差为0的特征
    value_num = {} #类别及每一类的数量
    n = 0
    classtype = unique(yclass) #类别
    while n < data.shape[0]:
        num = float(data[n, 414])
        if value_num.get(str(num)) == None:
            value_num.update({str(num): 1})
        else:
            value_num[str(num)] += 1
        n += 1
    print "类别";print value_num

    '''pca_data, pca_release, pcanum, lowDDataMat, resultMat = PCA(x, 0.85)
    print resultMat.shape
    plot_embedding(pca_data[:, :2], yclass, "PCA num: 2")

    tsne = TSNE(n_components=2, early_exaggeration=100, n_iter=1000)
    tsne_data = tsne.fit_transform(pca_release)
    plot_embedding(tsne_data, yclass, "PCA(0.85) and T-SNE")

    lda_data, lda_release, n = LDA(x, yclass, 0.85)
    tsne = TSNE(n_components=2, early_exaggeration=100, n_iter=1000)
    tsne_data = tsne.fit_transform(lda_release)
    plot_embedding(tsne_data, yclass, "LDA(0.85) and T-SNE")

    PCAlowDMat = SKPCA(n_components=2).fit_transform(x)
    plot_embedding(PCAlowDMat, yclass, "sklearn PCA")

    PCAlowDMat = LinearDiscriminantAnalysis(n_components=2).fit_transform(x, yclass)
    plot_embedding(PCAlowDMat, yclass, "sklearn LDA")'''
    lda_data, lda_release, n = myLDA(data, 0.85)
    tsne = TSNE(n_components=2, early_exaggeration=100, n_iter=1000)
    tsne_data = tsne.fit_transform(lda_release)
    plot_embedding(tsne_data, yclass, "myLDA(0.85) and T-SNE")









