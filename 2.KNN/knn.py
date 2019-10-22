from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 这几行代码是用来导入数据集的。在这里我们导入的数据集叫做iris数据集，也是开源数据中最为重要的数据集之一。
# 这个数据包含了3个类别，所以适合的问题是分类问题。另外，具体数据集的描述可以参考：
# https://archive.ics.uci.edu/ml/datasets/Iris/ 从print(x,y)结果可以看到X拥有四个特征，并且标签y拥有0，1，2三种不同的值。

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X, y)

# 在这里X存储的是数据的特征，y存储的每一个样本的标签或者分类。我们使用 train_test_split来把数据分成了训练集和测试集。
# 主要的目的是为了在训练过程中也可以验证模型的效果。如果没有办法验证，则无法知道模型训练的好坏。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)

# 这部分是KNN算法的主要模块。首先在这里我们定义了一个KNN object，它带有一个参数叫做n_neighbors=3， 意思就是说我们选择的K值是3.
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# 这部分的代码主要用来做预测以及计算准确率。计算准确率的逻辑也很简单，就是判断预测和实际值有多少是相等的。如果相等则算预测正确，否则预测失败。
correct = np.count_nonzero((clf.predict(X_test) == y_test) == True)
print('Accuracy is: %.3f' % (correct / len(X_test)))
