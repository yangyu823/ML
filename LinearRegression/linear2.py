from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

examDict = {"Time": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25,
                     2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
            "Mark": [10, 22, 13, 43, 20, 22, 33, 50, 62,
                     48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}
# print(examDict)

examOrderedDict = OrderedDict(examDict)
# print(examOrderedDict)

examDf = pd.DataFrame(examOrderedDict)
# print(examDf)

exam_X = examDf.loc[:, "Time"]
exam_y = examDf.loc[:, "Mark"]

# plt.scatter(exam_X, exam_y, color="b", label="Exam data")
# plt.xlabel("hours")
# plt.ylabel("score")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(exam_X, exam_y, train_size=0.8)

# plt.scatter(X_train, y_train, color="b", label="train data")
# plt.scatter(X_test, y_test, color="r", label="test data")
#
# plt.legend(loc=2)
# plt.xlabel("hours")
# plt.ylabel("pass")
#
# plt.show()

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X_train, y_train)

# 截距
a = reg.intercept_
# 回归系数
b = reg.coef_

y_train_pred = reg.predict(X_train)
# print(y_train_pred)

plt.scatter(X_train, y_train, color="b", label="traindata")
plt.scatter(X_test, y_test, color="r", label="test data")
plt.plot(X_train, y_train_pred, color="black", linewidth=3, label="best line")
plt.legend(loc=2)
plt.xlabel("time")
plt.ylabel("score")
plt.show()

# 相关系数矩阵：
rDf = examDf.corr()
print(rDf)

# 决定系数：
score = reg.score(X_test, y_test)
print(score)
