from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
                 [168, 57], [172, 60], [176, 62], [180, 65],
                 [184, 69], [188, 72]])

# Sample Data Plot
x, y = data[:, 0].reshape(-1, 1), data[:, 1]
plt.scatter(x, y, color="black")
plt.suptitle(data.shape)


# Regression Model
reg = linear_model.LinearRegression()
reg.fit(x, y)
a = reg.intercept_
b = reg.coef_
c = reg.score(x, y)


# Plot regression model
plt.plot(x, reg.predict(x), color='b')

# Plot x & y axio
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')
plt.show()

# Predict weight
print("Standard weight for person with 163 is %.2f" % reg.predict([[163]]))
