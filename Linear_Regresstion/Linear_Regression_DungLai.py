import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Visualize data 
plt.plot(X, y, 'ro')

ones = np.ones((X.shape[0], 1), dtype = np.int8)

X = np.concatenate((X, ones), axis = 1)

x= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


x0 = np.array([[147,190]]).T
y0 = x0*x[0][0] + x[1][0]

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

plt.plot(x0, y0)

plt.show()