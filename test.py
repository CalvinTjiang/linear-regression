import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import LinearRegression as lr

def generate_data(total_features, slope, randomness, range_y):
    '''
    Generate a data
        
    Parameters:
    x : array of data -> numpy array with shape = (n * m)

    Return :
    array of predicted output -> numpy array with shape = (n)
    '''
    y = np.arange(*range_y)
    random_data = np.random.rand(len(y), total_features)

    # Update each feature with slope and randomness
    for i in range(total_features):
        random_data[:,i] += (y * slope[i]) + np.random.uniform(-randomness[i], randomness[i], (len(y)))

    return (random_data[:,0 : total_features], y)

linear_regression = lr.LinearRegression(2)
x, y = generate_data(2, [-0.5, 0.2], [0.3, 1], (0, 10, 0.1))

linear_regression.batch_size = 25
linear_regression.total_epochs = 80
print(linear_regression.cost(x, y))
linear_regression.gradient_descent(x, y)
print(linear_regression.cost(x, y))
linear_regression.gradient_descent(x, y)
print(linear_regression.cost(x, y))
linear_regression.gradient_descent(x, y)
print(linear_regression.cost(x, y))
linear_regression.gradient_descent(x, y)
print(linear_regression.cost(x, y))

# X = x[:,0]
# Y = x[:,1]
# print(X)
# print(Y)
# X, Y = np.meshgrid(X, Y)
# print(X.shape)
# print(Y.shape)

# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-3, 3, 0.25)
# print(X.shape)
# print(Y.shape)
# X, Y = np.meshgrid(X, Y)
# print(X.shape)
# print(Y.shape)
# R = np.sqrt(X**2 + Y**2)
# print(R.shape)
# Z = np.sin(R)
# print(R.shape)
# # can only visualize the first 2 feature from
# # Initialize the 3D scatterplot

prediction = linear_regression.predict(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Draw the point
ax.scatter(x[:,0], x[:,1], y)
ax.scatter(x[:,0], x[:,1], prediction, marker='^')

# Add the label
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')

# Show the 3D scatterplot
plt.show()

