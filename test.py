import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import LinearRegression as lr
import timeit

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

# Generate random data
x, y = generate_data(2, [-0.5, 0.2], [0.3, 1], (0, 10, 0.1))

# First model use gradient descent
linear_regression = lr.LinearRegression(2)
linear_regression.batch_size = 25
linear_regression.total_epochs = 40

# Second model use normal equation
linear_regression_1 = lr.LinearRegression(2)
linear_regression_1.weight = np.copy(linear_regression.weight)

print(f'---Gradient Descent---')
print(f'Initial cost : {linear_regression.cost(x, y)}')

# Start gradient Descent on first model
start = timeit.default_timer()
linear_regression.gradient_descent(x, y)
linear_regression.gradient_descent(x, y)
taken = (timeit.default_timer() - start)

print(f'Time Taken : {taken}')
print(f'Result Cost : {linear_regression.cost(x, y)} \n')


print(f'---Normal Equation---')
print(f'Initial cost : {linear_regression_1.cost(x, y)}')

# Start Normal equation on second model
start = timeit.default_timer()
linear_regression_1.normal_equation(x, y)
taken = (timeit.default_timer() - start)

print(f'Time Taken : {taken}')
print(f'Result Cost : {linear_regression_1.cost(x, y)}')


# Predict the output
prediction = linear_regression.predict(x)
prediction_1 = linear_regression_1.predict(x)


# Initialize 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the point
ax.scatter(x[:,0], x[:,1], y)
ax.scatter(x[:,0], x[:,1], prediction, marker='^')
ax.scatter(x[:,0], x[:,1], prediction_1, marker='*')

# Add the label
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')

# Show the 3D scatterplot
plt.show()

