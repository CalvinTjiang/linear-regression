import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

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


x, y = generate_data(2, [-0.5, 0.2], [0.3, 1], (0, 10, 0.1))

# can only visualize the first 2 feature from
# Initialize the 3D scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the point
ax.scatter(x[:,0], x[:,1], y)

# Add the label
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')

# Show the 3D scatterplot
plt.show()
