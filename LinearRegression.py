import numpy as np

class LinearRegression:
    '''
    LinearRegression Class
    '''
    def __init__(self, total_features=1):
        self.total_features = total_features
        self.weight = np.random.rand(total_features)
        self.learning_rate = 1e-3
        self.batch_size = total_features
        self.total_epochs = total_features
    
    def predict(self, x):
        '''
        Predict an array of output from an array of input data using hypothesis function 
        parameter:
        x : array of data -> 2 Dimensional Numpy array

        Return :
        array of predicted output -> 1 Dimensional Numpy array
        '''
        # Check if x have a same number of features with weight
        if len(x[0]) != self.total_features:
            raise Exception("Number of features in x is not equal with total feature in weight!")
        
        # Return the output
        return (self.weight.T @ x)

    def cost(self, x, y):
        '''
        calculate the error rate of the current weight compared with x and y
        the error rate is calculated by using squared error function (the most common cost function for linear regression)
        parameter:
        x : array of data -> 2 Dimensional Numpy array
        y : array of ouput -> 2 Dimensional Numpy array

        Return :
        error rate of the current weight -> Float
        '''
        difference = self.predict(x) - y
        squared_error = difference.T @ difference
        return (squared_error/(2 * len(y)))

    def gradient_descent(self, x, y, learning_rate=self.learning_rate, batch_size=self.batch_size, total_epochs=self.total_epochs):
        for epoch in total_epochs:
            for batch in batch_size:
                pass
        pass
    
    def normal_equation(self,):
        pass