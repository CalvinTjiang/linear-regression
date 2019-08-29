import numpy as np

class LinearRegression:
    def __init__(self, total_features=1):
        '''
        '''
        self.total_features = total_features
        self.weight = np.random.rand(total_features)
        self.learning_rate = 1e-3
        self.batch_size = total_features
        self.total_epochs = total_features
    
    def predict(self, x):
        '''
        Predict an array of output from an array of input data using hypothesis function 
        parameter:
        x : array of data -> 2 Dimesional Numpy array

        Return :
        y : array of predicted output -> 1 Dimesional Numpy array
        '''
        rows, column = x.shape
        if column != self.total_features:
            raise Exception("Number of features in x is not equal with total feature in weight!")
        return (self.weight.T @ x)
        pass

    def cost(self):
        pass

    def gradient_descent(self, x, y, learning_rate=self.learning_rate, batch_size=self.batch_size, total_epochs=self.total_epochs):
        for epoch in total_epochs:
            for batch in batch_size:
                pass
        pass
    
    def normal_equation(self,):
        pass