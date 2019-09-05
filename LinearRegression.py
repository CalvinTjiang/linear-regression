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
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)

        Return :
        array of predicted output -> numpy array with shape = (n)
        '''
        # Check if x have a same number of features with weight
        if len(x[0]) != self.total_features:
            raise Exception("Number of features in x is not equal with total feature in weight!")
        
        # Return the output
        return (x @ self.weight[np.newaxis, :].T)[:,0]

    def cost(self, x, y):
        '''
        calculate the error rate of the current weight compared with x and y
        the error rate is calculated by using squared error function (the most common cost function for linear regression)
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)

        Return :
        error rate of the current weight -> float
        '''
        difference = self.predict(x) - y
        squared_error = difference.T @ difference
        return (squared_error/(2 * len(y)))

    def gradient_descent(self, x, y, learning_rate=None, batch_size=None, total_epochs=None):
        '''
        apply gradient descent to reduced the error rate of the current weight
        by using derivative of the cost function
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)
        learning_rate : learning rate of the gradient descent -> positive float
        batch_size : size of each batch -> positive int
        total_epoch : total x and y being fully iterated -> positive int

        Return:
        None
        '''
        if learning_rate == None:
            learning_rate = self.learning_rate
        
        if batch_size == None:
            batch_size = self.batch_size
        
        if total_epochs == None:
            total_epochs = self.total_epochs

        for epoch in range(total_epochs):
            # initial the start batch to 0
            start_batch = 0 
            end_batch = batch_size
            while start_batch < len(x):
                # if the end_batch is more than the length of x 
                if end_batch >= len(x):
                    end_batch = len(x)

                # Divide the train data into smaller batch
                current_x = x[start_batch:end_batch]
                current_y = y[start_batch:end_batch]
                
                # Calculate the derivative
                derivative = (self.predict(current_x) - current_y) @ current_x
                self.weight -= (learning_rate / len(current_y)) * derivative

                # Move to next batch
                start_batch += batch_size
                end_batch += batch_size
    
    def normal_equation(self, x, y):
        '''
        calculate the weight/parameter with lowest cost 
        by using Ordinary Least Squares (OLS) equation
        
        Parameters:
        x : array of data -> numpy array with shape = (n * m)
        y : array of output -> numpy array with shape = (n * m)

        Return:
        None
        '''
        self.weight = (np.linalg.pinv(x.T @ x) @ x.T) @ y