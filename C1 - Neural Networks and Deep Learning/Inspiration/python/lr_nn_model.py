import numpy as np
import copy
from base_model import Base_Model

class LR_NN_Model(Base_Model):

    def __init__(self):
        super().__init__()

    def about(self):
        """
        Method to print information about the model
        """
        print(f"""
            Model: {self.d["Model"]}
            learning_rate: {self.d["learning_rate"]}
            num_iterations: {self.d["num_iterations"]}
            w: {self.d["w"]}
            b: {self.d["b"]}
            costs: {self.d["costs"]}
            Y_prediction_test: {self.d["Y_prediction_test"]}
            Y_prediction_train: {self.d["Y_prediction_train"]}
        """)

    def __optimize(self, w, b, X, Y, num_iterations=100, learning_rate = 0.01, print_logs=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_logs -- True to print the loss every 100 steps
        
        Assign:
        params -- dictionary containing the weights w and bias b
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
         
        for i in range(num_iterations):
            # Cost and gradient calculation 
            cost = self.__propagate(w, b, X, Y)
            
            # Retrieve derivatives from grads
            dw = self.grads["dw"]
            db = self.grads["db"]
            
            # update rule (≈ 2 lines of code)
            w = w - (learning_rate*dw)
            b = b - (learning_rate*db)
        
            # Record the costs
            if i % 100 == 0:
                self.costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_logs:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        self.params = {"w": w,
                       "b": b}

    def __propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Assign:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        
        m = X.shape[1]
        
        # FORWARD PROPAGATION (FROM X TO COST)
        # compute activation
        A = self.sigmoid(np.dot(w.T, X) + b)
        # compute cost
        cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m                                

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (np.dot(X, (A - Y).T)) / m
        db = (np.sum(A - Y)) / m
        
        cost = np.squeeze(np.array(cost))
        
        self.grads = {"dw": dw,
                      "db": db}
        
        return cost

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        
        # Compute vector "A" predicting the probabilities of a cat being present in the picture    
        A = self.sigmoid(np.dot(w.T,X) + b)
        
        for i in range(A.shape[1]):
            
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0, i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0
        
        return Y_prediction

    def train(self, num_iterations=2000, learning_rate = 0.01, print_logs=False, model_name='model_lr_nn_XXX.pkl'):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to True to print the cost every 100 iterations
        
        Assign:
        d -- dictionary containing information about the model.
        """

        # initialize parameters with zeros 
        w = np.zeros((self.dataset["X_train"].shape[0], 1)) 
        b = 0.0
        
        # Gradient descent  
        self.__optimize(w, b, self.dataset["X_train"], self.dataset["Y_train"], num_iterations, learning_rate, print_logs)
        
        # Retrieve parameters w and b from dictionary "params"
        w = self.params["w"]
        b = self.params["b"]
        
        # Predict test/train set examples
        Y_prediction_test = self.predict(w, b, self.dataset["X_test"])
        Y_prediction_train = self.predict(w, b, self.dataset["X_train"])
        Y_train = self.dataset["Y_train"]
        Y_test = self.dataset["Y_test"]

        # Print train/test Errors
        if print_logs:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        self.d = {
            "Model": model_name,
            "costs": self.costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}

        self.save_model(model_name)
