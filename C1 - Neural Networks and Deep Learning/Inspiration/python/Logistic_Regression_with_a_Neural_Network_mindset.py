import numpy as np
import h5py
import copy
import pickle
from PIL import Image

class Base_Model:

    # dataset -- dictionary containing dataset's parameters: Train set features, Train set labels, Test set features, Test set labels, List of classes
    dataset = {}
   
    # params -- dictionary containing the weights w and bias b
    params = {}
    # grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    grads = {}
    # costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    costs = []
   
    # learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    learning_rate = 0

    # d -- dictionary containing information about the model.
    d = {}

    def __init__(self, learning_rate = 0.01):
        print("TEST")
        self.__load_dataset()
        self.learning_rate = learning_rate

    def __load_dataset(self):
        """
        Private method to load dataset from h5 database

        Assign:
        dataset -- dictionary containing dataset's parameters
        """

        train_dataset = h5py.File('../../datasets/train_catvnoncat.h5', "r")
        # Train set features
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
        # Train set labels
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File('../../datasets/test_catvnoncat.h5', "r")
        # Test set features
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        # Test set labels
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

        # The list of classes
        classes = np.array(test_dataset["list_classes"][:])
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
        
        train_set_x = train_set_x_flatten / 255
        test_set_x = test_set_x_flatten / 255

        self.dataset = {
            "X_train": train_set_x,
            "Y_train": train_set_y_orig, 
            "X_test" : test_set_x, 
            "Y_test" : test_set_y_orig, 
            "classes" : classes
        }

    def save_model(self):
        with open(f"model_lr_nn_001.pkl", 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Static method to download trained model

        Arguments:
        filename -- A scalar or numpy array of any size.

        Return:
        obj -- loaded object
        """
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
            return obj

    @classmethod
    def sigmoid(cls, z):
        """
        Static method to calculate the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        
        s = 1 / (1 + np.exp(-z))
        
        return s

class LR_NN_Model(Base_Model):

    def __init__(self, mode, infoMsg, learning_rate = 0.01):
        super().__init__(mode, learning_rate)

    def __optimize(self, w, b, X, Y, num_iterations=100, print_logs=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        
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
            w = w - (self.learning_rate*dw)
            b = b - (self.learning_rate*db)
        
            # Record the costs
            if i % 100 == 0:
                self.costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_logs:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        self.params = {"w": w,
                       "b": b}

    @classmethod
    def __propagate(cls, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        
        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
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

    def train(self, num_iterations=2000, print_logs=False):
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
        
        Returns:
        d -- dictionary containing information about the model.
        """

        # initialize parameters with zeros 
        w = np.zeros((self.dataset["X_train"].shape[0], 1)) 
        b = 0.0
        
        # Gradient descent  
        self.__optimize(w, b, self.dataset["X_train"], self.dataset["Y_train"], num_iterations, print_logs)
        
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

        
        self.d = {"costs": self.costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : self.learning_rate,
             "num_iterations": num_iterations}

        self.save_model()
        
if __name__ == "__main__":
    #lr_nn_model = LR_NN_Model("predict", "Need to add args parser", 0.01);
    #lr_nn_model.train(print_logs=True)
    
    lr_nn_model = Base_Model.load("model_lr_nn_001.pkl")

    # TODO: Add prepare image method (after parser)
    my_image = "my_image.jpg"  
    fname = "../../images/" + my_image
    image = np.array(Image.open(fname).resize((64, 64)))
    image = image / 255
    image = image.reshape((1, 64 * 64 * 3)).T
    my_predicted_image = lr_nn_model.predict(lr_nn_model.params["w"], lr_nn_model.params["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + lr_nn_model.dataset["classes"][int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")