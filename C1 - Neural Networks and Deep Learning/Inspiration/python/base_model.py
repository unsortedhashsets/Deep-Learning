import numpy as np
import h5py
import pickle

class Base_Model:

    # dataset -- dictionary containing dataset's parameters: Train set features, Train set labels, Test set features, Test set labels, List of classes
    dataset = {}
   
    # params -- dictionary containing the weights w and bias b
    params = {}
    # grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    grads = {}
    # costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    costs = []

    # d -- dictionary containing information about the model.
    d = {}

    def __init__(self):
        self.__load_dataset()

    def about(self):
        """
        Method to print information about the model
        """
        print(d)
    
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
    
    def save_model(self, model_name):
        with open(model_name, 'wb') as outp:
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