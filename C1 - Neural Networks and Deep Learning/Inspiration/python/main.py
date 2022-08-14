import argparse
import numpy as np
from PIL import Image

from base_model import Base_Model
from lr_nn_model import LR_NN_Model

def parse_arguments():
    """
    Init argparse object and add arguments
    Returns
    -------
    parser.parse_args()
        - options, an object containing values for all of your options
        - args, the list of positional arguments leftover after parsing options
    """

    parser = argparse.ArgumentParser(description = 'Process command line arguments.')
    parser.add_argument('-t',
                        '--train',
                        default = False,
                        action = "store_true",
                        help = 'Train new model on default or provided dataset')
    parser.add_argument('-v',
                        '--verbose',
                        default = False,
                        action = "store_true",
                        help = 'Verbose, show logs')
    parser.add_argument('-i',
                        '--image',
                        default = "../../images/my_image.jpg",
                        help = 'Relative or absolute path to image')
    parser.add_argument('-m',
                        '--model',
                        default = "model_lr_nn_001.pkl",
                        help = 'New trained model name')
    parser.add_argument('-l',
                        '--load',
                        default = "model_lr_nn_001.pkl",
                        help = 'Relative or absolute path to model')
    parser.add_argument('-r',
                        '--learning_rate',
                        type=float,
                        default = 0.01,
                        help = 'Hyperparameter representing the learning rate used in the update rule of optimize()')
    parser.add_argument('-n',
                        '--num_iterations',
                        type=int,
                        default = 2000,
                        help = 'Hyperparameter representing the number of iterations of the optimization loop')
    parser.add_argument('-a',
                        '--about',
                        default = False,
                        help = 'Print information about the model')  

                        
    return parser.parse_args()

if __name__ == "__main__":

    parsed_args = parse_arguments()

    if (parsed_args.train):
        lr_nn_model = LR_NN_Model();
        lr_nn_model.train(num_iterations = parsed_args.num_iterations, learning_rate = parsed_args.learning_rate, print_logs = parsed_args.verbose, model_name = parsed_args.model)
    elif (parsed_args.about):
        lr_nn_model = Base_Model.load(parsed_args.load)
        lr_nn_model.about()
    else:
        lr_nn_model = Base_Model.load(parsed_args.load)
        image = np.array(Image.open(parsed_args.image).resize((64, 64))) / 255
        image = image.reshape((1, 64 * 64 * 3)).T
        my_predicted_image = lr_nn_model.predict(lr_nn_model.params["w"], lr_nn_model.params["b"], image)
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + lr_nn_model.dataset["classes"][int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")