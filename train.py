mport argparse
import numpy as np
import time
from main import NeuralNetwork
import pandas as pd

parser = argparse.ArgumentParser(description='Neural Networks provided by Sina Pourmahmoud')
parser.add_argument('--activation', action='store', dest='activation', required=False, default='sigmoid', help='activation function: sigmoid/relu')
parser.add_argument('--batch_size', action='store', dest='batch_size', required=False, default=128)
parser.add_argument('--optimizer', action='store', dest='optimizer', required=False, default='momentum', help='optimizer: sgd/momentum')
parser.add_argument('--l_rate', action='store', dest='l_rate', required=False, default=1e-3, help='learning rate')
parser.add_argument('--beta', action='store', dest='beta', required=False, default=.9, help='beta in momentum optimizer')
args = parser.parse_args()


def one_hot(x,k,dtype = np.float64):
    x = x.reshape(-1)
    return np.array(x[:,None] == np.arange(k),dtype)

def main():
    
    df = pd.read_csv('./train.csv').values
    x = df[:,1:].astype(np.int32)
    y = df[:,0].astype(np.int32).reshape(-1,1)
    x = x / 255.0


    options = 10
    y_new = one_hot(y,options)
    train_size = 30000


    test_size = x.shape[0] - train_size
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y_new[:train_size], y_new[train_size:]
    shuffle_index = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    print("Training data: {} {}".format(x_train.shape, y_train.shape))
    print("Test data: {} {}".format(x_test.shape, y_test.shape))
    print("Start training!")


    dnn = NeuralNetwork(sizes=[784, 64, 10], activation=args.activation)
    dnn.train(x_train, y_train, x_test, y_test, batch_size=int(args.batch_size), optimizer=args.optimizer, l_rate=float(args.l_rate), beta=float(args.beta))



if __name__=="__main__":
    main()
