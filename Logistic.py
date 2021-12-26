from os import pread
import numpy as np
import pandas as pd
np.random.seed(0)

# predicted_test_labels_perceptron = perceptron.predict(test_data)
    # predicted_test_labels_unseen_perceptron = perceptron.predict(test_data_unseen)

    # predicted_train_labels_logistic = logistic.predict(train_data)
    # predicted_test_labels_logistic = logistic.predict(test_data)
    # predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    # print('\n\n-------------Perceptron Performance-------------\n')
    # # This command also runs the evaluation on the unseen test set
    # eval(train_data['Label'].tolist(), predicted_train_labels_perceptron, test_data['Label'].tolist(),
    #      predicted_test_labels_perceptron, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_perceptron)

    # print('\n\n-------------Logistic Function Performance-------------\n')
    # # This command also runs the evaluation on the unseen test
    # eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
    #      predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)

"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the logistic model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the logistic function but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (logistic) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produces as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the logistic function. 
It is up to your discretion on if you want to use them or add your own methods.
"""


class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the logistic function as instance
        attributes.
        """
        self.weights = None
        self.bias = None


    def sigmoid(self,x):
        x=np.array(x,dtype=np.float64)
        return 1/(1 + np.exp(-x))

    def feature_extraction(self, method=None):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training. You need to implement unigram, bigram and trigram.
        """
        if method == 'unigram':
            return

        if method == 'bigram':
            return
        
        if method == 'trigram':
            return

    def logistic_loss(self, predicted_label, true_label):
        """
        Optional helper method to code the loss function.
        """
        return
    
    def regularizer(self, method=None, lam=None):
        """
        You need to implement at least L1 and L2 regularizer
        """
        if method == 'L1':
            return
        if method == 'L2':
            return

    def stochastic_gradient_descent(self, data, labels):
        """
        Optional helper method to compute a gradient update for a single point.
        """
        return

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights during stochastic gradient descent.
        """
        self.weights = new_weights

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias during stochastic gradient descent.
        """
        self.bias = new_bias

    def predict_labels(self, data_point):
        
        """
        Optional helper method to produce predictions for a single data point
        """
        return

    def train(self, labeled_data, learning_rate, max_epochs, lam, feature_method='2gram', reg_method='L2'):
        X_train, Y_train=labeled_data['ngram_vector'], labeled_data['Label']
        # X_test, Y_test=test_data['ngram_vector'], test_data['Label']
        X_train = X_train.values
        Y_train = Y_train.values
        rv=X_train[0]
        for i in X_train[1:]:
            rv=np.vstack((rv,i))
        X_train=rv
        X_train = X_train.T
        Y_train = Y_train.reshape(1, X_train.shape[1])

    
        m = X_train.shape[1]
        n = X_train.shape[0]
        X=np.array(X_train, dtype=np.float64)
        Y=np.array(Y_train, dtype=np.float64)
        
        self.weights = np.zeros((n,1))
        self.bias = 0
        
        
        for i in range(max_epochs):
            ri=np.random.randint(m)
            Z = np.array(np.dot(self.weights.T, X[:,ri:ri+1]), dtype=np.float64) + self.bias
            A = np.array(self.sigmoid(Z), dtype=np.float64)
            # Stochastic Gradient Descent
            if(reg_method=='L2'):
                dW = np.dot(A-Y[:,ri:ri+1], X[:,ri:ri+1].T)+lam*np.sum(np.square(self.weights/m))
                dB = np.sum(A - Y[:,ri:ri+1])+lam*np.sum(np.square(self.weights/m))
            elif(reg_method=='L1'):
                dW = np.dot(A-Y[:,ri:ri+1], X[:,ri:ri+1].T)+lam*np.sum(self.weights/m)
                dB = np.sum(A - Y[:,ri:ri+1])+lam*np.sum(self.weights/m)
            
            self.weights = self.weights - learning_rate*dW.T
            self.bias = self.bias - learning_rate*dB
        

        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.
        
        'learning_rate' and 'max_epochs' are the same as in HW2. 'reg_method' represents the regularier, 
        which can be 'L1' or 'L2' as in the regularizer function. 'lam' is the coefficient of the regularizer term. 
        'feature_method' can be 'unigram', 'bigram' or 'trigram' as in 'feature_extraction' method. Once you find the optimal 
        values combination, update the default values for all these parameters.

        There is no limitation on how you implement the training process.
        """


    def predict(self, data):
        predicted_labels = []
        X, Y=data['ngram_vector'], data['Label']
        X = X.values
        rv=X[0]
        for i in X[1:]:
            rv=np.vstack((rv,i))
        X=rv
        X = X.T
        Z = np.dot(self.weights.T, X) + self.bias
        predicted_labels = self.sigmoid(Z)
        
        predicted_labels = predicted_labels > 0.5
        
        predicted_labels = np.array(predicted_labels, dtype = 'int64')

        
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        return predicted_labels.T

# def bagging(X_train, Y_train, num):
    #     Weights=[]
    #     Bias=[]
    #     for _ in range(num):
    #         rand=np.random.randint(X_train.shape[1], size=X_train.shape[1]//num)
    #         X_train_=X_train[:,rand]
    #         Y_train_=Y_train[:,rand]
    #         iterations = 7000
    #         learning_rate = 0.01
    #         W,B= model(X_train, Y_train, learning_rate = learning_rate, iterations = iterations)
    #         Weights.append(W)
    #         Bias.append(B)
    #     return Weights,Bias

    # def bag_accuracy(X, Y, W, B):
    #     A=arr = np.empty((0,Y.shape[1]), int)
    #     for i,j in enumerate(W):
    #         Z = np.dot(W[i].T, X) + B[i]
    #         A_ = sigmoid(Z)

    #         A_ = A_ > 0.5

    #         A =np.vstack((A, np.array(A_, dtype = 'int64')))
    #     AF= np.array(stats.mode(A))
    #     print(AF[0].shape)
    #     acc = (1 - np.sum(np.absolute(AF[0] - Y))/Y.shape[1])*100

    # print("Accuracy of the model is : ", round(acc, 2), "%")

"""In HW2, I implemented bag of words in HW2, the accuracy that it gave was around 75% on test data, on increasing the number of iterations there, the training data used to overfit and give high accuracy of 94% but in HW3, I have implemented 2-gram and L1 and L2 regularizer, the regularizer helped in greatly reducing the overfitting by punishing the high parameters. The L2 regularizer worked best for me."""