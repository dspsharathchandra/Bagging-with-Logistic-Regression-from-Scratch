import numpy as np
"""
You only need to implement bagging.
"""
from scipy import stats
"""(c)	For logistic regression I have tried various I have tried various feature engineering mechanisms in this. First the raw text in the csv file given has many punctuations, delimiters they are removed, next the stop words are removed to make sure that only words of significant importance are present. Next, I made a bag of words representation of the all the words in the file. In this using a dictionary I found out all the unique words an replaced all the text columns with a large vector of the word frequencies in their respective sentences.
Next I tried the n-gram approach in which for all the bag of words I tried to make n-grams which hugely exploded the feature space and increased the vector size. For instance for bag of words in got a vector of size around 38000  but for 2-gram the vector size is 430000. This hugely increased the computation time. Same is the case for 3-gram, 4-gram and 5-gram. One observation is although we made 4-gram , 5 -gram and exploded the feature space most of the vectors are 0 as our test data is very little. These approaches are very useful when our test data is very high so that we can find significant occurances in all the n-grams. One more observation is that as we increase to higher order n-grams it tends to overfitting, the training data increases while the test data seems to reduce drastically.
I have tried both L1 and L2 regularizer to overcome overfitting
For hyper-parameter tuning below are the graphs
"""
"""
(d)	For ensemble model, I randomly sampled a portion of data from the trainng data and created a logistic regression model from that with the optimal hyperparameters obtained earlier. This is done n times, where n is the number of trees. Thus, we get n different logistic regression models from n randomly sampled training data sets. Now during prediction, we make predictions from all the n models and take mode of those as the result. I tried this for various n and below is the plot."""
np.random.seed(16)
class Ensemble():
    def __init__(self):
        self.weights = []
        self.bias = []
        """
        You may initialize the parameters that you want and remove the 'return'
        """
        return

    def sigmoid(self,x):
        x=np.array(x,dtype=np.float64)
        return 1/(1 + np.exp(-x))

    def model(self, X_train,Y_train, learning_rate, max_epochs, lam, feature_method=None, reg_method=None):

    
        m = X_train.shape[1]
        n = X_train.shape[0]
        X=np.array(X_train, dtype=np.float64)
        Y=np.array(Y_train, dtype=np.float64)
        
        W = np.zeros((n,1))
        B = 0

        
        for i in range(max_epochs):
            ri=np.random.randint(m)
            Z = np.array(np.dot(W.T, X[:,ri:ri+1]), dtype=np.float64) + B
            A = np.array(self.sigmoid(Z), dtype=np.float64)

            
            # Stochastic Gradient Descent
            dW = np.dot(A-Y[:,ri:ri+1], X[:,ri:ri+1].T)+0.7*np.sum(np.square(W/m))
            dB = np.sum(A - Y[:,ri:ri+1])+0.7*np.sum(np.square(W/m))
            
            W = W - learning_rate*dW.T
            B = B - learning_rate*dB
        return W,B
 
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

    def feature_extraction(self):
        """
        Use the same method as in Logistic.py
        """
        return
    
    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        return

    def train(self, labeled_data, learning_rate, max_epochs, lam, num_clf):
        X_train, Y_train=labeled_data['ngram_vector'], labeled_data['Label']
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
        X_train=np.array(X_train, dtype=np.float64)
        Y_train=np.array(Y_train, dtype=np.float64)
        W=[]
        B=[]
        for _ in range(num_clf):
            rand=np.random.randint(X_train.shape[1], size=X_train.shape[1]//num_clf)
            X_train_sampled=X_train[:,rand]
            Y_train_sampled=Y_train[:,rand]
            learning_rate = 0.01
            W,B= self.model(X_train_sampled, Y_train_sampled,learning_rate, max_epochs, lam)
            self.weights.append(W)
            self.bias.append(B)


        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.

        There is no limitation on how you implement the training process.
        """
        return

    def predict(self, data):

        predicted_labels = []
        X, Y=data['ngram_vector'], data['Label']
        X = X.values
        Y= Y.values
        rv=X[0]
        for i in X[1:]:
            rv=np.vstack((rv,i))
        X=rv
        rv=Y[0]
        for i in Y[1:]:
            rv=np.vstack((rv,i))
        Y=rv
        rv=X[0]
        for i in X[1:]:
            rv=np.vstack((rv,i))
        X=rv
        X = X.T

        A = np.empty((0,Y.shape[0]), int)
        for i,j in enumerate(self.weights):
            Z = np.dot(self.weights[i].T, X) + self.bias[i]
            A_ = self.sigmoid(Z)

            A_ = A_ > 0.5

            A =np.vstack((A, np.array(A_, dtype = 'int64')))
        AF= np.array(stats.mode(A))
        predicted_labels = AF[0].T
        
        
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        return predicted_labels


"""In HW2, I implemented bag of words in HW2, the accuracy that it gave was around 75% on test data, on increasing the number of iterations there, the training data used to overfit and give high accuracy of 94% but in HW3, I have implemented 2-gram and L1 and L2 regularizer, the regularizer helped in greatly reducing the overfitting by punishing the high parameters. The L2 regularizer worked best for me."""
    

