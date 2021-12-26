import pandas as pd
import pandas as pd
import string
import re
import numpy as np

main_dict={}
vec=[]
sw=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", ""," ", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", "whens", "where", "\n","\r","wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
def generate_ngrams(text,n):
    tokens=re.split("\\s+",text)
    ngrams=[]
    for i in range(len(tokens)-n+1):
       temp=[tokens[j] for j in range(i,i+n)]
       ngrams.append(" ".join(temp))

    return ngrams

def remove_stopwords(text):
    text=text.lower()
    text1 = re.sub("[^\w]", " ",  text).split()
    # text1=text.lower().split(" ")
    
    for word in text1:
        if word in sw:
            text1.remove(word)
    return " ".join(text1)

def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans
def remove_nl(text):
    return text.replace('\r', '').replace('\n', '')

def generate_dict(ngrams):
    for i in ngrams:
        if(i in main_dict):
            main_dict[i]=main_dict[i]+1
        else:
            main_dict[i]=1
def vectorize(ngram):
    n_dict={}
    for elem in ngram:
        if(elem not in n_dict):
            n_dict[elem]=1
        else:
            n_dict[elem]=n_dict[elem]+1
    n_vect=[]
    for i in vec:
        if i in n_dict:
            n_vect.append(n_dict[i])
        else:
            n_vect.append(0)
    return n_vect
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
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
Execution.py is for evaluating your models on the datasets available to you. You can use 
this program to test the accuracy of your models by calling it in the following way:
    
    import Execution
    Execution.eval(o_train, p_train, o_test, p_test)
    
In the sample code, o_train is the observed training labels, p_train is the predicted training labels, o_test is the 
observed test labels, and p_test is the predicted test labels. 
"""

def split_dataset(all_data):
    train_data = None
    test_data = None
    test_data = all_data.sample(frac=0.2)
    train_data = all_data.drop(test_data.index)
    """
    This function will take in as input the whole dataset and you will have to program how to split the dataset into
    training and test datasets. These are the following requirements:
        -The function must take only one parameter which is all_data as a pandas dataframe of the raw dataset.
        -It must return 2 outputs in the specified order: train and test datasets
        
    It is up to you how you want to do the splitting of the data.
    """
    return train_data, test_data

"""
This function should not be changed at all.
"""
def eval(o_train, p_train, o_val, p_val, o_test, p_test):
    print('\nTraining Accuracy Result!')
    accuracy(o_train, p_train)
    print('\nTesting Accuracy Result!')
    accuracy(o_val, p_val)
    print('\nUnseen Test Set Accuracy Result!')
    accuracy(o_test, p_test)

    # return ([accuracy(o_train, p_train),accuracy(o_val, p_val),accuracy(o_test, p_test)])

"""
This function should not be changed at all.
"""
def accuracy(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    # return str(float(match) / num)
    print('***************\nAccuracy: '+str(float(match) / num)+'\n***************')

"""
(c)	For logistic regression I have tried various I have tried various feature engineering mechanisms in this. First the raw text in the csv file given has many punctuations, delimiters they are removed, next the stop words are removed to make sure that only words of significant importance are present. Next, I made a bag of words representation of the all the words in the file. In this using a dictionary I found out all the unique words an replaced all the text columns with a large vector of the word frequencies in their respective sentences.
Next I tried the n-gram approach in which for all the bag of words I tried to make n-grams which hugely exploded the feature space and increased the vector size. For instance for bag of words in got a vector of size around 38000  but for 2-gram the vector size is 430000. This hugely increased the computation time. Same is the case for 3-gram, 4-gram and 5-gram. One observation is although we made 4-gram , 5 -gram and exploded the feature space most of the vectors are 0 as our test data is very little. These approaches are very useful when our test data is very high so that we can find significant occurances in all the n-grams. One more observation is that as we increase to higher order n-grams it tends to overfitting, the training data increases while the test data seems to reduce drastically.
I have tried both L1 and L2 regularizer to overcome overfitting
"""


if __name__ == '__main__':
    """
    The code below these comments must not be altered in any way. This code is used to evaluate the predicted labels of
    your models against the ground-truth observations.
    """
    from Logistic import Logistic
    from Ensemble import Ensemble
    all_data = pd.read_csv('data.csv', index_col=0)
    train_data, test_data = split_dataset(all_data)

    train_data['Text']= train_data['Text'].apply(lambda x:remove_nl(x))
    train_data['text_']= train_data['Text'].apply(lambda x:remove_punctuation(x))
    train_data['text_sw']= train_data['text_'].apply(lambda x:remove_stopwords(x))
    train_data['text_sw']= train_data['text_sw'].apply(lambda x:remove_stopwords(x))
    train_data['text_sw']= train_data['text_sw'].apply(lambda x:remove_stopwords(x))
    train_data['text_ng']= train_data['text_sw'].apply(lambda x:generate_ngrams(x.strip(),2)) # 2 gram is selected
    train_data['text_ng'].apply(lambda x:generate_dict(x))
    # vec = list(main_dict.keys()) #This is the overall vocabulary present in the train set
    selected_set=dict(sorted(main_dict.items(), key=lambda x: x[1], reverse=True)[0:1000])
    vec=list(selected_set.keys()) #This is the this is the vocabulary we selected for test
    train_data['ngram_vector']= train_data['text_ng'].apply(lambda x:vectorize(x))
    train_data=train_data.drop(columns=['text_','text_sw','text_ng','Text'])

    test_data['Text']= test_data['Text'].apply(lambda x:remove_nl(x))
    test_data['text_']= test_data['Text'].apply(lambda x:remove_punctuation(x))
    test_data['text_sw']= test_data['text_'].apply(lambda x:remove_stopwords(x))
    test_data['text_sw']= test_data['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data['text_sw']= test_data['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data['text_ng']= test_data['text_sw'].apply(lambda x:generate_ngrams(x.strip(),2)) # # 2 gram is selected
    test_data['ngram_vector']= test_data['text_ng'].apply(lambda x:vectorize(x))
    test_data=test_data.drop(columns=['text_','text_sw','text_ng','Text'])

    # placeholder dataset -> when we run your code this will be an unseen test set your model will be evaluated on
    test_data_unseen = pd.read_csv('test_data.csv', index_col=0)
    test_data_unseen['Text']= test_data_unseen['Text'].apply(lambda x:remove_nl(x))
    test_data_unseen['text_']= test_data_unseen['Text'].apply(lambda x:remove_punctuation(x))
    test_data_unseen['text_sw']= test_data_unseen['text_'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_sw']= test_data_unseen['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_sw']= test_data_unseen['text_sw'].apply(lambda x:remove_stopwords(x))
    test_data_unseen['text_ng']= test_data_unseen['text_sw'].apply(lambda x:generate_ngrams(x.strip(),2)) # # 2 gram is selected
    test_data_unseen['ngram_vector']= test_data_unseen['text_ng'].apply(lambda x:vectorize(x))
    test_data_unseen=test_data_unseen.drop(columns=['text_','text_sw','text_ng','Text'])

    train_list=[]
    val_list=[]
    start = 0 
    for i in range(5):
        overall = list(range(train_data.shape[0]))
        tbr = list(range(start,start+train_data.shape[0]//5))
        start= start+train_data.shape[0]//5
        for nm in tbr:
            overall.remove(nm)
        train_list.append(train_data.iloc[overall])
        train_list.append(train_data.iloc[tbr])
        # perceptron = Perceptron()
    logistic = Logistic()
    print("Training")
    # perceptron.train(train_data)
    # logistic.train(train_data)

    # predicted_train_labels_perceptron = perceptron.predict(train_data)
    # predicted_test_labels_perceptron = perceptron.predict(test_data)
    # predicted_test_labels_unseen_perceptron = perceptron.predict(test_data_unseen)
    # logistic.train(train_data)

    # predicted_train_labels_logistic = logistic.predict(train_data)
    # predicted_test_labels_logistic = logistic.predict(test_data)
    # predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    # placeholder dataset -> when we run your code this will be an unseen test set your model will be evaluated on
    # test_data_unseen = pd.read_csv('test_data.csv', index_col=0)

    logistic = Logistic()
    logistic.train(train_data,learning_rate=0.07, max_epochs=10000, lam=0.1,reg_method='L2')
    predicted_train_labels_logistic = logistic.predict(train_data)
    predicted_test_labels_logistic = logistic.predict(test_data)
    predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    ensemble = Ensemble()
    ensemble.train(train_data,learning_rate=0.07, max_epochs=10000, lam=0.1,num_clf=5)
    predicted_train_labels_ensemble = ensemble.predict(train_data)
    predicted_test_labels_ensemble = ensemble.predict(test_data)
    predicted_test_labels_unseen_ensemble = ensemble.predict(test_data_unseen)



    print('\n\n-------------Logistic Function Performance-------------\n')
    # This command also runs the evaluation on the unseen test
    # print("train_data['Label']:",train_data['Label'].shape)
    # print("predicted_train_labels_logistic:",predicted_train_labels_logistic.shape)
    ll=eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
        predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)
    
    
    print('\n\n-------------Ensemble Method Performance-------------\n')
    # print("train_data['Label']:",train_data['Label'].shape)
    # print("predicted_train_labels_ensemble:",predicted_train_labels_ensemble.shape)
    eval(train_data['Label'].tolist(), predicted_train_labels_ensemble, test_data['Label'].tolist(),
        predicted_test_labels_ensemble, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_ensemble)

""""In HW2, I implemented bag of words in HW2, the accuracy that it gave was around 75% on test data, on increasing the number of iterations there, the training data used to overfit and give high accuracy of 94% but in HW3, I have implemented 2-gram and L1 and L2 regularizer, the regularizer helped in greatly reducing the overfitting by punishing the high parameters. The L2 regularizer worked best for me."""
