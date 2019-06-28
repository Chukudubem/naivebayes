import sys
import pandas as pd
import numpy as np



'''
Run using Python 3
'''
class ProcessingData:   
    def __init__(self):
        self.file=None
        self.data= None

    def Load_Data(self, file):
        data = pd.read_csv(file)
        return data
    
    def Train_Test_Split(self, data):
        d = len(data)
        train = data.sample(frac=.85)
        test = data.drop(data.index[train.index])
        return train.reset_index(drop = True), test.reset_index(drop=True)
    
    def Feature_Extraction(self, data):
        '''
        Features:
        1. has_television
        2. has_radio
        3. has_sem
        4. has_display
        5. has_video
        6. has_social
        7. has_no_evidence
        '''
        feature_dict = {}
        for item in data.iterrows():
            feature_dict[item[0]] = {}
            feature_dict[item[0]]['has_television'] = int(any(i in item[1]['map_publisher'] for i in ['television', 'tv']))
            feature_dict[item[0]]['has_radio'] = int(any(i in item[1]['map_publisher'] for i in ['radio', 'trading']))
            feature_dict[item[0]]['has_sem'] = int('sem' in item[1]['map_publisher'])
            feature_dict[item[0]]['has_display'] = int('display' in item[1]['map_publisher'])
            feature_dict[item[0]]['has_video'] = int('video' in item[1]['map_publisher'])
            feature_dict[item[0]]['has_social'] = int(any(i in item[1]['map_publisher'] for i in ['facebook', 'instagram', 'twitter']))
            feature_dict[item[0]]['has_no_evidence'] = int(all(i not in item[1]['map_publisher'] for i in ['television', 'tv', 'radio', 'trading','display', 'video' 'sem''facebook', 'instagram', 'twitter']))    
            feature_dict[item[0]]['target'] = item[1]['channel']
        featured_data = pd.DataFrame.from_dict(feature_dict, orient='index')
        return featured_data.drop('target', axis=1), featured_data.target

class NaiveBayes:
    '''
    Bayesian Classifier:
    Describes the probability of an outcome based on prior knowledge
    of conditions possibly related to the outcome.
    
    Given H = Outcome
          E = Evidence
    To evaluate posterior probability: P(H|E) 
    P(H|E) = P(E|H)P(H) / P(E)
    
    using prior probability: P(H)
          likelihood ratio: P(E|H)/P(E)
    '''
    def __init__(self):
        self.X = None
        self.y = None
        self.features = ['has_television', 'has_radio', 'has_sem', 'has_display', 'has_video',
       'has_social', 'has_no_evidence']
        self.classes = None
        self.conditnal_probs = None
        self.class_prob =None
    
    def prior_likelihood_estimators(self, X, y):
        '''
        Evaluate class probability (prior probability)
        and likelihood ratio
        '''
        self.X = X
        self.y = y
        self.classes = y.unique()
        conditnal_probs = {}
        class_prob = {}
        for class_ in self.classes:
            data_class = X[X.index.isin(y[y==class_].index.values)]
            clsp = {}
            class_total = len(data_class) # Total instances per class
            for col in self.features:
                colp = {}
                for val,cnt in data_class[col].value_counts().items(): 
                    colp[val] = np.divide(cnt,class_total)
                clsp[col] = colp
            conditnal_probs[class_] = clsp
            class_prob[class_] = np.divide(len(data_class),len(X))
        self.conditnal_probs = conditnal_probs
        self.class_prob = class_prob
        return conditnal_probs, class_prob

    def probability_distribution(self, X):
        '''
        Calculates probability distribution of observation
        '''
        probab = {}
        for cl in self.classes:
            pr = self.class_prob[cl]
            for col,val in X.iteritems():
                    try:
                        pr *= self.conditnal_probs[cl][col][val]
                    except KeyError:
                        pr = 0
            probab[cl] = pr    
        return probab

    def classifier(self, X):
        '''
        Returns maximum probable class as outcome
        '''
        probab = self.probability_distribution(X)
        max_ = 0
        max_class = ''
        for cl,pr in probab.items():
            if pr > max_:
                max_ = pr
                max_class = cl
        return max_class

    def fit(self,X, y):
        '''
        Performs classification on training dataset.
        Returns predicted class.
        '''
        self.prior_likelihood_estimators(X,y)
        b = []
        for i in X.index:
            b.append(self.classifier(X.loc[i]))
        return b
    
    def predict(self, test_x):
        '''
        Predicts test data class based on training dataset probability estimations.
        '''
        predictions = []
        for i in test_x.index:
            predictions.append(self.classifier(test_x.loc[i]))
        return predictions
    
    
    
def Accuracy_Score(actual_y, predicted_y):
    '''
    Returns prediction accuracy
    '''
    count=0
    for i in range(len(predicted_y)):
        if str(predicted_y[i]) == str(list(actual_y)[i]):
            count+=1
    print(count,"correct of",len(test))
    print("Accuracy Score: ", np.divide(count,len(actual_y)))


if __name__ == "__main__":
    input_path = str(sys.argv[1])
    '''
    Preprocessing stage:
    1. Loading data
    2. Splitting data into train and test
    3. Feature Creation
    '''
    pr = ProcessingData()
    data = pr.Load_Data(input_path)
    train, test = pr.Train_Test_Split(data)
    X_train, y_train = pr.Feature_Extraction(train)
    X_test, y_test = pr.Feature_Extraction(test)
    '''
    Naive Bayes Classifier
    1. Fit the training data
    2. Predict test data
    3. Print prediction accuracy
    '''
    NB = NaiveBayes()
    clf = NB.fit(X_train, y_train)
    pred = NB.predict(X_test)
    Accuracy_Score(y_test, pred)