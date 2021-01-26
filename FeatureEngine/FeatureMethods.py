from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
Two approaches:
(1) - Classifying transacations as HCP or NOT HCP
(2) - Classifying a non-HCP transaction as correct or incorrect (i.e. anomilies). Note that
while you could also check for HCP's classified as correct or incorrect, there's less risk,
and completeness on that side is less of a concern.

The second approach will perform better with supervised learning. 
'''

#If you need a beginner guide for the methods in this module:
#https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

#JE detection algo
#https://medium.com/analytics-vidhya/deep-anomaly-detection-9f19896c8b2

#Anomoly detection
#https://medium.com/datadriveninvestor/how-machine-learning-can-enable-anomaly-detection-eed9286c5306#:~:text=An%20example%20of%20performing%20anomaly,distance%20from%20the%20closest%20cluster.&text=During%20the%20testing%20process%2C%20determine,point%20from%20the%20mean%20value.
#supervised = better for anomoly detection (i.e. supervised = classification of correct AND wrong ones)
#unsupervised = not as good, but can work

#Pipeline methods to predict future vals
#https://machinelearningmastery.com/how-to-connect-model-input-data-with-predictions-for-machine-learning/

#Implementing scalers as methods
#interesting implemention https://stackoverflow.com/questions/51536227/sklearn-method-in-class
#Can I inherit current object into minmaxscaler one?
#https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/

#Consider automated model retraining
#https://www.inawisdom.com/machine-learning/machine-learning-automated-model-retraining-sagemaker/


#Add a final method that removes the ORIGINAL (unformated features) and returns a numpy array which can
#be passed to an ML model 

#Evaluate if it makes sense to implement HDF5 format

#Add a method for group labeling that takes two paramaters: 
#(1) - array of breakouts [1-100, 101-200, etc] 
#(2) - field name

#Workflow:
#1 initialize FunctionFeaturizer with df and transform method

#Add a method for feature hasing
class FunctionFeaturizer:

    def __init__(self, df):
        self.df = df
        self.dfscaled = None #val gets set if outputFeatures method is called
        self.originalFeatures = df.columns.tolist()
        self.labelEncodeObj = LabelEncoder()

    def concatVector(self, textSeries, ngramMin = 1, ngramMax = 1):
        '''
        Returns a matrix of token counts.
        Required param textSeries takes a string of the textfield name from df
        Second and third paramaters (ngramMin and ngramMax) are used to define
        features of unigram, bigrams, both, etc
        '''
        #scikit learn CountVertorizer object
        #Note: to consider permutating certain paramaters in the CV object
        cv = CountVectorizer(ngram_range=(ngramMin, ngramMax),stop_words='english')
        text_features = cv.fit_transform(self.df[textSeries].values)
        vect_df = pd.DataFrame(text_features.todense(), columns=cv.get_feature_names())
        dfConcat = pd.concat([self.df, vect_df], axis=1)
        self.df = dfConcat
        return
    
    def encodeLabels(self, labelSeries):
        '''
        Returns labels encoded as integers
        Required param labelSeries takes a string of the label field name from df.
        Note: utilize label encoded for encoding where order does not matter 
        '''
        #how to implement one hot encoding vs. label encode? https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor
        #https://datascience.stackexchange.com/questions/39317/difference-between-ordinalencoder-and-labelencoder/64177
        #Note: standard label encoding better for decision trees
        #labelEncodeObj = LabelEncoder()
        labels = self.labelEncodeObj.fit_transform(self.df[labelSeries])
        label_mappings = {index: label for index, label in enumerate(self.labelEncodeObj.classes_)}
        self.df[labelSeries+'_labeled'] = labels
        return
    
    def oneHotEncodeLabels(self, labelSeries):
        '''
        Returns labels encoded in one-hot format (i.e. one feature per label)
        Leverages the pandas get_dummies method. 
        '''
        self.df = pd.get_dummies(self.df, columns=[labelSeries])
        return
    
    #To add an optional param that will allow you to keep certain original features
    def outputFeatures(self):
        '''
        Returns only transformed features and excludes original untransformed features
        '''
        self.dfscaled = self.df[self.df.columns.difference(self.originalFeatures)]
        #outputDf = self.df[self.df.columns.difference(self.originalFeatures)]
        #return outputDf        
        return
  
    #To add a fillna method that takes different statistic paramaters as inputt to method fillnastat(median/mode/mean)

#https://www.codecademy.com/forum_questions/560afacd86f552c8a70001dd

#def minmax():
#    return preprocessing.MinMaxScaler()


class FeatureEngine(FunctionFeaturizer):
    def __init__(self, df):
        FunctionFeaturizer.__init__(self, df)
        self.scaler = preprocessing.MinMaxScaler()
        #self.scaler = scaler
#https://stackoverflow.com/questions/51536227/sklearn-method-in-class
    def fit(self, X, y=None):
        return self

    #https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
    def fit_transform(self):
        self.scaler.fit(self.dfscaled[self.dfscaled.columns])
        self.dfscaled[self.dfscaled.columns] = self.scaler.transform(self.dfscaled[self.dfscaled.columns]) 
        return
    
    def reverse_transform(self):
        self.dfscaled[self.dfscaled.columns] = self.scaler.inverse_transform(self.dfscaled[self.dfscaled.columns])
        return 



    

