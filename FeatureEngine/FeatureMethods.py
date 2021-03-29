from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from sklearn import svm #Import Support Vector Machine model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import itertools
import scipy.stats as ss
from xgboost.sklearn import XGBClassifier
from AutoEncoderTrain.autoencoder import encoder, decoder, FitAutoEncoder
from torch import autograd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt
from sklearn.feature_selection import RFECV


pathBase = str(Path.cwd())


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

#Webapp for predictions on model?

#Add a final method that removes the ORIGINAL (unformated features) and returns a numpy array which can
#be passed to an ML model 

#Evaluate if it makes sense to implement HDF5 format

#Add a method for group labeling that takes two paramaters: 
#(1) - array of breakouts [1-100, 101-200, etc] 
#(2) - field name

#Workflow:
#1 initialize FunctionFeaturizer with df and transform method

#Add a method for feature hasing

#to add some kind of pickling object or pickling method so that the objects can be carried over for production. Untill we do that
#we can only use these methods for testing.

#for datetime feature need to transform day of week/month into a
# a transformation based off of cos and sin function mapped to a circle, where cos = x val, and sin = y val.
# http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
# https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,dateString,datetype):
        self.datetype = datetype
        self.dateString = dateString
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.datetype == 'Hours':
            hour_sin = np.sin(x[self.dateString].dt.hour*(2.*np.pi/24))
            hour_cos = np.cos(x[self.dateString].dt.hour*(2.*np.pi/24))
            return np.concat([day_sin, day_cos], axis=1)
        #Days transform
        if self.datetype == 'Days':
            day_sin = np.sin(x[self.dateString].dt.dayofweek*(2.*np.pi/7))
            day_cos = np.cos(x[self.dateString].dt.dayofweek*(2.*np.pi/7))
            return pd.concat([day_sin, day_cos], axis=1)
        #Weeks transform
        if self.datetype == 'Weeks':
            day_sin = np.sin(x[self.dateString].dt.week*(2.*np.pi/52))
            day_cos = np.cos(x[self.dateString].dt.week*(2.*np.pi/52))
            return pd.concat([day_sin, day_cos], axis=1)    
        #Months transform
        if self.datetype == 'Months':
            month_sin = np.sin(x[self.dateString].dt.month-1*(2.*np.pi/12))
            month_cos = np.cos(x[self.dateString].dt.month-1*(2.*np.pi/12))
            return pd.concat([month_sin, month_cos], axis=1)
        #Years thansform
        if self.datetype == 'Years':
            year = x[self.dateString].dt.year.to_frame()
            return year
        if self.datetype == 'WeekendBinary':
            dfWkday = pd.DataFrame({'wkday':x[self.dateString].dt.dayofweek})
            dfWkday.loc[(dfWkday.wkday == 5) | (dfWkday.wkday == 6), 'wkdayBinary'] = 1
            dfWkday.loc[(dfWkday.wkday != 5) | (dfWkday.wkday != 6), 'wkdayBinary'] = 0
            return dfWkday[['wkdayBinary']]
        
        if self.datetype == 'Days-INT':
            return x[self.dateString].dt.dayofweek.to_frame()

        if self.datetype == 'Weeks-INT':
            return x[self.dateString].dt.week.to_frame()

        if self.datetype == 'Months-INT':
            return x[self.dateString].dt.month.to_frame()

        #Utilized to represent linear movement in year (i.e. 2020 and 1/12, 2020 and 2/12, etc)
        if self.datetype == 'Years-Months':
            yearsmonths = x[self.dateString].dt.year + x[self.dateString].dt.month/12      
            return yearsmonths
    
        if self.datetype == 'Months-Years':
            years = x[self.dateString].dt.year
            months = x[self.dateString].dt.month
            dfMonthsYears = pd.DataFrame({'Years':years, 'Months':months})
            dfMonthsYears['yearrank'] = dfMonthsYears.Years.rank(method='dense')
            dfMonthsYears['yearrank'] = dfMonthsYears['yearrank'] - 1 #convert to 0 based rank
            dfMonthsYears['AbsoluteYearScale'] = dfMonthsYears['yearrank']*12 + dfMonthsYears['Months']
            dfMonthsYears['RelativeYearScale'] = dfMonthsYears['AbsoluteYearScale']/12
            return dfMonthsYears[['RelativeYearScale']] 



class FunctionFeaturizer:

    def __init__(self, df):
        self.df = df
        self.dfscaled = None #val gets set if outputFeatures method is called
        self.originalFeatures = df.columns.tolist()
        self.labelEncodeObj = LabelEncoder()
        self.transformDict = {} #holds transformation objects to save for later use


    def concatVector(self, textSeries, ngramMin = 1, ngramMax = 1, min_df=0.10, max_df=0.90):
        '''
        Returns a matrix of token counts.
        Required param textSeries takes a string of the textfield name from df
        Second and third paramaters (ngramMin and ngramMax) are used to define
        features of unigram, bigrams, both, etc
        '''
        #scikit learn CountVertorizer object
        #Note: to consider permutating certain paramaters in the CV object
        cv = CountVectorizer(ngram_range=(ngramMin, ngramMax),stop_words='english', min_df=0.01, max_df=0.99)
        text_features = cv.fit_transform(self.df[textSeries].values)
        vect_df = pd.DataFrame(text_features.todense(), columns=cv.get_feature_names())
        self.df = self.df.reset_index()
        dfConcat = pd.concat([self.df, vect_df], axis=1)
        self.df = dfConcat
        return vect_df
    
    def encodeLabels(self, labelSeries):
        '''
        Returns labels encoded as integers
        Required param labelSeries takes a string of the label field name from df.
        Note: utilize label encoded for encoding where order does not matter.
        Note: method only takes one column at a time
        '''
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
        self.df.reset_index()
        self.df = pd.concat([self.df, pd.get_dummies(self.df[labelSeries])], axis=1)
        self.df.reset_index()
        #self.df = pd.get_dummies(self.df, columns=labelSeries, prefix="1h_")
        return
    
    #To add an optional param that will allow you to keep certain original features
    def outputFeatures(self):
        '''
        Returns only transformed features and excludes original untransformed features.
        Can be used to quickly obtain all engineered features.
        '''
        self.dfscaled = self.df[self.df.columns.difference(self.originalFeatures)]
        #outputDf = self.df[self.df.columns.difference(self.originalFeatures)]
        #return outputDf        
        return
    
    def transformDateTime(self, LabelSeries, type):
        '''
        Returns pairs of sin and cos transformed values from a given datetime series field.
        The field must be passed as a datetime
        A pair of sin/cos features is created for each of year/month/day
        Type paramater takes either "hours", "days", or "months".
        Note that this feature will not work well with decision tree type of models as 
        these models do not consider cross-feature, and consider one feature at a time.
        '''
        #Hours transform
        if type == 'Hours':
            self.df['hour_sin'] = np.sin(self.df[LabelSeries].dt.hour*(2.*np.pi/24))
            self.df['hour_cos'] = np.cos(self.df[LabelSeries].dt.hour*(2.*np.pi/24))

        #Days transform
        if type == 'Days':    
            self.df['day_sin'] = np.sin(self.df[LabelSeries].dt.dayofweek*(2.*np.pi/7))
            self.df['day_cos'] = np.cos(self.df[LabelSeries].dt.dayofweek*(2.*np.pi/7))


        #Months transform
        if type == 'months':
            self.df['month_sin'] = np.sin(self.df[LabelSeries].dt.month*(2.*np.pi/12))
            self.df['month_cos'] = np.cos(self.df[LabelSeries].dt.month*(2.*np.pi/12))            
        
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


#Class to process our various models, and score on metrics
#Pass a feature object with methods?
#Can we utilize inheritance to avoid duplication of methods between supervised and unsupervised analyzers?
class Supervised_Analyzer:

    def __init__(self,df):
        self.df = df
        self.models = dict()
    
    def __filterFieldDict(self, dictObj, callback):
        # Iterate over all the items in dictionary
        fieldList = []
        for (key, value) in dictObj.items():
            # Check if item satisfies the given condition then add to new dict
            if callback((key, value)):
                fieldList.append(key)
        return fieldList

    def set_features(self, features):
        'Sets the features var as a class property'
        self.featureDict = features
        self.features = self.__filterFieldDict(self.featureDict, lambda elem: elem[1] != 'ALL')
        self.oneHotFeatures = self.__filterFieldDict(self.featureDict, lambda elem: elem[1] == '1hot')
        self.passThroughFeatures = self.__filterFieldDict(self.featureDict, lambda elem: elem[1] == 'Passthrough_Transform')
        return self #self for method chaining

    def set_traintestsplit(self, target_field, test_split_size, modelTraining=True, imbalancedSampling=False):
        '''
        Generates input and target array with a given target field name as string, and split size as float <1.
        imbalancedSampling var used to hanlde imbalnced classifications
        '''
        #x, y and indices used to be to_numpy but now changed to be a df
        x = self.df[self.features].loc[:, self.df[self.features].columns != target_field]
        y = self.df[target_field]
        indices = self.df.index

        #https://stackoverflow.com/questions/35622396/getting-indices-while-using-train-test-split-in-scikit
        #Due to shuffling Will need to retain index to reappend predictions to original test df 

        if modelTraining == True:
            x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, indices, test_size=test_split_size, random_state=42)

            self.df_test_y = y_test #need to rework this to be less costly        
            self.df_test_x = x_test #need to rework this to be less costly
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_train_imbalanced = y_train #need to rework this to be less costly
            self.y_test = y_test
            self.target_field = target_field
            self.indices_train = indices_train
            self.indices_test = indices_test
        else:
            self.df_test_y = y #need to rework this to be less costly        
            self.df_test_x = x #need to rework this to be less costly
            self.x_train = x
            self.x_test = x
            self.y_train = y

        return self

    def GetColumnTransformer(self):
        '''
        Method to set a column transformer as a class property.
        Can be leveraged in single trained models for transforming features in future iterations.
        '''
        #Terriblly exepensive method. This should instead be a getter @property method, and should set property on transformFeatures method
        stop_words = [bytes(x.strip(), 'utf-8') for x in open(pathBase + "\Datasets\stopwords.txt",'r').read().split('\n')]
        #To consider dropping one of the ACCNT_GRPV labels as they are perfectly coorelated (i.e. text description vs. a code)
        oneHotFeatures = self.oneHotFeatures
                
        ct_x = ColumnTransformer(transformers=[
            ('1hot', OneHotEncoder(handle_unknown='ignore',sparse=False), oneHotFeatures)
            #('textvector', TfidfVectorizer(stop_words = stop_words, token_pattern='[^\s]*(hosp|medic|health|poly|farma|pharma|dent|clinic|nutri|pedia|drug|lab|neuro|dr |dr.|nurse|doctor|physic|practic|state |federal |govt |government|dept |department|county)[^\s]*', max_features=30),'textappend'),
            #('Days', DateTransformer('DOC_DATE', 'Days-INT'), ['DOC_DATE']),
            #('Months', DateTransformer('DOC_DATE', 'Months-INT'), ['DOC_DATE']),
            #('Weeks', DateTransformer('DOC_DATE', 'Weeks-INT'), ['DOC_DATE']),
            #('WKEnds', DateTransformer('DOC_DATE', 'WeekendBinary'), ['DOC_DATE']),
            ], 
            remainder='passthrough')
        
        ct_fit_x = ct_x.fit(self.x_train)

        self.x_transformer = ct_fit_x
        return self

    def transformFeatures(self):
        '''
        Transforms df_test/df_train and sets transformed train and test properties
        '''
        stop_words = [bytes(x.strip(), 'utf-8') for x in open(pathBase + "\Datasets\stopwords.txt",'r').read().split('\n')]
        oneHotFeatures = self.oneHotFeatures

        ct_x = ColumnTransformer(transformers=[
            ('1hot', OneHotEncoder(handle_unknown='ignore',sparse=False), oneHotFeatures)
            #('textvector', TfidfVectorizer(stop_words = stop_words, token_pattern='[^\s]*(hosp|medic|health|poly|farma|pharma|dent|clinic|nutri|pedia|drug|lab|neuro|dr|nurse|doctor|physic|practic|state |federal |govt |government|dept |Department|county)[^\s]*', max_features=30),'textappend'),
            #('Days', DateTransformer('DOC_DATE', 'Days-INT'), ['DOC_DATE']),
            #('Months', DateTransformer('DOC_DATE', 'Months-INT'), ['DOC_DATE']),
            #('Weeks', DateTransformer('DOC_DATE', 'Weeks-INT'), ['DOC_DATE']),
            #('WKEnds', DateTransformer('DOC_DATE', 'WeekendBinary'), ['DOC_DATE']),
            ], 
            remainder='passthrough')

        ct_fit_x = ct_x.fit(self.x_train)

        #Checks if a transformer already exists, which would be true if pretrained model
        if hasattr(self, 'x_transformer'):
            #transform f(x) dataframes to a working numpy format for input into models 
            x_train = self.x_transformer.transform(self.x_train)
            x_test = self.x_transformer.transform(self.x_test)

        else:
            x_train = ct_fit_x.transform(self.x_train)
            x_test = ct_fit_x.transform(self.x_test)  

        
        #transform y dataframe to a working numpy format for input into models
        #Only need y numpy for model training
        
        y_train = self.y_train.to_numpy()

        #set as properties to object
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

        return self

    @property #decorator to access method as property
    def Models(self):
        '''
        Utilized to access the models dictionary as a chained method.
        '''
        return self.models

    @property
    def numpy_features(self):
        '''
        Returns a numpy array of the provided df and features.
        Method should onyly be called after features are defined.
        set_features(['feature 1', 'feature 2']).numpy_feature
        '''
        numpyarray = self.df[self.features].to_numpy()
        return numpyarray


    def FitSVM(self):
        '''
        tuned = [
                {'kernel':['rbf'], 'gamma': ['scale'], 'C': [1]},
                {'kernel': ['linear'], 'C': [1]}
                ]
        '''
        tuned = {'kernel': ['linear'], 'C': [1]}
        #Nested cross validation
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        SVM = GridSearchCV(svm.SVC(), param_grid=tuned, cv=None, n_jobs=-1, verbose=3)
        model = SVM.fit(self.x_train, self.y_train) #self.df[self.features].to_numpy())

        ##instance of gridsearch model is returned from models dict https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        self.models['SVM'] = {
            "GSCV":SVM
            }
        

        return self #self for method chaining

    def FitKNeighbors(self):
        neighbors = np.arange(10) #array from 0 to 9

        tuned = {
                'n_neighbors':[5],
                'weights':['uniform', 'distance'], 
                'algorithm': ['auto']
                }
        #Nested cross validation
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        KN = GridSearchCV(KNeighborsClassifier(), param_grid=tuned, cv=cv, n_jobs=-1, verbose=3)
        model = KN.fit(self.x_train, self.y_train) #self.df[self.features].to_numpy())

        ##instance of gridsearch model is returned from models dict https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        self.models['KNeighbors'] = {
            "GSCV":KN
            }
        return self #self for method chaining

    def FitXGBOOST(self, splits=2):
        #imbalancedScaled = self.y_train_imbalanced.value_counts()['Non-Compliance'] / self.y_train_imbalanced.value_counts()['Compliance']
        imbalancedScaledReverse = self.y_train_imbalanced.value_counts()['Compliance'] / self.y_train_imbalanced.value_counts()['Non-Compliance']

        parameters = {
                    'booster':['gbtree'], #gblinear alternative model, and gbtree original (maybe fit linear regressor?)
                    'nthread':[-1], #when use hyperthread, xgboost may become slower
                    'objective':['binary:logistic'],
                    'learning_rate': [0.1], #so called `eta` value, want to do .1, .01, and .05
                    'max_depth': [5], #Want to do 3,5,7
                    'min_child_weight': [1],
                    'verbosity': [0],
                    'subsample': [0.9], #Want to do .8 and .9
                    'colsample_bytree': [0.9],
                    'n_estimators': [500, 1000], #Want to do 100, 250, 500, and 1000
                    'use_label_encoder':[True],
                    'early_stopping_rounds':[50],
                    'scale_pos_weight':[imbalancedScaledReverse],
                    'tree_method':['hist']
                    }
        
        cv = StratifiedKFold(n_splits=splits, random_state=42, shuffle=True)
        print("----Running grid search optimization----")
        XGB = GridSearchCV(XGBClassifier(),param_grid=parameters,cv=cv, n_jobs=1, verbose=3)
        XGB.fit(self.x_train, self.y_train)

        #**kwargs to pass list of best_params
        XGBModel = XGBClassifier(**XGB.best_params_)
        print("----Running final model fit----")
        XGB = XGBModel.fit(self.x_train, self.y_train)

        self.models['XGB'] = {
            "GSCV":XGB
            }

        return self



class Unsupervised_Analyzer:
    #Features as a class level paramater or method paramaters? To think through what makes the most sense
    def __init__(self, df):
        self.df = df
        self.models = dict()
    
    def set_features(self, features):
        'Sets the features var as a class property'
        self.features = features
        return self

    #Put autoencoder method here
    def FitAutoEncoder(self):
        #numpy array
        pathBase = str(Path.cwd())

        numpyarray = self.df[self.features].to_numpy()
        
        #Trains model for epochs designated in autoencoder.py
        FitAutoEncoder(numpyarray)
        
        #training network classes / architectures from autoencoder.py
        encoder_eval = encoder(numpyarray)
        decoder_eval = decoder(numpyarray)


        # restore pretrained model checkpoint
        #integer in name represents epoch save point, select latest.
        encoder_model_name = "ep_10_encoder_model.pth"
        decoder_model_name = "ep_10_decoder_model.pth"


        encoderSavePath = pathBase + "\\AutoEncoderDict\\" + encoder_model_name
        decoderSavePath = pathBase + "\\AutoEncoderDict\\" + decoder_model_name

        encoder_eval.load_state_dict(torch.load(encoderSavePath))
        decoder_eval.load_state_dict(torch.load(decoderSavePath))

        #Inserts trained encoding and ecoding models to the models object
        self.models['AutoEncoder'] = {
            "Encoder":encoder_eval,
            "Decoder":decoder_eval
            }

        #data = autograd.Variable(torch.from_numpy(numpyarray).float())
        
        return self

    def FitIsolationForest(self):
        '''
        Adds a grid searched IsolationForest model to models class property.
        '''
        #Isoloation forest scorer
        def scorer_f(estimator, X):   #your own scorer
            return np.mean(estimator.score_samples(X))
        #Defined permutations
        tuned = {
            'n_estimators':[100], 
            'max_samples':['auto'],
            'contamination':['auto'], 
            'max_features':[.85],
            'bootstrap':[False],
            'random_state':[None], 
            'verbose':[0], 
            'warm_start':[True], 
            'random_state':[12345]
            }  
        
        isolation_forest = GridSearchCV(IsolationForest(), tuned, scoring=scorer_f)
        model = isolation_forest.fit(self.df[self.features].to_numpy())
        
        self.models['IsolationForest'] = model
        
        return self
        #object.FitIsolationForest.models['IsolationForest'].predict('testvals')
    
    @property
    def Models(self):
        '''
        Utilized to access the models dictionary as a chained method.
        '''
        return self.models

    @property
    def numpy_features(self):
        '''
        Returns a numpy array of the provided df and features.
        Method should onyly be called after features are defined.
        set_features(['feature 1', 'feature 2']).numpy_feature
        '''
        numpyarray = self.df[self.features].to_numpy()
        return numpyarray


        


#Implementation of Principal Feature Analysis to help in identifying key features in an unsupervised learing context
#https://stats.stackexchange.com/questions/108743/methods-in-r-or-python-to-perform-feature-selection-in-unsupervised-learning?newreg=8eae58509e344faf824deb6cd0b02d0d
class PFA:
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]



#Wrapper function for implementing fuzzywuzzy match with two pandas DFs https://stackoverflow.com/questions/13636848/is-it-possible-to-do-fuzzy-match-merge-with-python-pandas
#May need to look into refactoring to speed things up, change scorer? standard strings?
#To also look into implementing RapidFuzz instead.
#Possiblly use a simpler approach to remove clear non-match records (i.e. only attempt match if at least one word in common)
def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=1):
    """
    :param df_1: the left table to join
    :param df_2: the right table to join
    :param key1: key column of the left table
    :param key2: key column of the right table
    :param threshold: how close the matches should be to return a match, based on Levenshtein distance
    :param limit: the amount of matches that will get returned, these are sorted high to low
    :return: dataframe with boths keys and matches
    """
    lookupList = df_2[key2].tolist()

    matches = df_1[key1].apply(lambda x: process.extract(x, lookupList, limit=limit))    


    df_1['matches'] = matches

    csvMatch = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    
    df_1['matches'] = csvMatch

    return df_1


def GenerateCramersVMatrix(df, categoricalFeatures):


    #Returns an iterable of combination of categorical features
    fieldPairs = list(itertools.combinations(categoricalFeatures, r=2))

    #Sets three arrays with len of combination array for input into pd.DataFrame
    name1 = np.zeros((len(fieldPairs)), dtype = object) #dtype as object replaces interger 0's as string type
    name2 = np.zeros((len(fieldPairs)), dtype = object)
    corr = np.zeros((len(fieldPairs)), dtype = object)

    
    #https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    def cramers_corrected_stat(confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

    #Populate arrays for input into pd.DataFrame
    count = 0
    for pair in fieldPairs:
        cramers = cramers_corrected_stat( pd.crosstab(df[pair[0]],df[pair[1]]))
        name1[count] = pair[0]
        name2[count] = pair[1]
        corr[count] = cramers
        count += 1


    coorDF = pd.DataFrame({"name":name1.tolist(),"name2":name2.tolist(), "coor":corr.tolist()})

    #Transforms dataframe into a coorelation matrix. Transformation code leveraged from below: 
    pivot = coorDF.pivot(*coorDF)
    coorMatrix = pivot.add(pivot.T, fill_value=0).fillna(1)

    #For single pair
    '''
    confusion_matrix = pd.crosstab(modelVals.df['HCPWord'], modelVals.df['MATL_GROUP'])
    print("HCP-MATL_GROUP: " + str(cramers_corrected_stat(confusion_matrix)))
    '''

    
    return coorMatrix