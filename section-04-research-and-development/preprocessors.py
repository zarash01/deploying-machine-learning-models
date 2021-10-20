import numpy as np
import pandas as pd

# BaseEstimator is to define the parameters,
# TransformerMixin to inherit the fit.transform functionality 
from sklearn.base import BaseEstimator, TransformerMixin


# create a class to inherit the methods and attributes from 
# scikit-learn objects BaseEstimator and TransformerMixin
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    # create a method with the parameters this class will take
    # when it is initialised
    def __init__(self, variables, reference_variable):
        
        # error handling
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        # initialise fit method. We need this step to fit
        # the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        
        # create the new feature which is the time elapsed (in years),
        # overwriting the original feature
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X



# categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        # if not isinstance(mappings, dict):
        #     raise ValueError('mappings should be a dictionary {A:1, B:2...}')

        # assign the variables to the class
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            # replace the values according to the mapping
            X[feature] = X[feature].map(self.mappings)

        return X