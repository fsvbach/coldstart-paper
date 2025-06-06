import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

import sys
sys.path.insert(0, '../')

from utils.visualization import get_meshgrid
from utils.preprocession import impute_dataframe
from utils.logger import logger

generator = np.random.Generator(np.random.PCG64(seed=0))

def idx2coordinates(maxidxs, area, r):
    X = get_meshgrid(area, d=0, r=r)
    return X[maxidxs]

def posteriors2coordinates(posteriors, area, r):
    maxidxs = np.argmax(posteriors.reshape(-1,r*r), axis=1)
    return idx2coordinates(maxidxs, area=area, r=r)

def modelParams(model):
    return np.concatenate((model.coef_, model.intercept_.reshape(1, 1)), axis=1) 

def addOnes(array):
    return np.hstack((array, np.ones((len(array),1))))

def binarize(array):
    random_matrix = generator.random(array.shape)
    return (random_matrix <= array).astype(int)

class RandomModel:
    def __init__(self):
        self.coef_      = generator.standard_normal(2).reshape((1, 2))
        self.intercept_ = generator.standard_normal(1)

    def predict(self, X):
        logits = np.dot(X, self.coef_.T) + self.intercept_
        # logits = np.dot(addOnes(X), self.coef_.T) 
        return (logits > 0).astype(int).flatten()

def train_LR_models(embedding, reactions):
    models = {}
    for task in reactions.columns:
        model = LogisticRegression(random_state=0)
        mask = ~reactions.loc[:, task].isna()
        train_data   = embedding[mask]
        train_labels = binarize(reactions.loc[mask, task])
        if len(train_labels.unique()) == 2:
            model.fit(train_data, train_labels)
        else: 
            model = RandomModel()
            logger.warning(f"No model fitted for Feature {task}.")
        models[task] = model
    return models

class XPLORE:
    def __init__(self, 
                 reactions, 
                 prior_mean = np.array([0,0]), 
                 prior_cov  = np.array([[.1, 0], [0, .1]]), 
                 xmin=-1, xmax=1, resolution=200):
        
        # Parameters
        self.xmin = xmin
        self.xmax = xmax
        self.r = resolution
        self.area = (xmin, xmax, xmin, xmax)

        # Creating the X-Space
        self.X = get_meshgrid(area=(xmin, xmax, xmin, xmax), d=0, r=resolution)
        self.prior_mean = prior_mean
        self.prior_cov  = prior_cov
        self.prior   = multivariate_normal(prior_mean, prior_cov)
        self.prior_X = self.prior.pdf(self.X) / self.prior.pdf(self.X).sum()

        self.fit_index     = reactions.index.to_series()
        self.fit_reactions = reactions
        self.N = len(self.fit_index)

        # Initialize with PCA
        if len(reactions.index) > 1:
            imp_reactions = impute_dataframe(self.fit_reactions)
            self.embedding = pd.DataFrame(PCA(n_components=2).fit_transform(imp_reactions), 
                                        index=imp_reactions.index, columns=['x','y'])
            centroid = (self.embedding[['x', 'y']].max() + self.embedding[['x', 'y']].min())/2
            self.embedding -= centroid
            max_extent = np.abs(self.embedding[['x', 'y']]).max()*1.05
            self.embedding /= max_extent
            self.models = train_LR_models(self.embedding.values, self.fit_reactions)
        else:
            logger.info('Empty Initialization: No models fitted at all.')
            self.embedding = pd.DataFrame([], columns=['x','y'])
            self.models = {task: RandomModel() for task in reactions.columns}

        logger.info(f'Initialized model with {self.N} users.')
        self.updateLikelihoods()

    def __str__(self) -> str:
        return 'XPLORE'
    
    ## This stores the predictions within in X-space for all questions
    def updateLikelihoods(self):
        # Concatenate intercept and coefficients along the second axis
        items = [modelParams(model) for model in self.models.values()]
        self.items = pd.DataFrame(np.vstack(items), columns=['beta1', 'beta2', 'alpha'], index=self.models.keys())
        self.likelihood_X = pd.DataFrame(self.predict(self.X), columns=self.models.keys())

    ## For values outside the X-space we can use self.predict
    def predict(self, params, queries=None):
        if queries is None:
            queries = self.items.index
        if not len(params):
            return np.array([])
        params = addOnes(params.reshape(-1,2))
        return expit(params@self.items.loc[queries,:].values.T)

    ## This computes the posterior for a single user with given answers in any space
    def posterior(self, params, answers):
        likelihood = self.predict(params.reshape(-1, 2), answers.index)
        likelihood = np.prod(1 - np.abs( answers.values - likelihood),axis=1)
        likelihood = likelihood * self.prior.pdf(params.reshape(-1, 2))
        return likelihood/likelihood.sum()
    
     ## This computes the posterior for a single user with given answers on the X-space
    def posterior_X(self, answers):
        answers = answers.loc[~answers.isna()]
        likelihood = self.likelihood_X.loc[:,answers.index].values
        likelihood = np.prod(1 - np.abs( answers.values - likelihood),axis=1)
        likelihood = likelihood * self.prior_X
        return likelihood/likelihood.sum()

    ## This computes posteriors on X for every user in train set
    def computePosteriors(self):
        posteriors = []
        for n in self.fit_index.index:
            user = self.fit_reactions.loc[n]
            answers = user.loc[~user.isna()]
            posteriors.append(self.posterior_X(answers))
        self.fit_posteriors = pd.DataFrame(posteriors, index=self.fit_index.index)

        # store new ideal points
        idealpts = posteriors2coordinates(self.fit_posteriors.values, area=self.area, r=self.r)
        self.embedding = pd.DataFrame(idealpts,index=self.embedding.index, columns=['x','y'])

    ## This computes the maximum likehood estimator for a single user with given answers
    ## Relevant for new users, the other are stored already in self.posteriors or self.embedding
    def embed(self, answers):
        return posteriors2coordinates(self.posterior_X(answers), area=self.area, r=self.r)[0]
    
    ## This computes the predictions for a single user with given answers (in Binary case!)
    def impute(self, answers):
        P_X_Yi  = self.posterior_X(answers).reshape(-1,1) 
        P_Yn1_X = self.likelihood_X
        P_XYn1_Yi = P_Yn1_X * P_X_Yi ## (40000, 45)
        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0) ## (45,)
        return pd.Series(P_Yn1_Yi, index=answers.index, name=answers.name)
    
    ## Evaluate the model fit (in binary case!)
    def evaluate(self):
        assert len(self.fit_index), 'No fit users given.'
        fit_predictions = self.predict(self.embedding.values)
        self.fit_predictions = pd.DataFrame(fit_predictions, index=self.fit_reactions.index, columns=self.fit_reactions.columns)
        self.fit_accuracy = 1 - np.abs(self.fit_reactions - self.fit_predictions.round()).mean().mean()
        self.fit_rmse = np.sqrt(np.mean(np.square((self.fit_reactions - self.fit_predictions))))
        return self.fit_rmse, self.fit_accuracy
    
    def upgrade(self, reactions):
        return XPLORE(reactions,
                      prior_mean = self.prior_mean, 
                      prior_cov  = self.prior_cov, 
                      xmin=self.xmin, xmax=self.xmax, resolution=self.r)


