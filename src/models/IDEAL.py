import matplotlib as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.spatial import distance
from scipy.optimize import minimize

def get_meshgrid(area=(-1,1,-1,1), d=0, r=100):
    # Create a grid of points to evaluate
    x_min, x_max, y_min, y_max = area
    xx, yy = np.meshgrid(np.linspace(x_min - d, x_max + d, r),
                         np.linspace(y_min - d, y_max + d, r))
    return np.c_[xx.ravel(), yy.ravel()]

class IDEAL:
    def __init__(self, betabar, xbar, index=None, columns=None):
        if index is None:
            index = xbar.index
        if columns is not None:
            betabar.index = columns
        self.train_embedding = pd.DataFrame(xbar.values, index=index, columns=['x','y'])
        self.items = betabar
    
    def __str__(self) -> str:
        return 'IDEAL'
    
    def make_probabilistic(self, mean=np.array([0,0]), cov=np.array([[1, 0], [0, 1]]), d=2, r=200):
        self.grid_params = d,r
        self.X = get_meshgrid(d=d, r=r)
        self.likelihood_X = pd.DataFrame(self.predict(self.X), columns = self.items.index)
        self.prior   = multivariate_normal(mean, cov)
        self.prior_X = self.prior.pdf(self.X) / self.prior.pdf(self.X).sum()

    def posterior_X(self, answers):
        likelihood = self.likelihood_X.loc[:, answers.index].values 
        likelihood = np.prod(np.abs(( answers.values - 1 + likelihood)),axis=1)
        likelihood = (likelihood * self.prior_X).reshape(-1,1)
        return likelihood/likelihood.sum()

    def posterior(self, params, answers):
        likelihood = self.predict(params.reshape(-1, 2), answers.index)
        likelihood = np.prod(np.abs(( answers.values - 1 + likelihood)),axis=1)
        likelihood = likelihood * self.prior.pdf(params.reshape(-1, 2))
        return likelihood/likelihood.sum()
    
    ## user is a pd.Series or row of pd.DataFrame which contains the (sparse) reactions 
    def predict_user(self, user): 
        given_answers = user.loc[~user.isna()]
        open_answers = user.loc[user.isna()]
        P_Yn1_X = self.likelihood_X.loc[:, open_answers.index].values ## (40000, 45)
        P_X_Yi = self.posterior_X(given_answers).reshape(-1,1) ## (40000,1)
        predictions = user.copy()
        predictions.loc[open_answers.index] = (P_Yn1_X * P_X_Yi).sum(axis=0)
        return predictions
    
    def predict(self, params, queries=None):
        if queries is None:
            queries = self.items.index
        params = params.reshape(-1,2)
        params = np.concatenate([params, -np.ones(len(params)).reshape(-1,1)], axis=1)
        abilities = params@self.items.loc[queries,:].values.T
        return norm.cdf(abilities)

    # take transformed coordinates directly (that's why for border we divide by weights)
    def objective(self, params, answers, regularize=True):
        probs = self.predict(params.reshape(-1, 2), answers.index)
        loss =  np.nansum((np.square( answers.values - probs)),axis=1)
        return loss + (0.5 * np.linalg.norm(params.reshape(-1, 2), axis=1) if regularize else 0)


    def encode(self, row):
        answers = row.loc[~row.isna()]
        res = minimize(self.objective, np.zeros(2), args=(answers,), method='BFGS')
        return res.x[0], res.x[1], res.fun