from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .metrics import NearestCandidates, overlap
from .logger import logger

def make_hashable(series):
    sorted_series = series.sort_index()
    return tuple(sorted_series.index), tuple(sorted_series.values)


class CacheDecorator:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.hit_count = {}
    
    def __call__(self, *args, **kwargs):
        cache_key = make_hashable(*args)
        if cache_key not in self.cache:
            result = self.func(*args, **kwargs)
            self.cache[cache_key] = result
            self.hit_count[cache_key] = 0
        else:
            result = self.cache[cache_key].copy()
            self.hit_count[cache_key] += 1
        
        return result
    
    def get_cache_data(self):
        """Return cache and hit count data for export."""
        return pd.DataFrame([(key, self.hit_count[key], self.cache[key]) for key in self.cache], 
                              columns=['Cache Key', 'Hits', 'Output'])

class SelectionMethod(ABC):
    def __init__(self, model=None, use_cache=False, name='SelectionMethod'):
        self.model = model
        self.name = name
        self.use_cache = use_cache
        if self.use_cache:
            self.compute_weights = CacheDecorator(self.compute_weights)
        
    def __str__(self):
        return self.name

    def evaluate(self, answers, **kwargs):
        assert isinstance(answers, pd.Series), "Answers must be a pandas Series"
        # Compute weights based on the data and fixed_order
        self.user = answers
        self.open_index  = answers.loc[answers.isna()].index
        given_answers = answers.loc[~answers.isna()]
        weights = self.compute_weights(given_answers, **kwargs)
        return pd.Series(weights, index=self.open_index, name=answers.name)

    @abstractmethod
    def compute_weights(self, given_answers, **kwargs):
        pass

class FixedOrder(SelectionMethod):
    def __init__(self, fixed_order, use_cache=True, model=None, name='FixedOrder'):
        super().__init__(use_cache=use_cache, model=model, name=name)
        assert isinstance(fixed_order, list), "fixed_order must be a list"
        self.fixed_order = dict(zip(fixed_order, range(len(fixed_order), 0, -1)))

    def compute_weights(self, given_answers, **kwargs):
        return self.open_index.map(self.fixed_order).values

class RandomSelection(SelectionMethod):
    def __init__(self, use_cache=False, model=None):
        super().__init__(use_cache=use_cache, model=model)
        self.RNG = np.random.default_rng(seed=0)

    def __str__(self):
        return 'RandomOrder'
    
    def compute_weights(self, given_answers=None, **kwargs):
        return self.RNG.random(len(self.open_index))

class Uncertainty(SelectionMethod):
    def __str__(self):
        return 'Uncertainty'
    
    def compute_weights(self, given_answers): 
        P_Yn1_X = self.model.likelihood_X.loc[:, self.open_index].values
        P_X_Yi = self.model.posterior_X(given_answers).reshape(-1,1) ## (40000,1)
        P_XYn1_Yi = P_Yn1_X * P_X_Yi ## (40000, 45)
        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0) ## (45,)
        return 2*P_Yn1_Yi*(1-P_Yn1_Yi)
    
class PosteriorRMSE(SelectionMethod):
    def __str__(self):
        return 'PosteriorRMSE'

    def compute_weights(self, given_answers): 
        P_Yn1_X = self.model.likelihood_X.loc[:,self.open_index].values ## (40000, 45)
        P_Yn0_X = 1 - P_Yn1_X
        P_X_Yi = self.model.posterior_X(given_answers).reshape(-1,1) ## (40000,1)

        P_XYn0_Yi = P_Yn0_X * P_X_Yi ## (40000, 45)
        P_XYn1_Yi = P_Yn1_X * P_X_Yi ## (40000, 45)
        assert np.allclose((P_XYn0_Yi+P_XYn1_Yi).sum(axis=0), 1, rtol=1e-5), "X Posteriors don't sum to 1"

        P_Yn1_Yi  = P_XYn1_Yi.sum(axis=0) ## (45,)
        # P_Yn0_Yi  = P_XYn0_Yi.sum(axis=0) ## (45,)
        # assert np.all(np.isclose(P_Yn1_Yi+P_Yn0_Yi, 1, rtol=1e-5)), "Y Marginals don't sum to 1"

        P_X_Yin1 = np.ones_like(P_XYn1_Yi, dtype=np.float64)/P_XYn1_Yi.shape[0]  ## (40000, 45) with uniform distribution
        mask = np.where(P_XYn1_Yi.sum(axis=0)!=0)
        P_X_Yin1[:,mask]=P_XYn1_Yi[:,mask]/P_XYn1_Yi[:,mask].sum(axis=0)

        P_X_Yin0 = np.ones_like(P_XYn0_Yi, dtype=np.float64)/P_XYn0_Yi.shape[0]  ## (40000, 45) with uniform distribution
        mask = np.where(P_XYn0_Yi.sum(axis=0)!=0)
        P_X_Yin0[:,mask]=P_XYn0_Yi[:,mask]/P_XYn0_Yi[:,mask].sum(axis=0)
        # P_X_Yin1  = P_XYn1_Yi / P_Yn1_Yi ## (40000, 45)
        # P_X_Yin0  = P_XYn0_Yi / P_Yn0_Yi ## (40000, 45)
        assert np.all(np.isclose(P_X_Yin1.sum(axis=0), 1, rtol=1e-5)), "X Posterior for Yn=1 don't sum to 1"
        assert np.all(np.isclose(P_X_Yin0.sum(axis=0), 1, rtol=1e-5)), "X Posterior for Yn=0 don't sum to 1"

        P_Yn1_Yin1 = np.tensordot(P_Yn1_X, P_X_Yin1, axes=[0,0]) ## (45,45)
        P_Yn1_Yin0 = np.tensordot(P_Yn1_X, P_X_Yin0, axes=[0,0]) ## (45,45)
        # P_Yn0_Yin1 = np.tensordot(P_Yn0_X, P_X_Yin1, axes=[0,0]) ## (45,45)
        # P_Yn0_Yin0 = np.tensordot(P_Yn0_X, P_X_Yin0, axes=[0,0]) ## (45,45)
        # assert np.allclose(P_Yn1_Yin1 + P_Yn0_Yin1, 1, rtol=1e-5)
        # assert np.allclose(P_Yn1_Yin0 + P_Yn0_Yin0, 1, rtol=1e-5)
        np.fill_diagonal(P_Yn1_Yin1, 1)
        np.fill_diagonal(P_Yn1_Yin0, 0)

        ### PRIOR GINI (FOR ALL REMAINING QUESTIONS)
        prior_gini = 2 * (P_Yn1_Yi * (1-P_Yn1_Yi))
        ### POSTERIOR GINI (If Yn=0) 
        gini_Yn0 = (2 * (1-P_Yn1_Yin0) * P_Yn1_Yin0).sum(axis=0)
        ### POSTERIOR GINI (If Yn=1) 
        gini_Yn1 = (2 * (1-P_Yn1_Yin1) * P_Yn1_Yin1).sum(axis=0)

        posterior_gini_sum = P_Yn1_Yi*gini_Yn1 + (1-P_Yn1_Yi)*gini_Yn0
        posterior_rmse_loss = prior_gini.sum() - posterior_gini_sum
        return posterior_rmse_loss
    


class ColdStartSimulation:
    def __init__(self, method, truth, candidates, k_neighbors=36, update_rate=5, number_queries=10, forget=0, forget_step=5):
        self.method = method
        self.truth  = truth

        self.reactions = pd.DataFrame([], columns=self.truth.columns, index=self.truth.index, dtype=np.float64)
        self.update_rate = update_rate
        self.number_queries = number_queries
        self.forget = forget
        self.forget_step = forget_step
        self.forgotten = 0

        self.k_neighbors = k_neighbors
        self.candidates = NearestCandidates(candidates, k=k_neighbors)
        self.true_neighbors = self.truth.apply(self.candidates.recommend, axis=1)

        self.results = {}

    def run(self, verbose=1):
        for i, n in enumerate(self.reactions.index):
            if i%self.update_rate == 0 and i>0:
                batch_data = self.reactions.iloc[i-self.update_rate:i]
                logger.debug("Upgrade with batch sparsity: {:.2f}".format(batch_data.isna().mean().mean()))
                new_reactions = pd.concat([self.method.model.fit_reactions, batch_data])
                if self.forgotten + self.forget_step <= self.forget:
                    self.forgotten += self.forget_step
                    new_reactions = new_reactions.iloc[self.forget_step:]                    
                    logger.debug("Now forgotten {:.2f} data points. {:.2f} remaining.".format(self.forgotten, len(new_reactions)))
                self.method.model = self.method.model.upgrade(new_reactions)
                # print('model updated')
            # print(self.reactions.isna().sum().sum())
            for k in range(self.number_queries):
                user = self.reactions.loc[n]
                result = self.method.evaluate(user)
                query = result.idxmax()
                self.reactions.loc[n, query] = self.truth.loc[n,query]                
            
            predictions = self.method.model.impute(user)
            asked = ~self.reactions.loc[n].isna()
            # return predictions, asked
        
            RMSE = np.sqrt(np.mean(np.square((self.truth.loc[n] - predictions))[~asked]))
            predictions.loc[asked] = self.reactions.loc[n, asked]

            ### Recompute Nearest Candidates 
            neighbors = self.candidates.recommend(predictions)
            CRA = overlap(self.true_neighbors.loc[n], neighbors)

            self.results[i] = {'User': n,
                               'RMSE': round(RMSE, 3), 
                               'CRA':round(CRA, 3), 
                               'ModelVersion':(i//self.update_rate)*self.update_rate, 
                               'TimeStamp': pd.Timestamp.now(),
                               'Queries': predictions.loc[asked].index.to_list(),
                               'Neighbors': list(neighbors),
                               'Predictions': predictions.round(2).to_list()}

            if verbose:
                if i % verbose == 0:
                    logger.info("User {}: {} gives {:.2f} RMSE and {:.2f}% CRA.".format(i,n,RMSE,CRA))

    def save_results(self, filename='', suffix=''):
        results = pd.DataFrame.from_dict(self.results, orient='index')
        if filename:
            results.to_csv(f'../results/aqvaa/{filename}_{self.method}_{self.number_queries}Q_{self.method.model}_{self.update_rate}U_{len(self.results)}V{suffix}.csv')
        return results