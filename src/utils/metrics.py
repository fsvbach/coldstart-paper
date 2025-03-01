import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors


columns = ['Dataset', 'Datatype', 'Train Set', 'Train Sparsity', 'Evaluation Set', 'Evaluation Sparsity', 'Embedding Method', 'Prediction Method', 'Task', 'Accuracy', 'RMSE']

def add_to_metrics(metrics, result, names):
    fit_acc, fit_mse, acc_imp, mse_imp, acc, mse = result
    metrics.append(names+['Fit', fit_acc, fit_mse])
    metrics.append(names+['Impute', acc_imp, mse_imp])
    metrics.append(names+['Overall', acc, mse])


def compute_metric(predictions, ground_truth, test_mask=None, prefix='', silent=False):
    assert ~np.any(np.isnan(ground_truth)), 'Ground truth contains np.NaN'
    if np.any(np.isnan(predictions)):
        print('Predictions contain np.NaN')
        predictions = predictions.fillna(0.5)
    
    fit_acc, fit_mse, gen_acc, gen_mse = [np.nan]*4

    if test_mask is None:
        test_mask = np.zeros_like(predictions).astype(bool)
    
    if not np.all(test_mask):
        fit_acc = 1 - np.mean(np.abs((np.round(ground_truth) - np.round(predictions))[~test_mask]))
        fit_mse = np.sqrt(np.mean(np.square((ground_truth - predictions))[~test_mask]))
    if np.any(test_mask):
        gen_acc = 1 - np.mean(np.abs((np.round(ground_truth) - np.round(predictions))[test_mask]))
        gen_mse = np.sqrt(np.mean(np.square((ground_truth - predictions))[test_mask]))

    if not silent:
        print(f'{prefix} Fit Accuracy: {fit_acc}')
        print(f'{prefix} Fit RMSE: {fit_mse}')
        print(f'{prefix} Imputation Accuracy: {gen_acc}')
        print(f'{prefix} Imputation RMSE: {gen_mse}', end='\n\n')

    test_size = np.mean(test_mask)
    acc = np.nansum([fit_acc*(1-test_size), gen_acc*test_size])
    mse = np.nansum([fit_mse*(1-test_size), gen_mse*test_size])
    return fit_acc, fit_mse, gen_acc, gen_mse, acc, mse


class NearestCandidates:
    def __init__(self, train_set, k=10, seed=None):
        """
        Initializes the Recommender with a training set.
        :param train_set: A DataFrame containing the training data.
        """
        self.RNG = np.random.default_rng(seed=seed)
        self.train_set = train_set
        self.k = k
        self.trees = {}  # Stores trees with keys as tuple of questions (columns)

    def cacheTree(self, questions):
        """
        Computes and stores a KD tree for the specified questions (columns).
        :param questions: A list of questions (column names) to build the KD tree with.
        """
        key = tuple(questions)
        if key not in self.trees:
            # Extract the relevant columns from the training set
            data = self.train_set.loc[:,questions].values
            # Create and fit the NearestNeighbors (KD tree)
            tree = NearestNeighbors(n_neighbors=self.k, p=1).fit(data)
            # Store the tree
            self.trees[key] = tree
        return self.trees[key]
    
    def recommend(self, given_answers: pd.Series):
        """
        Returns the nearest neighbors in the train set, using only the specified questions in given_answers.
        :param given_answers: A pd.Series where indices are question names and values are the user's answers.
        """
        given_answers = given_answers.dropna().sort_index()
        if len(given_answers):
            tree = self.cacheTree(given_answers.index)
            _, indices = tree.kneighbors(given_answers.values.reshape(1,-1))
            return self.train_set.iloc[indices[0]].index.values
        return self.train_set.index.to_series().sample(frac=1).iloc[:self.k].values

    def probabilistic(self, pseudo_answers: pd.DataFrame, n_samples=10):
        tree = self.cacheTree(pseudo_answers.columns)
        _, numpy_means = tree.kneighbors(pseudo_answers.values)
        probabilities = np.vstack([pseudo_answers]*n_samples)
        samples = self.RNG.binomial(1, probabilities)
        sample_means = np.vstack([numpy_means]*n_samples)
        _, sample_stds = tree.kneighbors(samples)
        index = np.hstack([pseudo_answers.index]*n_samples)
        stds = pd.DataFrame(np.hstack([sample_means,sample_stds]), index=index
                            ).apply(lambda row: overlap(row.iloc[:self.k], row.iloc[self.k:]), axis=1
                                    ).groupby(level=0).mean()
        means = pd.DataFrame(numpy_means, index=pseudo_answers.index
                             ).apply(lambda row: row.tolist(), axis=1)
        result = pd.concat([means, stds], axis=1)
        result.columns = ['recommendation', 'variance']
        return result
    
def overlap(list1, list2):
    set1, set2 = set(list1), set(list2)
    return len(set1.intersection(set2)) / len(set1)

