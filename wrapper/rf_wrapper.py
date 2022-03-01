# -*- coding: utf-8 -*-

"""
Created on Feb 2022
@author: Dina Berenbaum
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List


from uncertainty.resources.constants import Uncertainty
from uncertainty.utils.uncertainty_utils import calculate_entropy_uncertainties


class RandomForestClassifierWithUncertainty(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.leafs_content = None  # the split of training samples between the leafs of all decision trees within the forest
        self._labels = None
        self.used_features = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

        # produce a map of the counts of labels in each leaf
        self.leafs_content = self._binary_leaf_split_counter(X, y)
        # all possible labels:
        self._labels = list(set(y))
        # summarize the features used in the trees:
        self.used_features = self._output_used_features(X)

    def predict_with_uncertainty(self, X_test) -> (np.ndarray, List[Uncertainty]):
        predictions = self.predict(X_test)
        end_leafs = self.apply(X_test)
        uncertainties = self._extract_uncertainty_of_prediction(end_leafs, method='entropy')
        return predictions, uncertainties

    def predict_proba_with_uncertainty(self, X_test) -> (np.ndarray, List[Uncertainty]):
        predictions = self.predict_proba_1d(X_test, calibrated)
        end_leafs = self.apply(X_test)
        uncertainties = self._extract_uncertainty_of_prediction(end_leafs, method='entropy')
        return predictions, uncertainties

    def predict_proba_1d(self, x: pd.DataFrame) -> np.ndarray:
        res = self.predict_proba(x)
        if isinstance(res, pd.Series):  # already what we need
            return res
        if isinstance(res, list):
            res = np.asarray(res)
        res = res.squeeze()
        if len(res.shape) != 1:  # if it has 1 dim, then it is already in the format we need, just not as a series yet
            if len(x) > 1:  # there is more than a single sample to predict on, but gave 2 predictions for each sample (P(0), P(1))
                assert len(res.shape) == 2 and res.shape[1] == 2, f'Invalid result shape for binary classification: {res.shape}'
                res = res[:, 1]
            elif len(res.shape) == 0:  # returned a single value
                pass
            else:
                res = res[:, 1]
        elif len(x) == 1 and len(res) == 2:  # a single sample, but two predictions (P(0), P(1))
            res = res[1]
        predictions = res
        return predictions

    def _output_used_features(self, X_train) -> OrderedDict:
        """
        Go through all the trees and sum up the usage of each feature. Then summarize it in a sorted descending dictionary, from feature
        name to count.
        :param X_train: dataframe with the training data, used for feature names
        :return: oredered dictionary from feature name to usage count
        """
        feature_names = list(X_train.columns)
        features_count = {key: 0 for key in feature_names}
        for estimator in self.estimators_:
            tree_features = np.where(estimator.feature_importances_)[0]
            for n in tree_features:
                features_count[feature_names[n]] += 1
        features_count = {key: value for key, value in features_count.items() if value > 0}
        sorted_x = sorted(features_count.items(), key=lambda kv: kv[1], reverse=True)
        return OrderedDict(sorted_x)

    def _binary_leaf_split_counter(self, X_train, y_train) -> List[Dict[int, List[int]]]:
        """
        A method to count the number of training data samples that end up in each node and the split between the classes.
        Note that this method is only valid for binary case.
        We summarize the results per each tree in a separate dictionary, such that len(output) == num_trees.
        Each dictionary points from leaf number to a list [n_neg, n_pos] such that n_neg is the number of negative samples in this leaf
        and n_pos is the number of positive samples in this leaf. Σ (n_neg+n_pos) in each dict should equal to X_train.shape[0].
        :return: list of dictionaries the length of all trees
        """
        def _summarize_into_dict(r):
            unique_nodes, counts = np.unique(r, return_counts=True)
            d = {k: [0, 0] for k in list(set((abs(unique_nodes))))}
            for node, count in zip(unique_nodes, counts):
                s = (np.sign(node) > 0).astype(int)
                d[abs(node)][s] = count
            return d
        leaves_index = self.apply(X_train)
        f = lambda x: [-1, 1][x]
        y_train_ = np.expand_dims(np.vectorize(f)(y_train.values.astype(int)),  axis=1)  # map False to -1 and adjust dimensions
        leaves_index_with_signs = np.multiply(leaves_index, y_train_)  # multiply with labels to sum the different classes
        return np.apply_along_axis(_summarize_into_dict, 0, leaves_index_with_signs)

    def _extract_uncertainty_of_prediction(self, end_leafs, method='entropy') -> List[Uncertainty]:
        """
        Using the method specified calculate the uncertainty of a prediction that was made
        :param method: Currently we support "entropy" method only
        :return: list of uncertainty objects, each per sample.
        """
        uncertainty = []
        if method == 'entropy':
            for row in end_leafs:  # each row is the result for one sample
                uncertainty.append(calculate_entropy_uncertainties(self._labels, row, self.leafs_content))
        return uncertainty



