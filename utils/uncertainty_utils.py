# -*- coding: utf-8 -*-
"""
Created on Feb 2022
@author: Dina Berenbaum
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from uncertainty.constants import Uncertainty


def calculate_entropy_uncertainties(labels: list, end_leafs: np.ndarray, leafs_split: List[Dict[int, List[int]]]) -> Uncertainty:
    """
    Based on the paper Shaker MH, Hüllermeier E. Aleatoric and epistemic uncertainty with random forests. In International Symposium on
    Intelligent Data Analysis 2020 Apr 27 (pp. 444-456). Springer, Cham. (https://arxiv.org/pdf/2001.00893.pdf)
    We calculate three types of uncertainties:
    1. total
    2. aleatoric (statistical)
    3. epistemic (information related)

    1. This is the **total uncertainty estimation using entropy**.
        For discrete labels this is H[p(y | x)]=−∑y∈Y p(y | x) log2 p(y | x),
        An approximaton for ensemble techniques (and what we calculate here) is:
        −∑y∈Y (1/M * ∑i∈M p(y | hi, x) log2 (1/M * ∑i∈M p(y | hi, x))
    2. The aleatoric uncertainty can be estimated by:
        −(1/M * ∑i∈M ∑y∈Yp(y | hi, x) log2 p(y | hi, x)
    3. The epistemic uncertainty, which is the subtraction of total − aleatoric
    :param labels: a list with all the possible labels
    :param end_leafs: a list with all the leafs our sample ends up in, one per each tree in the ensemble.
    :param leafs_split: a summary of training samples ended up in each leaf and their split between classes. This is a list of dictionaries
    the length of all trees. Each dictionary points from leaf number to a list [n_neg, n_pos] such that n_neg is the number of negative
    samples in this leaf and n_pos is the number of positive samples in this leaf. Σ (n_neg+n_pos) in each dict should equal to
    X_train.shape[0].
    :return: A named tuple with the three uncertainties calculated
    """

    n_labels = len(labels)
    tot_u = 0  # total uncertainty
    al_u = 0  # aleatoric uncertainty
    for label in labels:  # go over the labels
        tot_p = 0
        tot_p_log_p = 0
        for tree_leafs_split, end_leaf in zip(leafs_split, end_leafs):  # go over all the hypotheses (trees)
            # We first want to calculate p(y | hi, x) for each tree, based on the leaf where each sample ends up. In random forest this
            # is the (ni,j(y) + 1) / (ni,j + |Y|)
            p = _calculate_class_conditional_probabilities(label, n_labels, end_leaf, tree_leafs_split)
            tot_p += p
            tot_p_log_p += p * np.log2(p)
        mean_tot_p = tot_p / len(end_leafs)  # get the average over all trees
        mean_tot_p_log_p = tot_p_log_p / len(end_leafs)  # get the average over all trees
        log_mean_tot_p = np.log2(mean_tot_p)
        tot_u += mean_tot_p * log_mean_tot_p
        al_u += mean_tot_p_log_p
    return Uncertainty(-1 * tot_u, -1 * al_u, -1 * (tot_u - al_u))


def _calculate_class_conditional_probabilities(label, n_labels, end_leaf, tree_leafs_split) -> float:
    """
    We calculate p(y | hi, x) for a given label y and a specific model(tree).
    :param label: label number∈[0,1,2,...], corresponds to the index in leaf_split to recover the number of training sample in a leaf
    with this label
    :param n_labels: total number of possible labels (i.e. for binary this would be 2)
    """
    n_y = tree_leafs_split[end_leaf][label]
    n = sum(tree_leafs_split[end_leaf])
    return (n_y + 1) / (n + n_labels)
