# -*- coding: utf-8 -*-
"""
Created on Feb 2022
@author: Dina Berenbaum
"""
from uncertainty.utils.uncertainty_utils import _calculate_class_conditional_probabilities, \
    calculate_entropy_uncertainties
import numpy as np


def test__calculate_class_conditional_probabilities():
    tree_leafs_split = {2: [1, 24], 5: [3, 6], 9: [12, 1]}
    end_leaf = 5
    label = 1
    n_labels = 2
    conditional_p = _calculate_class_conditional_probabilities(label, n_labels, end_leaf, tree_leafs_split)
    assert conditional_p == (6 + 1) / (9 + 2)


def test_calculate_entropy_uncertainties():
    labels = [0, 1]
    end_leafs = [4, 7, 5]
    # min epistemic uncertaiunty:
    leafs_split = [{4: [0, 24], 5: [3, 6], 9: [12, 1]},
                   {7: [0, 24], 5: [3, 6], 9: [12, 1]},
                   {5: [0, 24], 6: [3, 6], 9: [12, 1]}]
    uncertainty1 = calculate_entropy_uncertainties(labels, end_leafs, leafs_split)
    # max epistemic uncertaiunty:
    leafs_split = [{4: [23, 24], 5: [3, 6], 9: [12, 1]},
                   {7: [23, 24], 5: [3, 6], 9: [12, 1]},
                   {5: [23, 24], 6: [3, 6], 9: [12, 1]}]
    uncertainty2 = calculate_entropy_uncertainties(labels, end_leafs, leafs_split)
    assert uncertainty1.epistemic < uncertainty2.epistemic
    assert np.isclose(uncertainty1.aleatoric,  uncertainty2.aleatoric)
    # The epistemic uncertainties are very different between the two cases, but the aleatoric uncertainties are very similar since there
    # is no statistical noise within the ensemble
