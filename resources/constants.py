# -*- coding: utf-8 -*-
"""
Created on Feb 2022
@author: Dina Berenbaum
"""
from collections import namedtuple


class UncertaintyTypes:
    total = 'total'
    epistemic = 'epistemic'
    aleatoric = 'aleatoric'


Uncertainty = namedtuple('Uncertainty', [UncertaintyTypes.total, UncertaintyTypes.epistemic, UncertaintyTypes.aleatoric])
