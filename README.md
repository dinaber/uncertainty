**uncertainty**
==================
Wrapper classes and other tools to predict model uncertainty. 

## Installation
`git clone git@github.com:dinaber/uncertainty.git`

## Description
In this repository we provide you with easy to use, off the shelf wrapper classes and utils to calculate the uncertainty of your model.
We currently support Random Forest Classifier wrapper, which is sklearn compatible. 

The uncertainty calculation are based on Shannon's entropy and are implemented based on the paper [Shaker MH, Hüllermeier E. Aleatoric and epistemic uncertainty with random forests. InInternational Symposium on Intelligent Data Analysis 2020 Apr 27 (pp. 444-456). Springer, Cham](https://arxiv.org/pdf/2001.00893.pdf).


## Usage
```python
from uncertainty.wrapper import RandomForestClassifierWithUncertainty

rf = RandomForestClassifierWithUncertainty(bootstrap=True, max_depth=6)
rf.fit(X, y)

predictions, uncertainties = rf.predict_proba_with_uncertainty(X_test)
```
